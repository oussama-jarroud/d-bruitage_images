# %% [code]
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy

# %% [code]
# Cell 2: Setup Hyperparameters and Data Paths (MODIFIED)
# !!! This now points to your specific BUSI dataset structure !!!
#################################################################

# --- Training Hyperparameters ---
IMG_WIDTH = 256        # Reduced from 512 to fit Mac MPS memory
IMG_HEIGHT = 256       # Reduced from 512 to fit Mac MPS memory
IMG_CHANNELS = 1
BATCH_SIZE = 2         # Reduced from 4 to avoid MPS OOM
EPOCHS = 50     # As mentioned in the paper
LR = 1e-4       # As mentioned in the paper

# --- UPDATED PATHS ---
# Root directory from your image
DATA_ROOT = "./Dataset_BUSI_with_GT"

# Classes to use for segmentation
CLASSES_TO_USE = ["benign", "malignant", "normal"]

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# %% [code]
#################################################################
# Cell 3: Define Custom Losses and Metrics (CORRECTED)
# Uses BCEWithLogitsLoss for autocast safety
#################################################################

class CombinedLoss(nn.Module):
    """
    Combines Dice loss and Binary Cross-Entropy (BCE) from logits.
    """
    def __init__(self, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        # Use BCEWithLogitsLoss for numerical stability and autocast safety
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def dice_loss(self, y_pred_logits, y_true):
        # Apply sigmoid to logits to get probabilities for Dice
        y_pred = torch.sigmoid(y_pred_logits)
        y_pred_f = y_pred.view(-1)
        y_true_f = y_true.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        return 1.0 - (2. * intersection + self.smooth) / (y_pred_f.sum() + y_true_f.sum() + self.smooth)

    def forward(self, y_pred_logits, y_true):
        # y_pred_logits is the raw output from the model (no sigmoid)
        bce = self.bce_loss(y_pred_logits, y_true)
        dice = self.dice_loss(y_pred_logits, y_true)
        return bce + dice

def dice_score(y_pred_logits, y_true, smooth=1e-6):
    """Calculates the Dice score from logits."""
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float()
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)
    intersection = (y_pred_f * y_true_f).sum()
    return (2. * intersection + smooth) / (y_pred_f.sum() + y_true_f.sum() + smooth)

def iou_score(y_pred_logits, y_true, smooth=1e-6):
    """Calculates the Intersection over Union (IoU) metric from logits."""
    y_pred = (torch.sigmoid(y_pred_logits) > 0.5).float()
    y_pred_f = y_pred.view(-1)
    y_true_f = y_true.view(-1)
    intersection = (y_pred_f * y_true_f).sum()
    union = y_pred_f.sum() + y_true_f.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# %% [code]
#################################################################
# Cell 4: Model Definition (PyTorch Conversion) (CORRECTED)
# Removed all sigmoid activations from side outputs
#################################################################

class SeparableConv2d(nn.Module):
    """
    Equivalent to Keras' SeparableConv2D
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super(SeparableConv2d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, 
            padding=padding, dilation=dilation, groups=in_channels, bias=False
        ) 
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# %% [code]

class Wavelet(nn.Module):
    """
    [cite_start]Wavelet module using Sobel filter to extract edge features [cite: 19-26].
    """
    def __init__(self, in_channels, order_name):
        super(Wavelet, self).__init__()
        self.in_channels = in_channels
        
        # Sobel filter for X direction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x = nn.Parameter(sobel_x.repeat(in_channels, 1, 1, 1), requires_grad=False)
        
        # Sobel filter for Y direction
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = nn.Parameter(sobel_y.repeat(in_channels, 1, 1, 1), requires_grad=False)
        
        self.conv1 = nn.Conv2d(in_channels * 2, in_channels * 2, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels * 2, in_channels, 1, 1, padding="same")

    def forward(self, x):
        wav1 = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        wav2 = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        wav = torch.cat([wav1, wav2], dim=1)
        wav = self.conv1(wav)
        wav = self.conv2(wav)
        return wav

# %% [code]
class SE(nn.Module):
    """
    [cite_start]Squeeze-and-Excitation (SE) block [cite: 28-36].
    """
    def __init__(self, in_ch, out_ch=32, reduction=8):
        super(SE, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding="same") 
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense1 = nn.Linear(out_ch, out_ch // reduction) 
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(out_ch // reduction, out_ch) 
        self.sigmoid = nn.Sigmoid()
        self.out_ch = out_ch

    def forward(self, x):
        x_in = self.conv(x) 
        x1 = self.pool(x_in)
        x1 = x1.reshape(x1.size(0), -1) 
        x1 = self.dense1(x1)
        x1 = self.relu(x1)
        x1 = self.dense2(x1)
        x1 = self.sigmoid(x1)
        x1 = x1.reshape(x1.size(0), self.out_ch, 1, 1) 
        return x_in * x1 

# %% [code]
class FSiAM(nn.Module):
    """
    [cite_start]Feature Similarity-Based Attention Module (FSiAM) [cite: 38-46].
    """
    def __init__(self):
        super(FSiAM, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input x is (B, HW, C)
        x1 = x.permute(0, 2, 1) # B, C, HW
        x2 = torch.matmul(x1, x) # B, C, C
        x2 = self.pool(x2).squeeze(-1) # B, C
        x2 = self.sigmoid(x2)
        x2 = x2.unsqueeze(1).unsqueeze(1) # B, 1, 1, C
        return x2

# %% [code]
class EnsembleLayer(nn.Module):
    """
    [cite_start]Learnable weighted averaging ensemble layer [cite: 48-73].
    This layer now averages logits, and the output is a logit.
    """
    def __init__(self):
        super(EnsembleLayer, self).__init__()
        self.weights_variable = nn.Parameter(torch.ones(6, dtype=torch.float32), requires_grad=True)

    def forward(self, inputs):
        # inputs are list of 6 logit tensors, each (B, 1, H, W)
        y1, y2, y3, y4, y5, y6 = inputs
        
        # Stack on a new dimension (dim=-1)
        stacked_masks = torch.stack([y1, y2, y3, y4, y5, y6], dim=-1) # Shape: (B, 1, H, W, 6)
        
        # Get normalized weights
        normalized_weights = F.softmax(self.weights_variable, dim=0) # Shape: (6)
        
        # Reshape weights for broadcasting
        weights = normalized_weights.view(1, 1, 1, 1, 6) # Shape: (1, 1, 1, 1, 6)
        
        # Perform weighted sum on logits
        final_logit = torch.sum(stacked_masks * weights, dim=-1) # Shape: (B, 1, H, W)
        return final_logit

# %% [code]
class ConvBlock(nn.Module):
    """
    [cite_start]conv_block: SeparableConv2D -> BN -> ReLU [cite: 75-80].
    """
    def __init__(self, in_ch, out_ch, rate=1): 
        super(ConvBlock, self).__init__()
        padding = rate 
        self.conv = SeparableConv2d(in_ch, out_ch, 3, padding=padding, dilation=rate) 
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# %% [code]
class RSU_L(nn.Module):
    """
    [cite_start]RSU-L block [cite: 82-120].
    """
    def __init__(self, in_ch, out_ch, int_ch, num_layers):
        super(RSU_L, self).__init__()
        self.init_conv = ConvBlock(in_ch, out_ch, rate=1) 
        
        # Encoder
        self.encoders = nn.ModuleList()
        self.encoders.append(ConvBlock(out_ch, int_ch, rate=1)) 
        for _ in range(num_layers - 2):
            self.encoders.append(nn.Sequential(
                nn.MaxPool2d(2, 2),
                ConvBlock(int_ch, int_ch, rate=1) 
            ))
            
        # Bridge
        self.bridge = ConvBlock(int_ch, int_ch, rate=2) 
        
        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        self.decoders.append(ConvBlock(int_ch * 2, int_ch, rate=1))
        
        for _ in range(num_layers - 3):
            self.upsamplers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            self.decoders.append(ConvBlock(int_ch * 2, int_ch, rate=1))
            
        self.upsamplers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.final_decode = ConvBlock(int_ch * 2, out_ch, rate=1)


    def forward(self, x):
        init_feats = self.init_conv(x)
        
        skip = []
        enc = init_feats
        for encoder_block in self.encoders:
            enc = encoder_block(enc)
            skip.append(enc)
            
        bridge = self.bridge(enc)
        
        dec = bridge
        skip.reverse() 
        
        dec = torch.cat([dec, skip[0]], dim=1)
        dec = self.decoders[0](dec)
        
        for i in range(len(self.upsamplers) - 1): 
            dec = self.upsamplers[i](dec)
            dec = torch.cat([dec, skip[i+1]], dim=1)
            dec = self.decoders[i+1](dec)
            
        dec = self.upsamplers[-1](dec)
        dec = torch.cat([dec, skip[-1]], dim=1)
        dec = self.final_decode(dec)
        
        return dec + init_feats

# %% [code]
class RSU_4F(nn.Module):
    """
    [cite_start]RSU-4F block [cite: 122-149].
    """
    def __init__(self, in_ch, out_ch, int_ch):
        super(RSU_4F, self).__init__()
        self.x0 = ConvBlock(in_ch, out_ch, rate=1)
        
        self.x1 = ConvBlock(out_ch, int_ch, rate=1)
        self.x2 = ConvBlock(int_ch, int_ch, rate=2)
        self.x3 = ConvBlock(int_ch, int_ch, rate=4)
        
        self.x4 = ConvBlock(int_ch, int_ch, rate=8)
        
        self.d3 = ConvBlock(int_ch * 2, int_ch, rate=4)
        self.d2 = ConvBlock(int_ch * 2, int_ch, rate=2)
        self.d1 = ConvBlock(int_ch * 2, out_ch, rate=1)

    def forward(self, x):
        x0 = self.x0(x)
        
        x1 = self.x1(x0)
        x2 = self.x2(x1)
        x3 = self.x3(x2)
        x4 = self.x4(x3)
        
        x = torch.cat([x4, x3], dim=1)
        x = self.d3(x)
        
        x = torch.cat([x, x2], dim=1)
        x = self.d2(x)
        
        x = torch.cat([x, x1], dim=1)
        x = self.d1(x)
        
        return x + x0

# %% [code]
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super(AttentionGate, self).__init__()
        # Project gating and skip features to intermediate dimension F_int
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_x, F_int, kernel_size=1), nn.BatchNorm2d(F_int))
        # Combine and activate
        self.psi = nn.Sequential(nn.ReLU(),
                                  nn.Conv2d(F_int, 1, kernel_size=1),
                                  nn.Sigmoid())
    def forward(self, x_skip, g):
        # Align gating signal size to skip if necessary (assumed pre-upsampled)
        g_proj = self.W_g(g)
        x_proj = self.W_x(x_skip)
        attn = self.psi(g_proj + x_proj)       # shape: (B,1,H,W)
        return x_skip * attn                   # apply spatial attention mask

# %% [code]
class U2Net(nn.Module):
    """
    [cite_start]The main EU-2Net model [cite: 151-248].
    *** THIS IS THE FIXED BLOCK ***
    Removed sigmoid from all side outputs (y1-y6).
    """
    def __init__(self, input_shape, out_ch, int_ch, num_classes=1):
        super(U2Net, self).__init__()
        H, W, C = input_shape
        
        self.pool = nn.MaxPool2d(2, 2)
        self.up = lambda scale: nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)
        
        # Encoder
        self.s1 = RSU_L(C, out_ch[0], int_ch[0], 7)
        self.s2 = RSU_L(out_ch[0], out_ch[1], int_ch[1], 6)
        self.s3 = RSU_L(out_ch[1], out_ch[2], int_ch[2], 5)
        self.s4 = RSU_L(out_ch[2], out_ch[3], int_ch[3], 4)
        self.s5 = RSU_4F(out_ch[3], out_ch[4], int_ch[4])
        
        # Bridge
        self.b1 = RSU_4F(out_ch[4], out_ch[5], int_ch[5])
        self.b2_up = self.up(2)
        
        # Decoders and A2TF2 fusion blocks
        # Decoder D1 (between s5 and b2)
        self.se_d1 = SE(in_ch=out_ch[5])
        self.se_d1_proj = nn.Conv2d(32, out_ch[4], kernel_size=1)   # project SE output to match skip s5
        self.attn_s5 = AttentionGate(F_g=out_ch[5], F_x=out_ch[4], F_int=out_ch[4] // 2)
        self.fsiam_d1 = FSiAM()
        self.wavelet_d1 = Wavelet(out_ch[4], "edge_1")
        self.fuse_weights_d1 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.d1_conv1 = SeparableConv2d(out_ch[4], 512, 3, padding=1)  # after fusion, all tensors are out_ch[4]
        self.d1_conv2 = SeparableConv2d(512, 256, 1)
        self.d1_rsu = RSU_4F(256, out_ch[6], int_ch[6])
        self.u1_up = self.up(2)

        # Decoder D2 (skip: s4, decoder input: u1)
        self.se_d2 = SE(in_ch=out_ch[6])
        self.se_d2_proj = nn.Conv2d(32, out_ch[3], kernel_size=1)  # project SE output to match skip s4
        
        self.attn_s4 = AttentionGate(F_g=out_ch[6], F_x=out_ch[3], F_int=out_ch[3] // 2)
        self.fsiam_d2 = FSiAM()
        self.wavelet_d2 = Wavelet(out_ch[3], "edge_2")
        self.fuse_weights_d2 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.d2_conv1 = SeparableConv2d(out_ch[3], 256, 3, padding=1)  # input after fusion
        self.d2_conv2 = SeparableConv2d(256, 128, 1)
        self.d2_rsu = RSU_L(128, out_ch[7], int_ch[7], 4)
        self.u2_up = self.up(2)

        #decode D3
        
        self.se_d3 = SE(in_ch=out_ch[7])
        self.se_d3_proj = nn.Conv2d(32, out_ch[2], 1)
        self.fsiam_d3 = FSiAM()
        self.wavelet_d3 = Wavelet(out_ch[2], "edge_3")
        self.attn_s3 = AttentionGate(F_g=out_ch[7], F_x=out_ch[2], F_int=out_ch[2] // 2)
        self.fuse_weights_d3 = nn.Parameter(torch.ones(3), requires_grad=True)
        self.d3_conv1 = SeparableConv2d(out_ch[2], 128, 3, padding=1)
        self.d3_conv2 = SeparableConv2d(128, 64, 1)
        self.d3_rsu = RSU_L(64, out_ch[8], int_ch[8], 5)
        self.u3_up = self.up(2)


        # Decoder D4 (skip: s2, decoder input: u3)
        self.se_d4 = SE(in_ch=out_ch[8])
        self.se_d4_proj = nn.Conv2d(32, out_ch[1], kernel_size=1)  # project SE output to match skip s2
        
        self.attn_s2 = AttentionGate(F_g=out_ch[8], F_x=out_ch[1], F_int=out_ch[1] // 2)
        self.fsiam_d4 = FSiAM()
        self.wavelet_d4 = Wavelet(out_ch[1], "edge_4")
        self.fuse_weights_d4 = nn.Parameter(torch.ones(3), requires_grad=True)
        
        self.d4_conv1 = SeparableConv2d(out_ch[1], 64, 3, padding=1)  # input: fused (channels=out_ch[1])
        self.d4_conv2 = SeparableConv2d(64, 32, 1)
        self.d4_rsu = RSU_L(32, out_ch[9], int_ch[9], 6)
        
        self.u4_up = self.up(2)

        
        # Decoder D5 (skip: s1, decoder input: u4)
        self.se_d5 = SE(in_ch=out_ch[9])
        self.se_d5_proj = nn.Conv2d(32, out_ch[0], kernel_size=1)  # project SE output to match skip s1
        
        self.attn_s1 = AttentionGate(F_g=out_ch[9], F_x=out_ch[0], F_int=out_ch[0] // 2)
        self.fsiam_d5 = FSiAM()
        self.wavelet_d5 = Wavelet(out_ch[0], "edge_5")
        self.fuse_weights_d5 = nn.Parameter(torch.ones(3), requires_grad=True)
        
        self.d5_conv1 = SeparableConv2d(out_ch[0], 32, 3, padding=1)
        self.d5_conv2 = SeparableConv2d(32, 16, 1)
        self.d5_rsu = RSU_L(16, out_ch[10], int_ch[10], 7)


        
        # Side Outputs (logits)
        self.y1_conv = nn.Conv2d(out_ch[10], num_classes, 3, padding="same")
        self.y2_conv = nn.Conv2d(out_ch[9], num_classes, 3, padding="same")
        self.y3_conv = nn.Conv2d(out_ch[8], num_classes, 3, padding="same")
        self.y4_conv = nn.Conv2d(out_ch[7], num_classes, 3, padding="same")
        self.y5_conv = nn.Conv2d(out_ch[6], num_classes, 3, padding="same")
        self.y6_conv = nn.Conv2d(out_ch[5], num_classes, 3, padding="same")
        
        self.y_up2 = self.up(2)
        self.y_up4 = self.up(4)
        self.y_up8 = self.up(8)
        self.y_up16 = self.up(16)
        self.y_up32 = self.up(32)
        
        # self.sigmoid = nn.Sigmoid() # <-- Removed
        self.ensemble = EnsembleLayer()
        
        self.reshape_s1 = lambda x: x.reshape(x.size(0), -1, x.size(3)) 
        self.reshape_s2 = lambda x: x.reshape(x.size(0), -1, x.size(3)) 
        self.reshape_s3 = lambda x: x.reshape(x.size(0), -1, x.size(3)) 
        self.reshape_s4 = lambda x: x.reshape(x.size(0), -1, x.size(3)) 
        self.reshape_s5 = lambda x: x.reshape(x.size(0), -1, x.size(3)) 
        
    def forward(self, inputs):
        # Encoder
        s1 = self.s1(inputs) 
        p1 = self.pool(s1)
        s2 = self.s2(p1) 
        p2 = self.pool(s2)
        s3 = self.s3(p2) 
        p3 = self.pool(s3)
        s4 = self.s4(p3) 
        p4 = self.pool(s4)
        s5 = self.s5(p4) 
        p5 = self.pool(s5)
        
        # Bridge
        b1 = self.b1(p5) 
        b2 = self.b2_up(b1) 
        
        # --- Decoder ---
        # A2TF2 Block 1
        s5 = self.attn_s5(s5, b2)
        x_s5 = self.reshape_s5(s5.permute(0, 2, 3, 1))
        fsiam = self.fsiam_d1(x_s5)
        gate = torch.mean(b2, dim=(2, 3), keepdim=True)
        fsiam = fsiam * gate.permute(0, 2, 3, 1)
        fsiam = s5 * fsiam.permute(0, 3, 1, 2)
        fft = self.wavelet_d1(s5)
        se = self.se_d1_proj(self.se_d1(b2))
        w = F.softmax(self.fuse_weights_d1, dim=0)
        fused = w[0]*se + w[1]*fsiam + w[2]*fft
        d1 = self.d1_conv1(fused) 
        d1 = self.d1_conv2(d1) 
        d1 = self.d1_rsu(d1)   
        u1 = self.u1_up(d1)    
        
        # A2TF2 Block 2
        # Example for block 2:
        s4 = self.attn_s4(s4, u1)
        x_s4 = self.reshape_s4(s4.permute(0, 2, 3, 1))
        fsiam = self.fsiam_d2(x_s4)
        gate = torch.mean(u1, dim=(2, 3), keepdim=True)
        fsiam = fsiam * gate.permute(0, 2, 3, 1)
        fsiam = s4 * fsiam.permute(0, 3, 1, 2)
        
        fft = self.wavelet_d2(s4)
        se = self.se_d2_proj(self.se_d2(u1))
        
        w = F.softmax(self.fuse_weights_d2, dim=0)
        fused = w[0]*se + w[1]*fsiam + w[2]*fft
        
        d2 = self.d2_conv1(fused)
        d2 = self.d2_conv2(d2) 
        d2 = self.d2_rsu(d2)   
        u2 = self.u2_up(d2)    

        # A2TF2 Block 3
        s3 = self.attn_s3(s3, u2)
        x_s3 = self.reshape_s3(s3.permute(0, 2, 3, 1))
        fsiam = self.fsiam_d3(x_s3)
        gate = torch.mean(u2, dim=(2, 3), keepdim=True)
        fsiam = fsiam * gate.permute(0, 2, 3, 1)
        fsiam = s3 * fsiam.permute(0, 3, 1, 2)
        
        fft = self.wavelet_d3(s3)
        se = self.se_d3_proj(self.se_d3(u2))
        
        w = F.softmax(self.fuse_weights_d3, dim=0)
        fused = w[0]*se + w[1]*fsiam + w[2]*fft
        
        d3 = self.d3_conv1(fused)
        d3 = self.d3_conv2(d3) 
        d3 = self.d3_rsu(d3)   
        u3 = self.u3_up(d3)    
      
        # A2TF2 Block 4 (Enhanced)
        s2 = self.attn_s2(s2, u3)
        x_s2 = self.reshape_s2(s2.permute(0, 2, 3, 1))
        fsiam = self.fsiam_d4(x_s2)
        gate = torch.mean(u3, dim=(2, 3), keepdim=True)
        fsiam = fsiam * gate.permute(0, 2, 3, 1)
        fsiam = s2 * fsiam.permute(0, 3, 1, 2)
        
        fft = self.wavelet_d4(s2)
        se = self.se_d4_proj(self.se_d4(u3))
        
        w = F.softmax(self.fuse_weights_d4, dim=0)
        fused = w[0] * se + w[1] * fsiam + w[2] * fft
        
        d4 = self.d4_conv1(fused)
        d4 = self.d4_conv2(d4)
        d4 = self.d4_rsu(d4)
        u4 = self.u4_up(d4)
    
        
        # A2TF2 Block 5 (Enhanced)
        s1 = self.attn_s1(s1, u4)
        x_s1 = self.reshape_s1(s1.permute(0, 2, 3, 1))
        fsiam = self.fsiam_d5(x_s1)
        gate = torch.mean(u4, dim=(2, 3), keepdim=True)
        fsiam = fsiam * gate.permute(0, 2, 3, 1)
        fsiam = s1 * fsiam.permute(0, 3, 1, 2)
        
        fft = self.wavelet_d5(s1)
        se = self.se_d5_proj(self.se_d5(u4))
        
        w = F.softmax(self.fuse_weights_d5, dim=0)
        fused = w[0] * se + w[1] * fsiam + w[2] * fft
        
        d5 = self.d5_conv1(fused)
        d5 = self.d5_conv2(d5)
        d5 = self.d5_rsu(d5)
 
        
        # Side Outputs (NO SIGMOID)
        y1 = self.y1_conv(d5)
        y2 = self.y_up2(self.y2_conv(d4))
        y3 = self.y_up4(self.y3_conv(d3))
        y4 = self.y_up8(self.y4_conv(d2))
        y5 = self.y_up16(self.y5_conv(d1))
        y6 = self.y_up32(self.y6_conv(b1))
        
        # Ensemble
        y0_logit = self.ensemble([y1, y2, y3, y4, y5, y6]) # y0 is (B, 1, H, W) logit
        
        return y0_logit


def build_u2net(input_shape, num_classes=1):
    """
    Helper function to build the EU-2Net model.
    """
    out_ch = [64, 128, 256, 512, 512, 512, 512, 256, 128, 64, 64]
    int_ch = [32, 32, 64, 128, 256, 256, 256, 128, 64, 32, 16]
    model = U2Net(input_shape, out_ch, int_ch, num_classes=num_classes)
    return model

# %% [code]


# %% [code]
# Cell 5: Data Loading and Preprocessing (MODIFIED)
# This now accepts lists of file paths, which is more flexible.
#################################################################

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
        
        # Normalize and Binarize
        image = (image / 255.0).astype(np.float32)
        mask = (mask > 127).astype(np.float32) # Binarize
        
        # Add channel dimension
        image = np.expand_dims(image, axis=0) # (1, H, W)
        mask = np.expand_dims(mask, axis=0)   # (1, H, W)

        if self.transform:
            # You can add augmentations here
            pass
            
        return torch.from_numpy(image), torch.from_numpy(mask)

# %% [code]
#################################################################
# Cell 6: Training and Evaluation Functions (CORRECTED)
# Applies sigmoid to predictions before metric calculation
#################################################################

def train_one_epoch(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    
    for (data, targets) in loop:
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        
        # Forward
        with torch.autocast(device_type=DEVICE if DEVICE != "mps" else "cpu", enabled=DEVICE=="cuda"):
            # predictions are logits
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update loop metrics
        epoch_loss += loss.item()
        
        # Calculate metrics (from logits)
        epoch_dice += dice_score(predictions, targets).item()
        epoch_iou += iou_score(predictions, targets).item()
        
        loop.set_postfix(loss=loss.item())
        
    avg_loss = epoch_loss / len(loader)
    avg_dice = epoch_dice / len(loader)
    avg_iou = epoch_iou / len(loader)
    
    print(f"Train Loss: {avg_loss:.4f}, Train Dice: {avg_dice:.4f}, Train IoU: {avg_iou:.4f}")
    return avg_loss, avg_dice

def evaluate(loader, model, loss_fn):
    model.eval()
    loop = tqdm(loader, desc="Validation")
    
    val_loss = 0.0
    val_dice = 0.0
    val_iou = 0.0
    
    with torch.no_grad():
        for (data, targets) in loop:
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            
            # predictions are logits
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            val_loss += loss.item()
            
            # Calculate metrics (from logits)
            val_dice += dice_score(predictions, targets).item()
            val_iou += iou_score(predictions, targets).item()
            
    avg_loss = val_loss / len(loader)
    avg_dice = val_dice / len(loader)
    avg_iou = val_iou / len(loader)
    
    print(f"Val Loss:   {avg_loss:.4f}, Val Dice:   {avg_dice:.4f}, Val IoU:   {avg_iou:.4f}")
    return avg_loss, avg_dice

# %% [code]
# Cell 7: Main Training Script (MODIFIED)
# This now scans your dataset structure to build file lists
#################################################################

if __name__ == '__main__':
    print("PyTorch EU-2Net Training Script")
    print(f"Using device: {DEVICE}")

    try:
        # 1. Load Data Paths
        print("Scanning data directories...")
        all_image_paths = []
        all_mask_paths = []

        for class_name in CLASSES_TO_USE:
            class_dir = os.path.join(DATA_ROOT, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found, skipping: {class_dir}")
                continue
        
            # Find all image/mask pairs
            files = sorted(os.listdir(class_dir))
            for f in files:
                # Find original images — skip masks and secondary masks (e.g. _mask_1.png)
                if f.endswith(".png") and "_mask" not in f:
                    img_path = os.path.join(class_dir, f)
                
                    # Construct corresponding primary mask path (e.g., "benign (1)_mask.png")
                    mask_path = os.path.join(class_dir, f.replace(".png", "_mask.png"))
                
                    if os.path.exists(mask_path):
                        all_image_paths.append(img_path)
                        all_mask_paths.append(mask_path)
    
        if not all_image_paths:
            raise FileNotFoundError(f"No matching image/mask pairs found in {DATA_ROOT} under {CLASSES_TO_USE}")
    
        print(f"Found {len(all_image_paths)} image/mask pairs.")

        # Split: 80% train, 20% validation
        img_train, img_val, mask_train, mask_val = train_test_split(
            all_image_paths, all_mask_paths, test_size=0.2, random_state=42
        )
    
        # Create custom datasets
        train_ds = SegmentationDataset(img_train, mask_train)
        val_ds = SegmentationDataset(img_val, mask_val)

        print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # 0 required on macOS
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # 2. Build Model
        print("Building model...")
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        model = build_u2net(input_shape).to(DEVICE)
    
        # 3. Define Loss, Optimizer, Scaler
        loss_fn = CombinedLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler('cuda', enabled=DEVICE=="cuda")
    
        # 4. Training Loop
        best_val_dice = -1.0
        best_model_wts = None
    
        for epoch in range(EPOCHS):
            print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
            train_loss, train_dice = train_one_epoch(train_loader, model, optimizer, loss_fn, scaler)
            val_loss, val_dice = evaluate(val_loader, model, loss_fn)
        
            # Save the best model
            if val_dice > best_val_dice:
                best_val_dice = val_dice
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "eu2net_best_model.pth")
                print(f"New best model saved! (Val Dice: {val_dice:.4f})")
            
        print("\nTraining complete.")
        print(f"Best validation Dice score: {best_val_dice:.4f}")
        print("Best model weights saved to 'eu2net_best_model.pth'")

    except FileNotFoundError as e:
        print("="*50)
        print(f"ERROR: {e}")
        print(f"Please check your paths. Tried to load from: {DATA_ROOT}")
        print("Update the path in 'Cell 2' to match your dataset location.")
        print("="*50)
    except Exception as e:
        print(f"\nAn error occurred: {e}")