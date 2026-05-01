import streamlit as st
import torch
import cv2
import numpy as np
import importlib.util
import sys
import os

# --- Page Config & Aesthetics ---
st.set_page_config(
    page_title="EU-2Net Interactive Tester",
    page_icon="🧬",
    layout="wide"
)

# Custom CSS for modern premium feel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 600;
        margin-bottom: 20px;
    }
    
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 40px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #3b82f6, #8b5cf6);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5);
    }
    
    /* Ensure markdown text in columns adopts the overall font and color */
    .stMarkdown p {
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title Let's wow the user!
st.markdown("<h1>🌟 EU-2Net Model Evaluator</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload medical images and instantly generate precise segmentation masks using your trained EU-2Net architecture.</p>", unsafe_allow_html=True)


# --- Load Model Logic ---
@st.cache_resource
def load_model():
    """Load the model architecture from the local file and the weights."""
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "eu2net_best_model.pth")
    script_path = os.path.join(base_dir, "udiat-busi-improv.py")
    
    if not os.path.exists(model_path):
        return None, f"Could not find model file: {model_path}"
    if not os.path.exists(script_path):
        return None, f"Could not find architecture script: {script_path}"
        
    try:
        # Dynamically load the architecture script
        spec = importlib.util.spec_from_file_location("udiat", script_path)
        udiat = importlib.util.module_from_spec(spec)
        sys.modules["udiat"] = udiat
        spec.loader.exec_module(udiat)
        
        IMG_HEIGHT = udiat.IMG_HEIGHT
        IMG_WIDTH = udiat.IMG_WIDTH
        IMG_CHANNELS = udiat.IMG_CHANNELS
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else 'cpu'))
        
        # Instantiate Model
        input_shape = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
        model = udiat.build_u2net(input_shape).to(DEVICE)
        
        # Load weights (handle DataParallel 'module.' prefix if present)
        state = torch.load(model_path, map_location=DEVICE)
        if isinstance(state, dict) and "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
        # strip 'module.' if needed
        new_state = {}
        for k, v in state_dict.items():
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_state[new_key] = v
        model.load_state_dict(new_state, strict=False)
        model.eval()
        
        return {
            "model": model, 
            "device": DEVICE, 
            "h": IMG_HEIGHT, 
            "w": IMG_WIDTH, 
            "c": IMG_CHANNELS
        }, None
    except Exception as e:
        return None, str(e)


# Load model state
with st.spinner("Initializing Deep Learning Engine..."):
    model_data, err = load_model()

if err:
    st.error(f"Failed to load the model: {err}")
else:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.success(f"✅ Model loaded successfully and running on: **{model_data['device']}**")
    
    uploaded_file = st.file_uploader("Upload an Image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # read as color for display
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("### 🔬 Inference Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
        with st.spinner("Processing image through EU-2Net..."):
            # Preprocess for model
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (model_data['w'], model_data['h']))
            norm_img = (resized_img / 255.0).astype(np.float32)

            # Convert to tensor: (1, 1, H, W)
            tensor_img = torch.from_numpy(norm_img).unsqueeze(0).unsqueeze(0).to(model_data['device'])
            
            # Predict
            with torch.no_grad():
                device = model_data['device']
                is_cuda = getattr(device, "type", str(device)).startswith("cuda")
                if is_cuda:
                    ctx = torch.autocast(device_type="cuda", enabled=True)
                else:
                    # no-op context manager when autocast not enabled
                    from contextlib import nullcontext
                    ctx = nullcontext()
                with ctx:
                    pred_logits = model_data['model'](tensor_img)
                    pred_mask = (torch.sigmoid(pred_logits) > 0.5).float()
            
            # Postprocess back to numpy
            mask_np = pred_mask.squeeze().cpu().numpy()
            mask_display = (mask_np * 255).astype(np.uint8)
            
            # Resize back to original image size for display
            original_h, original_w = image.shape[:2]
            mask_display_resized = cv2.resize(mask_display, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
            
            # Colored Overlay (blend properly)
            overlay = image.astype(np.float32).copy()
            color = np.array([246, 92, 139], dtype=np.float32)  # BGR
            mask_bool = mask_display_resized > 127
            overlay[mask_bool] = (overlay[mask_bool] * 0.5) + (color * 0.5)
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
        with col2:
            st.markdown("#### AI Segmentation Mask")
            st.image(mask_display_resized, use_container_width=True, clamp=True)
            
        with col3:
            st.markdown("#### Overlay")
            st.image(cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)