import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Set page config
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        background-color: rgba(255, 255, 255, 0.1);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .ai-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    .real-result {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Define transform
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB')),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Header
st.markdown("<h1>🔍 AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload an image to detect if it's AI-generated or real art</p>", unsafe_allow_html=True)

# Create columns for better layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # File uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Make prediction
        with st.spinner('🔄 Analyzing image...'):
            # Preprocess
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
            ai_prob = probabilities[0].item() * 100
            real_prob = probabilities[1].item() * 100
            
        # Display results with nice styling
        st.markdown("---")
        
        if ai_prob > real_prob:
            st.markdown(f"""
                <div class='result-box ai-result'>
                    🤖 AI-GENERATED IMAGE<br>
                    <span style='font-size: 2rem;'>{ai_prob:.1f}%</span> confidence
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='result-box real-result'>
                    ✅ REAL IMAGE<br>
                    <span style='font-size: 2rem;'>{real_prob:.1f}%</span> confidence
                </div>
            """, unsafe_allow_html=True)
        
        # Show probability breakdown
        st.markdown("### 📊 Detailed Breakdown")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("🤖 AI-Generated", f"{ai_prob:.2f}%")
            st.progress(ai_prob / 100)
        
        with col_b:
            st.metric("✅ Real Image", f"{real_prob:.2f}%")
            st.progress(real_prob / 100)
        
        # Info box
        st.info("💡 This model was trained on 152,000+ images and achieves 85.68% accuracy on test data.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: white; padding: 20px;'>
        <p>Built with PyTorch & Streamlit | Model: ResNet50 (Fine-tuned)</p>
        <p style='font-size: 0.9rem;'>⚠️ This is a demonstration project. Results may not be 100% accurate.</p>
    </div>
""", unsafe_allow_html=True)