# 🔍 AI Image Detector

Deep learning web app that identifies AI-generated images using a fine-tuned ResNet50 CNN. **85.68% test accuracy** on 152K+ images.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)

Upload an image → Get instant AI detection with confidence scores

## ⚡ Features

- 🎯 85.68% accuracy on unseen data
- 🚀 Real-time predictions
- 🎨 Modern gradient UI
- 📊 Confidence breakdowns
- 🔄 Transfer learning from ImageNet

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/AI_Image_Recognition.git
cd AI_Image_Recognition
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 85.68% |
| Dataset Size | 152,710 images |
| Model | ResNet50 (fine-tuned) |
| Training Time | ~2 hours on GPU |

## 🛠️ Tech Stack

**ML**: PyTorch • torchvision • Hugging Face Datasets  
**Web**: Streamlit  
**Training**: Google Colab (GPU)

## 🧠 How It Works

1. **Transfer Learning**: Used pre-trained ResNet50 (ImageNet)
2. **Fine-tuning**: Trained final layer on 152K AI vs Real images
3. **Data Split**: 70% train / 15% validation / 15% test
4. **Optimization**: Adam optimizer, CrossEntropyLoss, 5 epochs

### Training Results

```
Epoch 1: 80.77% → Epoch 5: 84.14% train accuracy
Best validation: 86.18% | Test: 85.68%
```

## 📁 Project Structure

```
├── app.py                    # Streamlit web interface
├── best_model.pth           # Trained model (97.8 MB)
├── AI_Recognition_Project.ipynb  # Training notebook
└── requirements.txt         # Dependencies
```

## 🎓 What I Learned

- Deep learning model training and evaluation
- Transfer learning techniques
- Computer vision preprocessing
- Model deployment with Streamlit
- End-to-end ML project workflow

## ⚠️ Limitations

- Trained on art-style images (may not generalize to all types)
- AI generators evolve rapidly (model may need retraining)
- Demonstration project (not production-ready)

## 🔮 Future Improvements

- [ ] Train on more diverse datasets
- [ ] Try modern architectures (EfficientNet, ViT)
- [ ] Add ensemble methods
- [ ] Implement attention visualization
- [ ] Deploy to Hugging Face Spaces

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

## 🙏 Credits

- Dataset: [Hemg/AI-Generated-vs-Real-Images-Datasets](https://huggingface.co/datasets/Hemg/AI-Generated-vs-Real-Images-Datasets)
- Model: PyTorch ResNet50
- Platform: Google Colab

## 👤 Author

**Your Name**  
[GitHub](https://github.com/yourusername) • [LinkedIn](https://linkedin.com/in/yourprofile) • [Portfolio](https://your-website.com)

---

⭐ Star this repo if you found it helpful!
