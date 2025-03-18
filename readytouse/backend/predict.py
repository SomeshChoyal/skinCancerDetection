import torch
import torchvision.transforms as transforms
from PIL import Image
from backend.model_loader import MODEL, DEVICE, label_encoder

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Predict Function
def predict(image_path):
    image = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        output = MODEL(image)
        predicted_class = output.argmax(dim=1).item()
    
    # Convert numeric prediction to class name
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label
