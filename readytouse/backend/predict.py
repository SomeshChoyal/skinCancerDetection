import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from backend.model_loader import MODEL, DEVICE, label_encoder

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict(image_path):
    image = preprocess_image(image_path).to(DEVICE)
    with torch.no_grad():
        output = MODEL(image)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return {
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 2)  # % confidence
    }
