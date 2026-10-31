import torch
import json
from pathlib import Path
from collections import defaultdict
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from simple_cnn_model import SimpleCNN

ROOT = Path("/home/rag/datasets/FaceForensicsPP")
IMAGE_ROOT = ROOT / "frames_advo"
SPLIT_FILE = IMAGE_ROOT / "splits_advo.json"

device = torch.device("cpu")

# Load model
model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("advo_cnn_trained.pth", map_location=device)
)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

with open(SPLIT_FILE) as f:
    splits = json.load(f)

video_actual = []
video_predicted = []

# Evaluate test videos
for video in splits["test"]:

    predictions = []

    for frame in video["frames"]:

        image_path = Path(frame)

        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            predictions.append(probabilities[0, 1].item())

    # Average fake probability across video frames
    average_fake_probability = sum(predictions) / len(predictions)

    predicted_label = (
        1 if average_fake_probability >= 0.5 else 0
    )

    actual_label = 0 if video["label"] == "real" else 1

    video_actual.append(actual_label)
    video_predicted.append(predicted_label)

print("\n========== VIDEO-LEVEL RESULTS ==========")
print("Test videos:", len(video_actual))

print(
    "Accuracy:",
    accuracy_score(video_actual, video_predicted)
)

print("\nClassification Report:")
print(classification_report(
    video_actual,
    video_predicted,
    target_names=["Real", "Fake"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(video_actual, video_predicted))