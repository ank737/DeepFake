import torch
import torch.nn.functional as F
from pathlib import Path

from unet_model import UNet
from hybrid_gcn_model import HybridGCN

# Paths
DATASET_ROOT = Path("/home/rag/datasets/FaceForensicsPP")
IMAGE_ROOT = DATASET_ROOT / "frames_advo"

UNET_PATH = "/home/rag/Code/unet_trained.pth"
GCN_PATH = "/home/rag/Code/hybrid_gcn_trained.pth"

device = torch.device("cpu")

# Load U-Net
unet = UNet(in_channels=1, out_channels=1).to(device)
unet.load_state_dict(torch.load(UNET_PATH, map_location=device))
unet.eval()

# Load Hybrid GCN
gcn = HybridGCN(num_classes=2).to(device)
gcn.load_state_dict(torch.load(GCN_PATH, map_location=device))
gcn.eval()

# Select one ADVO-processed image
image_path = next((IMAGE_ROOT / "real").glob("*/*.jpg"))

# Load image
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

image = transform(Image.open(image_path)).unsqueeze(0).to(device)

# U-Net edge prediction
with torch.no_grad():
    edge_output = unet(image)
    edge_prediction = torch.sigmoid(edge_output)

# Hybrid GCN classification
# Use the U-Net edge prediction as the classifier input
with torch.no_grad():
    output = gcn(edge_prediction)
    probabilities = F.softmax(output, dim=1)
    predicted_class = output.argmax(dim=1).item()

print("\n========== INTEGRATED PIPELINE RESULTS ==========")
print("Input image:", image_path)
print("U-Net output shape:", edge_prediction.shape)
print("Real probability:", probabilities[0, 0].item())
print("Fake probability:", probabilities[0, 1].item())
print("Prediction:", "Fake" if predicted_class == 1 else "Real")