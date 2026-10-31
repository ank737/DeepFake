import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

from unet_model import UNet

# Paths
ROOT = Path("/home/rag/datasets/FaceForensicsPP")
IMAGE_ROOT = ROOT / "frames_advo"
OUTPUT_ROOT = ROOT / "unet_predictions"

MODEL_PATH = "/home/rag/Code/unet_trained.pth"

device = torch.device("cpu")

# Load trained U-Net
model = UNet(in_channels=1, out_channels=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Collect images
image_paths = (
    list((IMAGE_ROOT / "real").glob("*/*.jpg")) +
    list((IMAGE_ROOT / "fake").glob("*/*.jpg"))
)

print("Total images:", len(image_paths))


class ImageDataset(Dataset):

    def __len__(self):
        return len(image_paths)

    def __getitem__(self, index):
        path = image_paths[index]
        image = transform(Image.open(path).convert("RGB"))
        return image, str(path)


dataset = ImageDataset()
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Generate predictions
with torch.no_grad():

    for images, paths in tqdm(loader, desc="Generating U-Net predictions"):

        images = images.to(device)

        outputs = model(images)
        predictions = torch.sigmoid(outputs)

        for prediction, original_path in zip(predictions, paths):

            original_path = Path(original_path)
            relative_path = original_path.relative_to(IMAGE_ROOT)

            save_path = OUTPUT_ROOT / relative_path
            save_path = save_path.with_suffix(".png")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            edge_image = (
                prediction.squeeze().cpu().numpy() * 255
            ).clip(0, 255).astype("uint8")

            Image.fromarray(edge_image).save(save_path)

print("\nU-Net prediction generation completed.")
print("Saved location:", OUTPUT_ROOT)
print("Total images processed:", len(image_paths))