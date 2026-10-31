import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T


# Paths
DATA_ROOT = Path("/home/rag/datasets/FaceForensicsPP")
IMAGE_ROOT = DATA_ROOT / "frames_advo"
MASK_ROOT = DATA_ROOT / "edge_targets"

SPLIT_FILE = IMAGE_ROOT / "splits_advo.json"

IMAGE_SIZE = 224
BATCH_SIZE = 4


class EdgeDataset(Dataset):

    def __init__(self, split):

        with open(SPLIT_FILE, "r") as f:
            split_data = json.load(f)

        self.samples = []

        for item in split_data[split]:

            video = item["video"]
            frames = item["frames"]

            for frame in frames:

                image_path = Path(frame)
                relative_path = image_path.relative_to(IMAGE_ROOT)
                mask_path = MASK_ROOT / relative_path
                mask_path = mask_path.with_suffix(".png")

                if image_path.exists() and mask_path.exists():
                    self.samples.append((image_path, mask_path))

        self.image_transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor()
        ])

        self.mask_transform = T.Compose([
            T.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=T.InterpolationMode.NEAREST
            ),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, mask_path = self.samples[index]

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Convert edge map to binary target
        mask = (mask > 0).float()

        return image, mask


# Create datasets
train_dataset = EdgeDataset("train")
val_dataset = EdgeDataset("validation")
test_dataset = EdgeDataset("test")

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Test
images, masks = next(iter(train_loader))

print("========== U-NET DATALOADER TEST ==========")
print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Testing samples:", len(test_dataset))

print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)

print("Image range:", images.min().item(), images.max().item())
print("Mask range:", masks.min().item(), masks.max().item())

print("U-Net DataLoader test completed.")