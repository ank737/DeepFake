import torch
import torch.nn as nn
import torch.optim as optim

from unet_model import UNet
from unet_dataloader import train_loader, val_loader


# Configuration
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")

# Model
model = UNet(in_channels=1, out_channels=1).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training
for epoch in range(EPOCHS):

    model.train()
    train_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)


    # Validation
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f}"
    )


# Save model
torch.save(
    model.state_dict(),
    "/home/rag/Code/unet_trained.pth"
)

print("\nTraining completed.")
print("Model saved: ~/Code/unet_trained.pth")