import torch
import torch.nn as nn
import torch.optim as optim

from hybrid_gcn_model import HybridGCN
from unet_gcn_dataloader import train_loader, val_loader

EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")

model = HybridGCN(num_classes=2).to(device)

# Weighted loss to address class imbalance
class_weights = torch.tensor(
    [0.70, 1.79], dtype=torch.float32
).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)

for epoch in range(EPOCHS):

    model.train()
    train_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)
    accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} "
        f"Validation Loss: {val_loss:.4f} "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

torch.save(
    model.state_dict(),
    "/home/rag/Code/hybrid_gcn_unet_trained.pth"
)

print("\nTraining completed.")
print("Model saved: hybrid_gcn_unet_trained.pth")