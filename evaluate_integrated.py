import torch
import torch.nn.functional as F
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from unet_model import UNet
from hybrid_gcn_model import HybridGCN
from gcn_dataloader import test_loader

device = torch.device("cpu")

# Load trained U-Net
unet = UNet(in_channels=1, out_channels=1).to(device)
unet.load_state_dict(
    torch.load("unet_trained.pth", map_location=device)
)
unet.eval()

# Load trained Hybrid GCN
gcn = HybridGCN(num_classes=2).to(device)
gcn.load_state_dict(
    torch.load("hybrid_gcn_trained.pth", map_location=device)
)
gcn.eval()

all_labels = []
all_predictions = []

# Evaluate complete test dataset
with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)

        # U-Net edge prediction
        edge_output = unet(images)
        edge_prediction = torch.sigmoid(edge_output)

        # Hybrid GCN classification
        outputs = gcn(edge_prediction)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
cm = confusion_matrix(all_labels, all_predictions)

print("\n========== INTEGRATED PIPELINE RESULTS ==========")
print("Test samples:", len(all_labels))
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    all_labels,
    all_predictions,
    target_names=["Real", "Fake"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(cm)

# Save results
results = pd.DataFrame({
    "Actual": all_labels,
    "Predicted": all_predictions
})

results.to_csv("integrated_results.csv", index=False)

print("\nResults saved to integrated_results.csv")