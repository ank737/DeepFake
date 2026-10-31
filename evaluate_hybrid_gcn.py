import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from hybrid_gcn_model import HybridGCN
from gcn_dataloader import test_loader

# Load trained model
device = torch.device("cpu")

model = HybridGCN(num_classes=2).to(device)
model.load_state_dict(
    torch.load(
        "/home/rag/Code/hybrid_gcn_trained.pth",
        map_location=device
    )
)

model.eval()

all_labels = []
all_predictions = []

# Evaluate
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.numpy())

# Results
print("\n========== TEST RESULTS ==========")

print("Accuracy:", accuracy_score(all_labels, all_predictions))

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions)) 