import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_advo_cnn import SimpleCNN
from advo_dataloader import test_loader

device = torch.device("cpu")

model = SimpleCNN().to(device)
model.load_state_dict(
    torch.load("advo_cnn_trained.pth", map_location=device)
)
model.eval()

actual = []
predicted = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        actual.extend(labels.numpy())
        predicted.extend(predictions.numpy())

print("\n========== ADVO CNN TEST RESULTS ==========")
print("Accuracy:", accuracy_score(actual, predicted))

print("\nClassification Report:")
print(classification_report(
    actual, predicted,
    target_names=["Real", "Fake"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))