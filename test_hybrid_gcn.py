import torch
from hybrid_gcn_model import HybridGCN

device = torch.device("cpu")

model = HybridGCN(num_classes=2).to(device)

dummy_input = torch.randn(2, 1, 224, 224)

with torch.no_grad():
    output = model(dummy_input)

print("========== HYBRID GCN TEST ==========")
print("Input shape:", dummy_input.shape)
print("Output shape:", output.shape)
print("Parameters:", sum(p.numel() for p in model.parameters()))
print("Device:", device)

print("Hybrid GCN test completed.")