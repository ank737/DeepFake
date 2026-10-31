import torch

from unet_model import UNet


# Use CPU
device = torch.device("cpu")

# Create model
model = UNet(
    in_channels=1,
    out_channels=1
).to(device)

# Create a sample batch
dummy_input = torch.randn(
    8, 1, 224, 224
).to(device)

# Forward pass
with torch.no_grad():

    output = model(dummy_input)

# Display results
print("========== U-NET TEST ==========")

print("Input shape:", dummy_input.shape)

print("Output shape:", output.shape)

print(
    "Total parameters:",
    sum(p.numel() for p in model.parameters())
)

print("Device:", device)

print("U-Net architecture test completed.")