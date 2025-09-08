import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet50
from streamlit_drawable_canvas import st_canvas

# ----------------------------
# Load model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50()
checkpoint = torch.load("resnet50_cifar10_lr01.pth", map_location=device)
model.load_state_dict(checkpoint["net"])
model.to(device)
model.eval()

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
# ----------------------------
# Streamlit UI
# ----------------------------
st.title("CIFAR-10 ResNet50")
st.write("upload an image to see the prediction.")

# ----------------------------
# Image upload
# ----------------------------
uploaded_file = st.file_uploader("Or upload a digit image", type=["png", "jpg", "jpeg"])

# ----------------------------
# Process image
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    tensor = transform(img).unsqueeze(0).to(device) # type: ignore
    with torch.no_grad():
        output = model(tensor)
        pred_idx = int(torch.argmax(output, dim=1).item())
        probs = F.softmax(output, dim=1).cpu().numpy()[0]

    st.success(f"Predicted class: {classes[pred_idx]}")


# ----------------------------
# Prediction and visualization
# ----------------------------
if st.checkbox("Show debug visuals"):
        first_layer = F.relu(model.conv1(tensor)).squeeze(0)  # [64,32,32]
        num_features = min(32, first_layer.shape[0])
        cols = 8
        rows = int(np.ceil(num_features / cols))

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])

        # Image
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(img.resize((32,32)))
        ax1.set_title("Input image (32x32)")
        ax1.axis("off")

        # Probabilities
        ax2 = fig.add_subplot(gs[0,1])
        ax2.bar(range(10), probs)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(classes, rotation=45)
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction probabilities")

        # Feature maps
        ax3 = fig.add_subplot(gs[1, :])
        ax3.set_title("First Conv Layer Feature Maps")
        ax3.axis("off")

        for i in range(num_features):
            row_idx = i // cols
            col_idx = i % cols
            ax_inset = ax3.inset_axes((
                col_idx / cols,
                1 - (row_idx + 1) / rows,
                1 / cols,
                1 / rows
            ))
            ax_inset.imshow(first_layer[i].cpu().detach().numpy(), cmap="gray")
            ax_inset.axis("off")

        plt.tight_layout()
        st.pyplot(fig)