import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from model import ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50()
checkpoint = torch.load("resnet50_cifar10_lr01.pth", map_location=device)
model.load_state_dict(checkpoint["net"])
model.to(device)
model.eval()


classes = ['plane','car','bird','cat','deer',
           'dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

class App:
    def __init__(self, master):
        self.master = master
        self.master.title("CIFAR-10 ResNet50 Predictor")
        
        self.button_predict = tk.Button(master, text="Select Image & Predict", command=self.predict)
        self.button_predict.pack(side='left', padx=5, pady=5)
        
        self.button_debug = tk.Button(master, text="Debug Visuals", command=self.debug_visuals)
        self.button_debug.pack(side='left', padx=5, pady=5)
        
        self.label_img = tk.Label(master)
        self.label_img.pack(padx=20, pady=10)
        
        self.img = None
        self.tensor = None

    def predict(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        
        img = Image.open(path).convert("RGB")
        self.img = img
        img_disp = img.resize((200,200))
        tk_img = ImageTk.PhotoImage(img_disp)
        self.label_img.config(image=tk_img)
        self.label_img.image = tk_img # type: ignore
        
        self.tensor = transform(img).unsqueeze(0).to(device) # type: ignore

        with torch.no_grad():
            output = model(self.tensor)
            pred = int(torch.argmax(output, dim=1).item())
        
        messagebox.showinfo("Prediction", f"Predicted class: {classes[pred]}")

    def debug_visuals(self):
        if self.img is None or self.tensor is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        
        with torch.no_grad():
            output = model(self.tensor)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]

            first_layer = model.conv1(self.tensor)  # ResNet50 -> conv1
            x = F.relu(first_layer).squeeze(0)      # [64, 32, 32]

        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 2])
        
        ax1 = fig.add_subplot(gs[0,0])
        ax1.imshow(self.img.resize((32,32)))
        ax1.set_title("Input image (32x32)")
        ax1.axis("off")
        
        ax2 = fig.add_subplot(gs[0,1])
        ax2.bar(range(10), probs)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(classes, rotation=45)
        ax2.set_ylabel("Probability")
        ax2.set_title("Prediction probabilities")
        
        num_features = min(32, x.shape[0])
        cols = 8
        rows = int(np.ceil(num_features / cols))

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
            ax_inset.imshow(x[i].cpu(), cmap="gray")
            ax_inset.axis("off")
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
