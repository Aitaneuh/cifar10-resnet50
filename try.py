import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from model import ResNet50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50()
checkpoint = torch.load("resnet50_cifar10_lr01.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["net"])
model.to(device)
model.eval()

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def predict_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    img = Image.open(path).convert("RGB")
    
    img_disp = img.resize((200,200))
    tk_img = ImageTk.PhotoImage(img_disp)
    label_img.config(image=tk_img)
    label_img.image = tk_img # type: ignore

    tensor = transform(img).unsqueeze(0).to(device) # type: ignore

    with torch.no_grad():
        output = model(tensor)
        pred = int(torch.argmax(output, dim=1).item())
    
    messagebox.showinfo("Prediction", f"Predicted class: {classes[pred]}")

root = tk.Tk()
root.title("CIFAR-10 Predictor")

btn = tk.Button(root, text="Select Image & Predict", command=predict_image)
btn.pack(padx=20, pady=10)

label_img = tk.Label(root)
label_img.pack(padx=20, pady=10)

root.mainloop()
