import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128*8*8, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


model = CNN().to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

st.title("Cat vs Dog Classifier")
st.write("Upload an image and see whether the model predicts Cat or Dog.")

# Update with your latest training accuracies
train_accuracy_list=[58.15, 66.025, 67.95, 69.7, 70.0, 71.3, 72.8, 73.175, 73.175, 74.15, 74.625, 74.675, 76.05, 76.1, 77.1, 77.75, 77.425, 78.4, 78.6, 79.95]

val_accuracy_list=[52.3, 63.5, 64.9, 66.2, 66.5, 66.7, 66.8, 67.2, 68.9, 69.1, 69.2, 69.4, 70.2, 69.6, 70.6, 70.8, 70.7, 72.2, 71.4, 72.9]

st.write(f"Final Train Accuracy: {train_accuracy_list[-1]:.2f}%")

st.subheader("Model Training Progress")
fig, ax = plt.subplots()
ax.plot(range(1, len(train_accuracy_list)+1), train_accuracy_list, label="Train Accuracy")
ax.plot(range(1, len(val_accuracy_list)+1), val_accuracy_list, label="Validation Accuracy")
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
st.pyplot(fig)

st.subheader("Upload an Image for Prediction")
uploaded_file = st.file_uploader("Choose a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            raw_output = output.item()
            pred = (output > 0.5).item()
            label = "Dog" if pred else "Cat"
            confidence = raw_output if pred else (1 - raw_output)
            
        st.write(f"### Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2%}**")
        
        if st.checkbox("Show raw model output"):
            st.write(f"Raw model output: {raw_output:.4f}")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")