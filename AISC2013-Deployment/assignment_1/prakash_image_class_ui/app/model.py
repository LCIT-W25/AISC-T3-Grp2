import joblib
import torch
import torch.nn as nn
import numpy as np
from PIL import Image


knn_model = joblib.load("models/knn_model.pkl")
label_encoder = joblib.load("models/label_encoder_dnn.pkl")
label_encoder_knn = joblib.load("models/label_encoder_knn.pkl")


# Define DNN model class
class OptimizedDNN(nn.Module):
    def __init__(self, input_size=4096, hidden_sizes=[2048, 1024, 512, 256, 128], num_classes=len(label_encoder.classes_), dropout=0.6):
        super(OptimizedDNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]), nn.BatchNorm1d(
                hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.BatchNorm1d(
                hidden_sizes[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]), nn.BatchNorm1d(
                hidden_sizes[2]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[2], hidden_sizes[3]), nn.BatchNorm1d(
                hidden_sizes[3]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[3], hidden_sizes[4]), nn.BatchNorm1d(
                hidden_sizes[4]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_sizes[4], num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Load DNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dnn_model = OptimizedDNN(num_classes=len(label_encoder.classes_)).to(device)
dnn_model.load_state_dict(torch.load(
    "models/dnn_model.pth", map_location=device))
dnn_model.eval()


# Preprocess function
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((64, 64))  # Resize to match training dimensions
    image_array = np.array(image).astype(np.float32) / 255.0  # Normalize
    image_array = image_array.flatten().reshape(1, -1)  # Flatten
    return image_array
