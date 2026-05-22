import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import os
import numpy as np
from model import SignLanguageCNN

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'please',
         'stop', 'water', 'help', 'sorry', 'come']
DATA_DIR = 'data'
FRAMES_PER_CLIP = 20
BATCH_SIZE = 16

class GestureDataset(Dataset):
    def __init__(self):
        self.samples = []
        self.labels = []

        for label, sign in enumerate(SIGNS):
            sign_dir = os.path.join(DATA_DIR, sign)
            clips = {}

            for filename in os.listdir(sign_dir):
                clip_num = int(filename.split('_')[0].replace('clip', ''))
                frame_num = int(filename.split('_')[1].replace('frame', '').replace('.jpg', ''))
                if clip_num not in clips:
                    clips[clip_num] = {}
                clips[clip_num][frame_num] = filename

            for clip_num, frames in clips.items():
                if len(frames) == FRAMES_PER_CLIP:
                    sequence = []
                    for i in range(FRAMES_PER_CLIP):
                        img_path = os.path.join(sign_dir, frames[i])
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = img.astype(np.float32) / 255.0
                        sequence.append(img)
                    self.samples.append(np.array(sequence))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SignLanguageCNN(num_classes=len(SIGNS))
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# Load dataset with 80/20 split
print("Loading dataset...")
dataset = GestureDataset()
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size],
                               generator=torch.Generator().manual_seed(42))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Total clips loaded: {len(dataset)}")
print(f"Test set size: {len(test_dataset)}")

# Evaluate
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct / total * 100:.2f}%")

# Per-sign breakdown
print("\nPer-sign accuracy:")
sign_correct = {sign: 0 for sign in SIGNS}
sign_total = {sign: 0 for sign in SIGNS}

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        for label, pred in zip(labels, predicted):
            sign = SIGNS[label.item()]
            sign_total[sign] += 1
            if label == pred:
                sign_correct[sign] += 1

for sign in SIGNS:
    acc = sign_correct[sign] / sign_total[sign] * 100 if sign_total[sign] > 0 else 0
    print(f"  {sign:<12} {acc:.2f}%")