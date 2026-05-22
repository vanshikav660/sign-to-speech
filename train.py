import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from model import SignLanguageCNN

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'please',
         'stop', 'water', 'help', 'sorry', 'come']
DATA_DIR = 'data'
FRAMES_PER_CLIP = 20
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.0003

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

    def augment(self, sequence):
        if np.random.rand() > 0.5:
            sequence = [np.fliplr(frame) for frame in sequence]
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.7, 1.3)
            sequence = [np.clip(frame * factor, 0, 1) for frame in sequence]
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, sequence[0].shape)
            sequence = [np.clip(frame + noise, 0, 1) for frame in sequence]
        return sequence

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence = list(self.samples[idx])
        sequence = self.augment(sequence)
        x = torch.tensor(np.array(sequence), dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# Load dataset
print("Loading dataset...")
dataset = GestureDataset()
print(f"Total clips loaded: {len(dataset)}")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = SignLanguageCNN(num_classes=len(SIGNS))

# Class weights
class_weights = torch.FloatTensor([
    1.0,  # hello    ✅ good
    3.0,  # thankyou ❌ weak
    3.0,  # yes      ❌ weak
    2.0,  # no       🟡 average
    1.5,  # please   🟡 average
    2.0,  # stop     🟡 average
    3.0,  # water    ❌ weak
    2.0,  # help     🟡 average
    1.0,  # sorry    ✅ good
    1.0,  # come     ✅ good
])
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # ✅

# Best accuracy tracker
best_accuracy = 0
no_improve = 0

# Training loop
print("Starting training...")
for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

    scheduler.step()  # ✅

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        no_improve = 0
        torch.save(model.state_dict(), 'model.pth')
        print(f"  ✅ Best model saved! Accuracy: {best_accuracy:.2f}%")
    else:
        no_improve += 1
        print(f"  No improvement for {no_improve} epochs")
        if no_improve >= 7:
            print("  Early stopping triggered!")
            break

print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
print("Model saved as model.pth!")
