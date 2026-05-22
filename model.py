import torch
import torch.nn as nn

class SignLanguageCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SignLanguageCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # ✅ Added dropout layers
        self.dropout_cnn = nn.Dropout(p=0.3)
        self.dropout_lstm = nn.Dropout(p=0.5)

        self.lstm = nn.LSTM(128 * 8 * 8, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        batch_size, seq_len, h, w = x.size()
        x = x.view(batch_size * seq_len, 1, h, w)

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)  # ✅ dropout after conv1

        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)  # ✅ dropout after conv2

        x = self.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(x)
        x = self.dropout_lstm(lstm_out[:, -1, :])  # ✅ dropout before final layer
        x = self.fc(x)
        return x
       