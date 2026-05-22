# Real-Time Sign Language to Speech Translation

A deep learning system that recognizes dynamic sign language gestures from a live webcam feed and converts them into spoken audio in real time.

## Project Overview

This project implements a real-time sign language recognition system using a custom CNN+LSTM architecture trained entirely from scratch. The system captures live video, isolates hand gestures using skin detection, recognizes the sign, and speaks it out loud.

## Supported Signs

The system recognizes 10 dynamic signs:
- Hello
- Thank You
- Yes
- No
- Please
- Stop
- Water
- Help
- Sorry
- Come

## System Architecture

- **Data Collection** — Live webcam capture with skin-based hand isolation
- **Preprocessing** — Skin color detection to extract hand region from background
- **CNN** — 3 convolutional layers for spatial feature extraction
- **LSTM** — Temporal modeling of gesture motion across 20 frames
- **Classification** — Softmax output with confidence thresholding
- **Speech Output** — pyttsx3 text-to-speech conversion

## Tech Stack

- Python
- PyTorch
- OpenCV
- NumPy
- pyttsx3

## Project Structure
```
sign_language_macs/
│
├── model.py          # CNN+LSTM architecture
├── train.py          # Training pipeline with augmentation
├── evaluate.py       # Model evaluation with per-sign accuracy
├── collect_Data.py   # Data collection with skin detection
├── test_webcam.py    # Real-time inference with speech output
```

## Model Performance

- **Training Accuracy:** 99.75%
- **Test Accuracy:** 92%
- 9 out of 10 signs achieve 100% accuracy

## How to Run

### Install dependencies
```bash
pip install torch torchvision opencv-python numpy pyttsx3
```

### Collect data
```bash
python collect_Data.py
```

### Train model
```bash
python train.py
```

### Evaluate model
```bash
python evaluate.py
```

### Run real-time detection
```bash
python test_webcam.py
```

## Key Features

- CNN trained entirely from scratch — no pretrained models used
- Skin color detection for background-independent hand isolation
- LSTM for explicit temporal gesture modeling
- Data augmentation — flipping, brightness, noise
- Early stopping to prevent overfitting
- Real-time speech output using pyttsx3