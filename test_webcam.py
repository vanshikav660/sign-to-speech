import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pyttsx3
import threading
from model import SignLanguageCNN

SIGNS = ['hello', 'thankyou', 'yes', 'no', 'please',
         'stop', 'water', 'help', 'sorry', 'come']
FRAMES_PER_CLIP = 20
IMG_SIZE = 64

engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    def run():
        print(f"Speaking: {text}")  
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


model = SignLanguageCNN(num_classes=len(SIGNS))
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

cap = cv2.VideoCapture(0)
frames = []
prediction_buffer = []

last_prediction = None
cooldown_counter = 0
COOLDOWN_FRAMES = 20
CONFIDENCE_THRESHOLD = 50  
BUFFER_SIZE = 12
REQUIRED_AGREEMENT = 5     

kernel = np.ones((3, 3), np.uint8)

def extract_hand(roi):
   
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

x1, y1, x2, y2 = 150, 100, 570, 520

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame[y1:y2, x1:x2]

    mask = extract_hand(roi)
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

    mask_preview = cv2.resize(mask, (150, 150))
    mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
    frame[0:150, 0:150] = mask_preview_bgr
    cv2.putText(frame, "Model view", (0, 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    gray = mask_resized.astype(np.float32) / 255.0
    frames.append(gray)

    if len(frames) > FRAMES_PER_CLIP:
        frames.pop(0)

   
    if cooldown_counter > 0:
        cooldown_counter -= 1

    
    if len(frames) >= FRAMES_PER_CLIP:
        white_pixels = np.sum(gray > 0.5)
        if white_pixels < 50:  
            cv2.putText(frame, 'No hand detected', (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        else:
           
            clip = np.array(frames[-FRAMES_PER_CLIP:]).astype(np.float32)  
            clip = torch.tensor(clip).unsqueeze(0)

            with torch.no_grad():
                output = model(clip)
                probs = F.softmax(output, dim=1)
                confidence, predicted = torch.max(probs, 1)
                conf_pct = confidence.item() * 100

            
            print(f"Conf: {conf_pct:.1f}%, Pred: {predicted.item()}, Buf len: {len(prediction_buffer)}")

            prediction_buffer.append(predicted.item())
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

            if prediction_buffer: 
                final_pred = max(set(prediction_buffer), key=prediction_buffer.count)
                agreement = prediction_buffer.count(final_pred)
                label = SIGNS[final_pred]

                
                if (conf_pct > CONFIDENCE_THRESHOLD and
                    agreement >= REQUIRED_AGREEMENT and
                    cooldown_counter == 0 and
                    final_pred != last_prediction):

                    last_prediction = final_pred
                    cooldown_counter = COOLDOWN_FRAMES
                    
                    speak(label)
                    cv2.putText(frame, f'Sign: {label}', (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, f'Conf: {conf_pct:.1f}% Agr: {agreement}/{BUFFER_SIZE}', (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                elif cooldown_counter > 0:
                    cv2.putText(frame, f'Sign: {SIGNS[last_prediction]}', (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, f'Cooldown: {cooldown_counter}', (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                else:
                    cv2.putText(frame, f'Analyzing... {label} ({agreement}/{BUFFER_SIZE})', (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
                    cv2.putText(frame, f'Conf: {conf_pct:.1f}%', (10, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            else:
                cv2.putText(frame, 'Warming up buffer...', (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)
    else:
        cv2.putText(frame, f'Warming up... {len(frames)}/20', (10, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 2)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
