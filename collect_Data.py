import cv2
import os
import numpy as np

SIGNS = [ 'please',
         'stop', 'water', 'help', 'sorry', 'come']

CLIPS_PER_SIGN = 40
FRAMES_PER_CLIP = 20
DATA_DIR = 'data'

INSTRUCTIONS = {
    'hello':    "Big horizontal wave left to right with open palm",
    'thankyou': "ONE hand flat at chin move STRAIGHT FORWARD slowly",
    'yes':      "Whole arm move UP then DOWN slowly with open hand",
    'no':       "Index finger only wag side to side fast",
    'please':   "Big fast circle on chest with open palm",
    'stop':     "One hand flat horizontal other hand CHOPS DOWN on it",
    'water':    "THREE fingers W shape TAP chin 3 times",
    'help':     "Thumbs up make big circle in air",
    'sorry':    "Both hands clasped praying shake UP and DOWN",
    'come':     "Whole arm sweep toward yourself big motion",
}

def extract_hand(roi):
    # Convert to HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Skin color range
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # Create mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    # Clean up
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    return mask

for sign in SIGNS:
    os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

print("Folders created successfully!")
print("Ready to collect data for", SIGNS)

cap = cv2.VideoCapture(0)
x1, y1, x2, y2 = 150, 100, 570, 520
print("\n=== Data collection started ===")
print("Controls: SPACE=start recording | Q=quit\n")

for sign in SIGNS:
    print(f"\n{'='*40}")
    print(f"NEXT SIGN: {sign.upper()}")
    print(f"HOW TO DO IT: {INSTRUCTIONS[sign]}")
    print(f"{'='*40}")
    print("Press SPACE when ready....")

    while True:
        ret, frame = cap.read()
        roi = frame[y1:y2, x1:x2]

        # Show skin mask preview
        mask = extract_hand(roi)
        mask_preview = cv2.resize(mask, (150, 150))
        mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
        frame[0:150, 0:150] = mask_preview_bgr

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Sign: {sign.upper()}", (10, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, INSTRUCTIONS[sign][:40], (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
        cv2.putText(frame, "Check top-left: hand should be WHITE", (10, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Press SPACE when ready", (10, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break

    for clip_num in range(CLIPS_PER_SIGN):

        # 1 second pause
        for _ in range(20):
            ret, frame = cap.read()
            roi = frame[y1:y2, x1:x2]
            mask = extract_hand(roi)
            mask_preview = cv2.resize(mask, (150, 150))
            mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            frame[0:150, 0:150] = mask_preview_bgr

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"GET READY... Clip {clip_num+1}/{CLIPS_PER_SIGN}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, INSTRUCTIONS[sign][:40], (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.imshow('Data Collection', frame)
            cv2.waitKey(50)

        # Record clip
        for frame_num in range(FRAMES_PER_CLIP):
            ret, frame = cap.read()
            roi = frame[y1:y2, x1:x2]

            # Extract hand using skin detection
            mask = extract_hand(roi)
            mask_resized = cv2.resize(mask, (64, 64))

            # Show preview
            mask_preview = cv2.resize(mask, (150, 150))
            mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            frame[0:150, 0:150] = mask_preview_bgr

            clip_id = clip_num + 100
            filename = f"clip{clip_id:03d}_frame{frame_num:02d}.jpg"
            path = os.path.join(DATA_DIR, sign, filename)
            cv2.imwrite(path, mask_resized)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"RECORDING: {sign.upper()}", (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Clip {clip_num+1}/{CLIPS_PER_SIGN}", (10, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, INSTRUCTIONS[sign][:40], (10, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
            cv2.imshow('Data Collection', frame)

            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                exit()

        print(f"  Clip {clip_num+1}/{CLIPS_PER_SIGN} done for '{sign}'")

    print(f"Finished collecting '{sign}'!\n")

cap.release()
cv2.destroyAllWindows()
print("=== ALL DONE! Check your data folder ===")