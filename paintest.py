import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 600)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

colors = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 0),
    (255, 43, 138),
    (255, 255, 0),
    (0, 255, 255),
    (0, 0, 0),
]
labels = ["Red", "Blue", "Green", "Purple", "Cyan", "Yellow", "Erase"]

color = colors[0]
xp, yp = 0, 0
brush_thickness = 10
erase_thickness = 50
header_height = 100

canvas = None

while True:
    success, frame = cap.read()
    if not success:
        break

    # Flip frame for a mirrored effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Initialize canvas if not already done
    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    # Draw color selection header
    section_width = w // len(colors)
    for i, (label, col) in enumerate(zip(labels, colors)):
        cv2.rectangle(frame, (i * section_width, 0), ((i + 1) * section_width, header_height), col, -1)
        text_color = (255, 255, 255) if col != (255, 255, 255) else (0, 0, 0)
        cv2.putText(frame, label, (i * section_width + 10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    # Process hand landmarks
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

        # Get fingertip coordinates
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[12][1:]

        # Selection mode: Both fingertips are up
        if y1 < header_height and y2 < header_height:
            xp, yp = 0, 0  # Reset previous points
            section_index = min(max(x1 // section_width, 0), len(colors) - 1)
            color = colors[section_index]

        # Drawing mode: Index finger is down
        elif y1 < y2:
            if xp == 0 and yp == 0:  # First point
                xp, yp = x1, y1
            thickness = erase_thickness if color == (0, 0, 0) else brush_thickness
            cv2.line(canvas, (xp, yp), (x1, y1), color, thickness)
            xp, yp = x1, y1

    else:
        xp, yp = 0, 0

    # Create mask for overlaying drawings
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    frame_with_canvas = cv2.bitwise_and(frame, cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR))
    frame_with_canvas = cv2.bitwise_or(frame_with_canvas, canvas)

    # Display frames
    cv2.imshow("Virtual Painter", frame_with_canvas)
    cv2.imshow("Canvas", canvas)

    # Quit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
