"""
Motion Detection from Camera
------------------------------
Uses OpenCV to capture live video from a webcam, detect movement
between frames, and print descriptive text to the console based on
what is detected (no motion / motion detected / direction / intensity).

Requirements:
    pip install opencv-python numpy

Usage:
    python motion_detector.py
    Press 'q' to quit the preview window.
"""

import cv2
import numpy as np
import time

# ---------------- CONFIG ----------------
CAMERA_INDEX = 0          # change if you have multiple cameras (0, 1, 2...)
MIN_AREA = 500            # minimum contour area to count as "motion"
BLUR_SIZE = (21, 21)      # gaussian blur kernel for noise reduction
THRESHOLD = 25            # pixel intensity diff threshold
COOLDOWN_SECONDS = 1.0    # minimum time between printed messages
# -----------------------------------------


def describe_motion(total_area, frame_area, cx_diff):
    """Return a text message describing the detected motion."""
    percent = (total_area / frame_area) * 100

    if percent < 0.5:
        intensity = "slight"
    elif percent < 3:
        intensity = "moderate"
    else:
        intensity = "large"

    if abs(cx_diff) < 15:
        direction = "in place"
    elif cx_diff > 0:
        direction = "moving right"
    else:
        direction = "moving left"

    return f"Motion detected: {intensity} movement, {direction} ({percent:.2f}% of frame)"


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open camera. Check CAMERA_INDEX or camera permissions.")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("Error: could not read initial frame from camera.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, BLUR_SIZE, 0)

    prev_cx = None
    last_print_time = 0
    frame_area = prev_frame.shape[0] * prev_frame.shape[1]

    print("Starting motion detection. Press 'q' in the video window to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: lost camera feed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_SIZE, 0)

        # Compute frame difference
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, THRESHOLD, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        total_area = 0
        cx_list = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_AREA:
                continue
            total_area += area
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx_list.append(x + w / 2)

        now = time.time()

        if total_area > 0:
            cx = np.mean(cx_list) if cx_list else 0
            cx_diff = (cx - prev_cx) if prev_cx is not None else 0

            if now - last_print_time > COOLDOWN_SECONDS:
                message = describe_motion(total_area, frame_area, cx_diff)
                print(message)
                last_print_time = now

            prev_cx = cx
            status_text = "Status: Motion Detected"
            color = (0, 0, 255)
        else:
            if now - last_print_time > COOLDOWN_SECONDS * 3:
                print("No motion detected.")
                last_print_time = now
            status_text = "Status: No Motion"
            color = (0, 255, 0)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2)

        cv2.imshow("Motion Detection", frame)
        prev_gray = gray

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
