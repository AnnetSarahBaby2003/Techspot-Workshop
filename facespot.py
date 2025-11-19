import cv2
import numpy as np

# Load Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

# tuning constants (adjust if needed)
VERTICAL_OFFSET_FACTOR = 0.35  # how far above eye-midpoint to place the bindi (fraction of eye-to-eye distance)
RADIUS_FACTOR = 0.18          # bindi radius as fraction of inter-eye distance
MIN_RADIUS = 4

while True:
    ret, img = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        # restrict search for eyes to upper half of face to avoid false detections
        eyey1 = y + int(h * 0.10)
        eyey2 = y + int(h * 0.55)
        ex1 = x
        ex2 = x + w
        face_gray_upper = gray[eyey1:eyey2, ex1:ex2]
        face_color_upper = img[eyey1:eyey2, ex1:ex2]

        eyes = eye_cascade.detectMultiScale(face_gray_upper, scaleFactor=1.1, minNeighbors=6, minSize=(20, 20))

        eye_centers = []
        for (ex, ey, ew, eh) in eyes:
            cx = ex + ew // 2 + ex1
            cy = ey + eh // 2 + eyey1
            eye_centers.append((cx, cy))
            # optional: draw eye centers for debugging
            # cv2.circle(img, (cx, cy), 2, (255,255,0), -1)

        spot_x = None
        spot_y = None
        radius = None

        if len(eye_centers) >= 2:
            # choose the two eyes with largest separation (robust if extra detections)
            eye_centers = sorted(eye_centers, key=lambda p: p[0])  # left-to-right
            left_eye = eye_centers[0]
            right_eye = eye_centers[-1]
            mid_x = (left_eye[0] + right_eye[0]) // 2
            mid_y = (left_eye[1] + right_eye[1]) // 2

            # place bindi slightly above the eye-midpoint
            eye_dist = np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
            offset = int(eye_dist * VERTICAL_OFFSET_FACTOR)
            spot_x = int(mid_x)
            spot_y = int(mid_y - offset)
            radius = max(MIN_RADIUS, int(eye_dist * RADIUS_FACTOR))

        elif len(eye_centers) == 1:
            # estimate the other eye symmetrically using face box
            single = eye_centers[0]
            # estimate inter-eye distance as fraction of face width
            est_eye_dist = int(w * 0.30)
            # determine left/right by comparing to face center
            face_center_x = x + w // 2
            if single[0] < face_center_x:
                left_eye = single
                right_eye = (single[0] + est_eye_dist, single[1])
            else:
                right_eye = single
                left_eye = (single[0] - est_eye_dist, single[1])

            mid_x = (left_eye[0] + right_eye[0]) // 2
            mid_y = (left_eye[1] + right_eye[1]) // 2
            eye_dist = max(1, np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))
            offset = int(eye_dist * VERTICAL_OFFSET_FACTOR)
            spot_x = int(mid_x)
            spot_y = int(mid_y - offset)
            radius = max(MIN_RADIUS, int(eye_dist * RADIUS_FACTOR))

        else:
            # fallback: use simple forehead center (older method)
            spot_x = x + w // 2
            spot_y = y + int(h * 0.20)
            radius = max(MIN_RADIUS, int(w * 0.03))

        # ensure spot is inside image bounds
        H, W = img.shape[:2]
        spot_x = min(max(0, int(spot_x)), W - 1)
        spot_y = min(max(0, int(spot_y)), H - 1)

        # draw bindi (red filled circle)
        cv2.circle(img, (spot_x, spot_y), radius, (0, 0, 255), -1)

        # optional: draw small black center for style
        cv2.circle(img, (spot_x, spot_y), max(1, radius // 4), (0, 0, 0), -1)

    cv2.imshow("Bindi Aligned (press 'q' to quit)", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
