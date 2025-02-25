import cv2 as cv
import mediapipe as mp 
import time
import numpy as np
import math

# Variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CONSECUTIVE_CLOSED_EYES = 0 
MAX_CONSECUTIVE_CLOSED_EYES = 10  # Adjust this value to fit your needs for detecting sleep
STOP_BLINK_COUNTING = False  # Flag to stop counting blinks

# Constants
CLOSED_EYES_FRAME = 1  # Number of frames with closed eyes to count as a blink
FONTS = cv.FONT_HERSHEY_COMPLEX

# Face boundary indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

map_face_mesh = mp.solutions.face_mesh

# Camera object
camera = cv.VideoCapture(0)

# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)*2 + (y1 - y)*2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    start_time = time.time()

    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        eyes_detected = False

        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)

            if ratio > 5:
                CEF_COUNTER += 1
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    if not STOP_BLINK_COUNTING:
                        TOTAL_BLINKS += 1
                    CONSECUTIVE_CLOSED_EYES += 1
                    if CONSECUTIVE_CLOSED_EYES > MAX_CONSECUTIVE_CLOSED_EYES:
                        cv.putText(frame, 'Person is Sleeping', (200, 100), FONTS, 2.5, (0, 0, 255), 2)
                        STOP_BLINK_COUNTING = True  # Stop counting blinks
                    CEF_COUNTER = 0
                cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, (255, 0, 0), 2)
            else:
                CEF_COUNTER = 0
                CONSECUTIVE_CLOSED_EYES = 0  # Reset the counter if eyes are open
                STOP_BLINK_COUNTING = False  # Resume counting blinks if eyes are open

            cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 150), FONTS, 2, (0, 255, 0), 2)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
            cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, (0, 255, 0), 1, cv.LINE_AA)
            eyes_detected = True

        if eyes_detected:
            focus_status = 'Focused'
            focus_color = (0, 255, 0)  # Green color for Focused
        else:
            focus_status = 'Not Focused'
            focus_color = (0, 0, 255)  # Red color for Not Focused

        # Display the focus status with colors
        cv.putText(frame, f'Focus: {focus_status}', (30, 100), FONTS, 1.0, focus_color, 2)

        # Calculate FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time
        cv.putText(frame, f'FPS: {round(fps, 1)}', (30, 50), FONTS, 1.0, (255, 255, 255), 2)

        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            break

    cv.destroyAllWindows()
    camera.release()