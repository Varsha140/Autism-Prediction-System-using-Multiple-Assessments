import pygame
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import gluPerspective
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose and Hands
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands()
# Initialize Pygame and PyOpenGL
pygame.init()
display = (800, 600)
pygame.display.set_mode(display, pygame.OPENGL | pygame.DOUBLEBUF)
gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
glTranslatef(0.0, 0.0, -5)

# Function to draw the stick figure and fingers
def draw_stick_figure(body_joints, finger_joints):
    glBegin(GL_LINES)
    # Draw body joints as lines
    for joint in body_joints:
        glVertex3fv(np.array(joint[0], dtype=np.float32))  # Convert to numpy array of floats
        glVertex3fv(np.array(joint[1], dtype=np.float32))  # Convert to numpy array of floats
    glEnd()

    # Draw fingers as lines
    glBegin(GL_LINES)
    for finger in finger_joints:
        glVertex3fv(np.array(finger[0], dtype=np.float32))  # Convert to numpy array of floats
        glVertex3fv(np.array(finger[1], dtype=np.float32))  # Convert to numpy array of floats
    glEnd()

# Capture pose and hand landmarks using MediaPipe
def capture_pose_and_hands(cap):
    body_joints = []
    finger_joints = []

    ret, frame = cap.read()
    if not ret:
        return body_joints, finger_joints

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process pose and hand landmarks
    pose_results = pose.process(frame_rgb)
    hand_results = hands.process(frame_rgb)

    # Extract body joints from pose
    if pose_results.pose_landmarks:
        for landmark in pose_results.pose_landmarks.landmark:
            x = landmark.x - 0.5  # Center in the frame
            y = landmark.y - 0.5
            z = landmark.z
            body_joints.append([[x, y, z], [x + 0.1, y + 0.1, z]])  # Simple joint connection

    # Extract finger joints from hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for i in range(0, 21, 4):  # Simulate finger joint connection (thumb, index, etc.)
                x = hand_landmarks.landmark[i].x - 0.5
                y = hand_landmarks.landmark[i].y - 0.5
                z = hand_landmarks.landmark[i].z
                finger_joints.append([[x, y, z], [x + 0.05, y + 0.05, z]])

    # Display the frame for debugging
    cv2.imshow('Camera', frame)

    return body_joints, finger_joints

# Main loop
cap = cv2.VideoCapture(0)  # Initialize camera capture
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            cap.release()
            cv2.destroyAllWindows()
            quit()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # Capture pose and hand landmarks
    body_joints, finger_joints = capture_pose_and_hands(cap)

    # Draw the stick figure with body and fingers if detected
    if body_joints or finger_joints:
        draw_stick_figure(body_joints, finger_joints)

    pygame.display.flip()
    pygame.time.wait(10)

    # Check if 'q' is pressed to quit the OpenCV window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()