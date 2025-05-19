import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
from PIL import Image
import time
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

def main():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

        cv2.imshow("EyeCareAI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()