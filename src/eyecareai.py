import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye_points, landmarks, frame_shape):
    coords = np.array([(landmarks[i].x * frame_shape[1], landmarks[i].y * frame_shape[0]) for i in eye_points])
    vert_dist1 = distance.euclidean(coords[1], coords[5])
    vert_dist2 = distance.euclidean(coords[2], coords[4])
    hor_dist = distance.euclidean(coords[0], coords[3])
    ear = (vert_dist1 + vert_dist2) / (2.0 * hor_dist)
    return ear

class EyeCareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EyeCareAI")
        self.running = False
        self.cap = None
        self.fatigue_times = []
        self.start_times = []

        # Interface
        self.start_button = tk.Button(root, text="Iniciar Monitoramento", command=self.start_monitoring)
        self.start_button.pack(pady=10)
        self.stop_button = tk.Button(root, text="Parar Monitoramento", command=self.stop_monitoring, state="disabled")
        self.stop_button.pack(pady=10)
        self.status_label = tk.Label(root, text="Status: Parado")
        self.status_label.pack(pady=10)

        # Gráfico
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(pady=10)

    def start_monitoring(self):
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_label.config(text="Status: Monitorando")
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
                self.stop_monitoring()
                return
            threading.Thread(target=self.monitor, daemon=True).start()

    def stop_monitoring(self):
        self.running = False
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_label.config(text="Status: Parado")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(self.start_times, self.fatigue_times, label="Tempo de Olhos Fechados (s)")
        self.ax.set_xlabel("Tempo (s)")
        self.ax.set_ylabel("Olhos Fechados (s)")
        self.ax.legend()
        self.canvas.draw()

    def monitor(self):
        closed_eyes_time = 0
        session_start = time.time()
        EAR_THRESHOLD = 0.2
        FATIGUE_THRESHOLD = 3.0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                    left_ear = calculate_ear(LEFT_EYE, face_landmarks.landmark, frame.shape)
                    right_ear = calculate_ear(RIGHT_EYE, face_landmarks.landmark, frame.shape)
                    avg_ear = (left_ear + right_ear) / 2.0

                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    if avg_ear < EAR_THRESHOLD:
                        if closed_eyes_time == 0:
                            closed_eyes_time = time.time()
                        elif time.time() - closed_eyes_time > FATIGUE_THRESHOLD:
                            cv2.putText(frame, "FADIGA DETECTADA!", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        closed_eyes_time = 0

                    self.start_times.append(time.time() - session_start)
                    self.fatigue_times.append(time.time() - closed_eyes_time if closed_eyes_time else 0)
                    self.update_plot()

            cv2.imshow("EyeCareAI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_monitoring()

if __name__ == "__main__":
    root = tk.Tk()
    app = EyeCareApp(root)
    root.mainloop()