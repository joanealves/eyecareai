import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.running = False
        self.ear_threshold = 0.25 
        self.consecutive_frames = 30  
        self.closed_eyes_counter = 0
        self.fatigue_detected = False
        self.start_time = time.time()
        self.fatigue_events = []

    def calculate_ear(self, eye_points, landmarks, frame):
        """Calcula o Eye Aspect Ratio (EAR) para um olho."""
        coords = [np.array([landmarks[p].x * frame.shape[1], landmarks[p].y * frame.shape[0]]) for p in eye_points]
        
        vert1 = np.linalg.norm(coords[1] - coords[5])
        vert2 = np.linalg.norm(coords[2] - coords[4])
        
        hor = np.linalg.norm(coords[0] - coords[3])
        
        ear = (vert1 + vert2) / (2.0 * hor)
        return ear

    def process_frame(self, frame):
        """Processa um frame de vídeo e detecta fadiga."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [33, 160, 158, 133, 153, 144]
                right_eye = [362, 385, 387, 263, 373, 380]
                
                left_ear = self.calculate_ear(left_eye, face_landmarks.landmark, frame)
                right_ear = self.calculate_ear(right_eye, face_landmarks.landmark, frame)
                ear = (left_ear + right_ear) / 2.0
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION)
                
                if ear < self.ear_threshold:
                    self.closed_eyes_counter += 1
                    if self.closed_eyes_counter >= self.consecutive_frames and not self.fatigue_detected:
                        self.fatigue_detected = True
                        self.fatigue_events.append(time.time())
                        self.gui.update_fatigue_status("Fadiga Detectada!")
                        winsound.Beep(1000, 500)
                else:
                    self.closed_eyes_counter = 0
                    self.fatigue_detected = False
                    self.gui.update_fatigue_status("Normal")
                
                self.gui.update_metrics(ear, len(self.fatigue_events))
        
        return frame

    def start(self):
        """Inicia o processamento de vídeo em uma thread separada."""
        self.running = True
        threading.Thread(target=self.video_loop, daemon=True).start()

    def video_loop(self):
        """Loop principal de processamento de vídeo."""
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self.process_frame(frame)
            cv2.imshow('EyeCareAI', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.stop()

    def stop(self):
        """Para o processamento de vídeo e libera recursos."""
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()