import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import os
import json
from datetime import datetime, timedelta
import platform

class VideoProcessor:
    def __init__(self, gui, config_path="config.json"):
        self.gui = gui
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        
        self.config_path = config_path
        self.load_config()
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.cap = cv2.VideoCapture(0)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = False
        self.paused = False
        
        self.closed_eyes_counter = 0
        self.fatigue_detected = False
        self.last_blink_time = time.time()
        self.blink_count = 0
        self.blink_rate_history = []
        
        self.fatigue_events = []
        self.fatigue_start_time = None
        
        self.screen_time_start = time.time()
        self.last_break_reminder = time.time()
        self.break_taken = False
        
        self.face_distance = 0
        self.distance_history = []
        self.distance_too_close_counter = 0
        
        self.posture_data = []
        self.bad_posture_counter = 0
        self.posture_warning_active = False
        
        self.start_time = time.time()
        self.session_stats = {
            "total_duration": 0,
            "fatigue_events": 0,
            "blink_rate": 0,
            "posture_warnings": 0,
            "distance_warnings": 0,
            "break_reminders": 0
        }
        
        self.load_historical_data()

        self.left_eye = [33, 160, 158, 133, 153, 144] 
        self.right_eye = [362, 385, 387, 263, 373, 380]  
        
        self.frame_times = []
        self.max_frame_times = 30  
        self.last_frame_time = time.time()
        
        self.translations = {
            "fatigue_detected": "Fadiga Detectada!",
            "normal_status": "Normal",
            "paused_status": "Pausado",
            "monitoring_status": "Monitorando",
            "distance_warning": "AVISO: Muito perto da tela!",
            "posture_warning": "AVISO: Corrija sua postura!",
            "break_reminder": "HORA DA PAUSA! Olhe para longe por 20s",
            "calibration_start": "Iniciando calibração automática...",
            "calibration_failed": "Falha na calibração - nenhum rosto detectado.",
            "calibration_complete": "Calibração concluída. Novo threshold EAR: {:.3f}"
        }
        
    def load_config(self):
        """Carrega as configurações do usuário ou cria configurações padrão."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                self.ear_threshold = config.get("ear_threshold", 0.25)
                self.consecutive_frames = config.get("consecutive_frames", 20)
                self.use_audio_alerts = config.get("use_audio_alerts", True)
                
                self.break_interval = config.get("break_interval", 20 * 60)  
                self.break_duration = config.get("break_duration", 20) 
                self.use_break_reminders = config.get("use_break_reminders", True)
                
                self.min_face_distance = config.get("min_face_distance", 0.5) 
                self.use_distance_alerts = config.get("use_distance_alerts", True)
                
                self.max_head_tilt = config.get("max_head_tilt", 15)  
                self.use_posture_alerts = config.get("use_posture_alerts", True)
                
                self.dark_mode = config.get("dark_mode", False)
                self.language = config.get("language", "pt-BR")
                
            except Exception as e:
                print(f"Erro ao carregar configurações: {e}")
                self.set_default_config()
        else:
            self.set_default_config()
    
    def set_default_config(self):
        """Define as configurações padrão."""
        self.ear_threshold = 0.25
        self.consecutive_frames = 20
        self.use_audio_alerts = True
        
        self.break_interval = 20 * 60 
        self.break_duration = 20  
        self.use_break_reminders = True
        
        self.min_face_distance = 0.5 
        self.use_distance_alerts = True
        
        self.max_head_tilt = 15 
        self.use_posture_alerts = True
        
        self.dark_mode = False
        self.language = "pt-BR"
        
        self.save_config()
    
    def save_config(self):
        """Salva as configurações do usuário."""
        config = {
            "ear_threshold": self.ear_threshold,
            "consecutive_frames": self.consecutive_frames,
            "use_audio_alerts": self.use_audio_alerts,
            "break_interval": self.break_interval,
            "break_duration": self.break_duration,
            "use_break_reminders": self.use_break_reminders,
            "min_face_distance": self.min_face_distance,
            "use_distance_alerts": self.use_distance_alerts,
            "max_head_tilt": self.max_head_tilt,
            "use_posture_alerts": self.use_posture_alerts,
            "dark_mode": self.dark_mode,
            "language": self.language
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
    
    def load_historical_data(self):
        """Carrega dados históricos do uso."""
        data_path = os.path.join("data", "user_history.json")
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    self.historical_data = json.load(f)
            except Exception as e:
                print(f"Erro ao carregar dados históricos: {e}")
                self.historical_data = {"days": {}, "weekly_summary": {}, "monthly_summary": {}}
        else:
            self.historical_data = {"days": {}, "weekly_summary": {}, "monthly_summary": {}}
    
    def save_historical_data(self):
        """Salva os dados históricos do uso."""
        data_path = os.path.join("data", "user_history.json")
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            if today not in self.historical_data["days"]:
                self.historical_data["days"][today] = {
                    "total_session_time": 0,
                    "fatigue_events": [],
                    "blink_rates": [],
                    "posture_warnings": 0,
                    "distance_warnings": 0,
                    "break_reminders": 0
                }

            session_duration = time.time() - self.start_time
            self.historical_data["days"][today]["total_session_time"] += session_duration
            self.historical_data["days"][today]["fatigue_events"].extend(self.fatigue_events)
            self.historical_data["days"][today]["blink_rates"].extend(self.blink_rate_history)

            with open(data_path, 'w') as f:
                json.dump(self.historical_data, f, indent=4)
        except Exception as e:
            print(f"[ERRO AO SALVAR DADOS HISTÓRICOS] {e}")
    
    def calculate_ear(self, eye_points, landmarks, frame):
        """Calcula o Eye Aspect Ratio (EAR) para um olho."""
        coords = [np.array([landmarks[p].x * frame.shape[1], landmarks[p].y * frame.shape[0]]) for p in eye_points]
        
        vert1 = np.linalg.norm(coords[1] - coords[5])
        vert2 = np.linalg.norm(coords[2] - coords[4])
        
        hor = np.linalg.norm(coords[0] - coords[3])
        
        ear = (vert1 + vert2) / (2.0 * hor) if hor > 0 else 0
        return ear, coords
    
    def estimate_distance(self, face_landmarks, frame):
        """Estima a distância aproximada entre o usuário e a câmera."""
        if not face_landmarks:
            return None
        
        left_eye_center = np.mean([np.array([face_landmarks.landmark[p].x * frame.shape[1], 
                                             face_landmarks.landmark[p].y * frame.shape[0]]) 
                                    for p in self.left_eye], axis=0)
        
        right_eye_center = np.mean([np.array([face_landmarks.landmark[p].x * frame.shape[1], 
                                              face_landmarks.landmark[p].y * frame.shape[0]]) 
                                     for p in self.right_eye], axis=0)
        
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        estimated_distance = 100 / eye_distance if eye_distance > 0 else 0
        
        return estimated_distance
    
    def check_posture(self, pose_landmarks):
        """Verifica se a postura do usuário está adequada."""
        if not pose_landmarks:
            return True, 0
        
        try:
            nose = np.array([
                pose_landmarks.landmark[0].x,
                pose_landmarks.landmark[0].y
            ])
            
            left_shoulder = np.array([
                pose_landmarks.landmark[11].x,
                pose_landmarks.landmark[11].y
            ])
            
            right_shoulder = np.array([
                pose_landmarks.landmark[12].x,
                pose_landmarks.landmark[12].y
            ])
            
            mid_shoulder = (left_shoulder + right_shoulder) / 2
            
            shoulder_to_nose = nose - mid_shoulder
            
            vertical = np.array([0, -1])  
            
            if np.linalg.norm(shoulder_to_nose) > 0:
                shoulder_to_nose = shoulder_to_nose / np.linalg.norm(shoulder_to_nose)
            
            dot_product = np.dot(vertical, shoulder_to_nose)
            angle = np.arccos(np.clip(dot_product, -1.0, 1.0)) * 180 / np.pi
            
            return angle <= self.max_head_tilt, angle
            
        except Exception as e:
            print(f"Erro ao verificar postura: {e}")
            return True, 0
    
    def play_alert(self, frequency=1000, duration=500):
        """Reproduz um alerta sonoro."""
        if not self.use_audio_alerts:
            return
            
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(frequency, duration)
            else:
                pass
        except Exception as e:
            print(f"Erro ao reproduzir alerta sonoro: {e}")
    
    def check_break_time(self):
        """Verifica se é hora de fazer uma pausa seguindo a regra 20-20-20."""
        if not self.use_break_reminders:
            return False
            
        current_time = time.time()
        time_since_last_break = current_time - self.last_break_reminder
        
        if time_since_last_break >= self.break_interval and not self.break_taken:
            self.last_break_reminder = current_time
            self.session_stats["break_reminders"] += 1
            return True
        
        if self.break_taken and (current_time - self.last_break_reminder >= self.break_duration):
            self.break_taken = False
            
        return False
    
    def process_frame(self, frame):
        """Processa um frame de vídeo e detecta fadiga, postura e distância."""
        if self.paused:
            return frame
        
        current_time = time.time()
        
        self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
        if len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = self.face_mesh.process(frame_rgb)
        
        pose_results = self.pose.process(frame_rgb)
        
        ear = 0.0
        distance_warning = False
        posture_warning = False
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                left_ear, left_coords = self.calculate_ear(self.left_eye, face_landmarks.landmark, frame)
                right_ear, right_coords = self.calculate_ear(self.right_eye, face_landmarks.landmark, frame)
                ear = (left_ear + right_ear) / 2.0
                
                for coord in left_coords:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 2, (0, 255, 0), -1)
                for coord in right_coords:
                    cv2.circle(frame, (int(coord[0]), int(coord[1])), 2, (0, 255, 0), -1)
                
                distance = self.estimate_distance(face_landmarks, frame)
                if distance:
                    self.face_distance = distance
                    self.distance_history.append(distance)
                    if len(self.distance_history) > 60: 
                        self.distance_history.pop(0)
                    
                    avg_distance = sum(self.distance_history) / len(self.distance_history)
                    if avg_distance < self.min_face_distance:
                        self.distance_too_close_counter += 1
                        if self.distance_too_close_counter > 30 and self.use_distance_alerts:  
                            distance_warning = True
                            if self.distance_too_close_counter == 31:  
                                self.session_stats["distance_warnings"] += 1
                                if self.use_audio_alerts:
                                    self.play_alert(800, 300)
                    else:
                        self.distance_too_close_counter = 0
                
                if ear < self.ear_threshold:
                    self.closed_eyes_counter += 1
                    if self.closed_eyes_counter >= self.consecutive_frames and not self.fatigue_detected:
                        self.fatigue_detected = True
                        self.fatigue_events.append(current_time)
                        self.session_stats["fatigue_events"] += 1
                        self.gui.update_fatigue_status(self.translations["fatigue_detected"])
                        if self.use_audio_alerts:
                            self.play_alert(1000, 500)
                else:
                    if 3 <= self.closed_eyes_counter < self.consecutive_frames:
                        self.blink_count += 1
                        self.last_blink_time = current_time
                    
                    self.closed_eyes_counter = 0
                    self.fatigue_detected = False
                    self.gui.update_fatigue_status(self.translations["normal_status"])
                
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1))
        
        if pose_results.pose_landmarks:
            good_posture, angle = self.check_posture(pose_results.pose_landmarks)
            
            if not good_posture:
                self.bad_posture_counter += 1
                if self.bad_posture_counter > 30 and self.use_posture_alerts: 
                    posture_warning = True
                    if not self.posture_warning_active:
                        self.posture_warning_active = True
                        self.session_stats["posture_warnings"] += 1
                        if self.use_audio_alerts:
                            self.play_alert(600, 300)
            else:
                self.bad_posture_counter = 0
                self.posture_warning_active = False
            
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2))
        
        elapsed_time = current_time - self.start_time
        if elapsed_time > 0:
            blink_rate = (self.blink_count / elapsed_time) * 60  
            self.session_stats["blink_rate"] = blink_rate
        
        break_reminder = self.check_break_time()
        
        self.session_stats["total_duration"] = elapsed_time
        
        self.gui.update_all_metrics(
            ear=ear,
            event_count=len(self.fatigue_events),
            blink_rate=self.session_stats["blink_rate"],
            face_distance=self.face_distance,
            posture_warning=posture_warning,
            distance_warning=distance_warning,
            break_reminder=break_reminder,
            fps=fps
        )
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60), font, 0.7, (0, 255, 0), 2)
        
        if distance_warning:
            cv2.putText(frame, self.translations["distance_warning"], (10, 90), font, 0.7, (0, 0, 255), 2)
        
        if posture_warning:
            cv2.putText(frame, self.translations["posture_warning"], (10, 120), font, 0.7, (0, 0, 255), 2)
        
        if break_reminder:
            cv2.putText(frame, self.translations["break_reminder"], (10, 150), font, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def auto_calibrate(self, frames=100):
        """Calibra automaticamente o threshold EAR baseado nos olhos do usuário."""
        if not self.cap.isOpened():
            return False
            
        print(self.translations["calibration_start"])
        ear_values = []
        
        for _ in range(frames):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_ear, _ = self.calculate_ear(self.left_eye, face_landmarks.landmark, frame)
                    right_ear, _ = self.calculate_ear(self.right_eye, face_landmarks.landmark, frame)
                    ear = (left_ear + right_ear) / 2.0
                    ear_values.append(ear)
            
            cv2.imshow('Calibrando EyeCareAI', frame)
            cv2.waitKey(1)
        
        cv2.destroyWindow('Calibrando EyeCareAI')
        
        if not ear_values:
            print(self.translations["calibration_failed"])
            return False
        
        avg_ear = sum(ear_values) / len(ear_values)
        self.ear_threshold = avg_ear * 0.7
        print(self.translations["calibration_complete"].format(self.ear_threshold))
        
        self.save_config()
        return True
    
    def start(self):
        """Inicia o processamento de vídeo em uma thread separada."""
        self.running = True
        threading.Thread(target=self.video_loop, daemon=True).start()
    
    def video_loop(self):
        """Loop principal de processamento de vídeo."""
        os.makedirs("data", exist_ok=True)
        
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = self.process_frame(frame)
            
            cv2.imshow('EyeCareAI', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 'q' para sair
            if key == ord('q'):
                break
            # 'p' para pausar/continuar
            elif key == ord('p'):
                self.toggle_pause()
            # 'c' para calibrar
            elif key == ord('c'):
                self.auto_calibrate()
            # 'b' para indicar que está fazendo uma pausa
            elif key == ord('b'):
                self.break_taken = True
                self.last_break_reminder = time.time()
        
        self.stop()
    
    def stop(self):
        """Para o processamento de vídeo e libera recursos."""
        if self.running:
            self.save_historical_data()
            
        self.running = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
    
    def toggle_pause(self):
        """Pausa ou continua o processamento."""
        self.paused = not self.paused
        status_text = self.translations["paused_status"] if self.paused else self.translations["monitoring_status"]
        self.gui.update_fatigue_status(status_text)
    
    def update_settings(self, settings):
        """Atualiza as configurações do sistema."""
        if "ear_threshold" in settings:
            self.ear_threshold = float(settings["ear_threshold"])
        if "consecutive_frames" in settings:
            self.consecutive_frames = int(settings["consecutive_frames"])
        if "use_audio_alerts" in settings:
            self.use_audio_alerts = bool(settings["use_audio_alerts"])
        if "break_interval" in settings:
            self.break_interval = float(settings["break_interval"]) * 60 
        if "break_duration" in settings:
            self.break_duration = float(settings["break_duration"])
        if "use_break_reminders" in settings:
            self.use_break_reminders = bool(settings["use_break_reminders"])
        if "min_face_distance" in settings:
            self.min_face_distance = float(settings["min_face_distance"])
        if "use_distance_alerts" in settings:
            self.use_distance_alerts = bool(settings["use_distance_alerts"])
        if "max_head_tilt" in settings:
            self.max_head_tilt = float(settings["max_head_tilt"])
        if "use_posture_alerts" in settings:
            self.use_posture_alerts = bool(settings["use_posture_alerts"])
        if "dark_mode" in settings:
            self.dark_mode = bool(settings["dark_mode"])
        if "language" in settings:
            self.language = settings["language"]
            
        self.save_config()
        
        if hasattr(self.gui, "apply_theme"):
            self.gui.apply_theme(self.dark_mode)
    
        return True
    
    def get_session_report(self):
        """Gera um relatório da sessão atual."""
        session_duration = time.time() - self.start_time
        hours, remainder = divmod(session_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        report = {
            "duration": f"{int(hours)}h {int(minutes)}m {int(seconds)}s",
            "fatigue_events": len(self.fatigue_events),
            "blink_rate": f"{self.session_stats['blink_rate']:.1f} piscadas/min",
            "posture_warnings": self.session_stats["posture_warnings"],
            "distance_warnings": self.session_stats["distance_warnings"],
            "break_reminders": self.session_stats["break_reminders"]
        }
        
        recommendations = []
        
        if self.session_stats["blink_rate"] < 12:  
            recommendations.append("Sua taxa de piscadas está abaixo do recomendado. Tente piscar mais conscientemente.")
        
        if len(self.fatigue_events) > 3:
            recommendations.append("Muitos eventos de fadiga detectados. Considere fazer uma pausa mais longa.")
        
        if self.session_stats["posture_warnings"] > 5:
            recommendations.append("Você recebeu vários avisos de postura. Considere ajustar sua estação de trabalho.")
        
        if self.session_stats["distance_warnings"] > 5:
            recommendations.append("Muitos avisos de proximidade da tela. Tente manter maior distância da tela.")
        
        report["recommendations"] = recommendations
        return report

    def generate_weekly_report(self):
        """Gera um relatório semanal baseado nos dados históricos."""
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)
        
        weekly_data = {
            "total_usage_time": 0,
            "fatigue_events": 0,
            "avg_blink_rate": [],
            "posture_warnings": 0,
            "distance_warnings": 0,
            "break_reminders": 0,
            "days_active": 0
        }
        
        for i in range(7):
            day = start_of_week + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            
            if day_str in self.historical_data["days"]:
                day_data = self.historical_data["days"][day_str]
                weekly_data["total_usage_time"] += day_data.get("total_session_time", 0)
                weekly_data["fatigue_events"] += len(day_data.get("fatigue_events", []))
                
                avg_blink_rate = sum(day_data.get("blink_rates", [])) / len(day_data.get("blink_rates", [1])) if day_data.get("blink_rates", []) else 0
                if avg_blink_rate > 0:
                    weekly_data["avg_blink_rate"].append(avg_blink_rate)
                    
                weekly_data["posture_warnings"] += day_data.get("posture_warnings", 0)
                weekly_data["distance_warnings"] += day_data.get("distance_warnings", 0)
                weekly_data["break_reminders"] += day_data.get("break_reminders", 0)
                weekly_data["days_active"] += 1
        
        if weekly_data["avg_blink_rate"]:
            weekly_data["avg_blink_rate"] = sum(weekly_data["avg_blink_rate"]) / len(weekly_data["avg_blink_rate"])
        else:
            weekly_data["avg_blink_rate"] = 0
        
        hours, remainder = divmod(weekly_data["total_usage_time"], 3600)
        minutes, _ = divmod(remainder, 60)
        weekly_data["formatted_usage_time"] = f"{int(hours)}h {int(minutes)}m"
        
        week_key = f"{start_of_week.strftime('%Y-%m-%d')} to {end_of_week.strftime('%Y-%m-%d')}"
        self.historical_data["weekly_summary"][week_key] = weekly_data
        
        return weekly_data

    def analyze_usage_patterns(self):
        """Analisa padrões de uso para gerar insights personalizados."""
        patterns = {
            "fatigue_peak_times": [],
            "best_blink_rate_days": [],
            "worst_posture_days": [],
            "improvement_areas": []
        }
        
        today = datetime.now()
        days_to_analyze = {}
        
        for i in range(30):
            day = today - timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            
            if day_str in self.historical_data["days"]:
                days_to_analyze[day_str] = self.historical_data["days"][day_str]
        
        if len(days_to_analyze) < 3:
            patterns["improvement_areas"].append("Dados insuficientes para análise completa. Continue usando o aplicativo para insights mais precisos.")
            return patterns
        
        fatigue_day_counts = {}
        for day_str, day_data in days_to_analyze.items():
            fatigue_events = day_data.get("fatigue_events", [])
            fatigue_count = len(fatigue_events)
            
            if fatigue_count > 0:
                fatigue_day_counts[day_str] = fatigue_count
                
                hour_counts = {}
                for timestamp in fatigue_events:
                    event_time = datetime.fromtimestamp(timestamp)
                    hour = event_time.hour
                    hour_counts[hour] = hour_counts.get(hour, 0) + 1
                
                if hour_counts:
                    peak_hour = max(hour_counts, key=hour_counts.get)
                    day_date = datetime.strptime(day_str, "%Y-%m-%d").strftime("%d/%m")
                    patterns["fatigue_peak_times"].append(f"{day_date}: {peak_hour}:00 - {peak_hour+1}:00")
        
        blink_rates = {}
        for day_str, day_data in days_to_analyze.items():
            rates = day_data.get("blink_rates", [])
            if rates:
                avg_rate = sum(rates) / len(rates)
                blink_rates[day_str] = avg_rate
        
        if blink_rates:
            sorted_blink_days = sorted(blink_rates.items(), key=lambda x: x[1], reverse=True)
            for day_str, rate in sorted_blink_days[:3]:
                day_date = datetime.strptime(day_str, "%Y-%m-%d").strftime("%d/%m")
                patterns["best_blink_rate_days"].append(f"{day_date}: {rate:.1f} piscadas/min")
            
            avg_global_blink = sum(blink_rates.values()) / len(blink_rates)
            if avg_global_blink < 15:
                patterns["improvement_areas"].append(f"Sua taxa média de piscadas ({avg_global_blink:.1f}/min) está abaixo do ideal (15-20/min). Tente fazer exercícios para os olhos regularmente.")
        
        posture_warnings = {}
        for day_str, day_data in days_to_analyze.items():
            warnings = day_data.get("posture_warnings", 0)
            if warnings > 0:
                posture_warnings[day_str] = warnings
        
        if posture_warnings:
            sorted_posture_days = sorted(posture_warnings.items(), key=lambda x: x[1], reverse=True)
            for day_str, count in sorted_posture_days[:3]:
                day_date = datetime.strptime(day_str, "%Y-%m-%d").strftime("%d/%m")
                patterns["worst_posture_days"].append(f"{day_date}: {count} avisos")
            
            if len(sorted_posture_days) >= 7:
                recent_days = sorted([datetime.strptime(d, "%Y-%m-%d") for d in posture_warnings.keys()])[-7:]
                recent_days_str = [d.strftime("%Y-%m-%d") for d in recent_days]
                recent_warnings = [posture_warnings[d] for d in recent_days_str if d in posture_warnings]
                
                if recent_warnings and sum(recent_warnings) / len(recent_warnings) > 5:
                    patterns["improvement_areas"].append("Você recebeu muitos avisos de postura na última semana. Considere ergonomia e ajuste sua cadeira/mesa.")
        
        usage_times = {}
        for day_str, day_data in days_to_analyze.items():
            usage_time = day_data.get("total_session_time", 0)
            usage_times[day_str] = usage_time
        
        if usage_times:
            avg_daily_usage = sum(usage_times.values()) / len(usage_times)
            hours, remainder = divmod(avg_daily_usage, 3600)
            minutes, _ = divmod(remainder, 60)
            
            if hours >= 4:
                patterns["improvement_areas"].append(f"Seu tempo médio de uso diário é de {int(hours)}h {int(minutes)}m. Considere fazer pausas mais frequentes e limitar o tempo de tela.")
        
        if not patterns["improvement_areas"]:
            patterns["improvement_areas"].append("Seus padrões de uso parecem adequados. Continue mantendo bons hábitos!")
        
        return patterns

    def translate_ui(self, language_code):
        """Traduz a interface do usuário para o idioma especificado."""
        translations = {
            "pt-BR": {
                "fatigue_detected": "Fadiga Detectada!",
                "normal_status": "Normal",
                "paused_status": "Pausado",
                "monitoring_status": "Monitorando",
                "distance_warning": "AVISO: Muito perto da tela!",
                "posture_warning": "AVISO: Corrija sua postura!",
                "break_reminder": "HORA DA PAUSA! Olhe para longe por 20s",
                "calibration_start": "Iniciando calibração automática...",
                "calibration_failed": "Falha na calibração - nenhum rosto detectado.",
                "calibration_complete": "Calibração concluída. Novo threshold EAR: {:.3f}"
            },
            "en-US": {
                "fatigue_detected": "Fatigue Detected!",
                "normal_status": "Normal",
                "paused_status": "Paused",
                "monitoring_status": "Monitoring",
                "distance_warning": "WARNING: Too close to screen!",
                "posture_warning": "WARNING: Fix your posture!",
                "break_reminder": "BREAK TIME! Look far away for 20s",
                "calibration_start": "Starting automatic calibration...",
                "calibration_failed": "Calibration failed - no face detected.",
                "calibration_complete": "Calibration complete. New EAR threshold: {:.3f}"
            }
        }
        
        if language_code not in translations:
            language_code = "en-US"
            
        self.translations = translations[language_code]
        
        if hasattr(self.gui, "update_translations"):
            self.gui.update_translations(self.translations)
            
        return True
    
    def generate_monthly_report(self):
        """Gera um relatório mensal baseado nos dados históricos."""
        today = datetime.now()
        current_month = today.month
        current_year = today.year
        
        start_of_month = datetime(current_year, current_month, 1)
        if current_month == 12:
            end_of_month = datetime(current_year + 1, 1, 1) - timedelta(days=1)
        else:
            end_of_month = datetime(current_year, current_month + 1, 1) - timedelta(days=1)
        
        monthly_data = {
            "total_usage_time": 0,
            "fatigue_events": 0,
            "avg_blink_rate": [],
            "posture_warnings": 0,
            "distance_warnings": 0,
            "break_reminders": 0,
            "days_active": 0
        }
        
        current_day = start_of_month
        while current_day <= end_of_month and current_day <= today:
            day_str = current_day.strftime("%Y-%m-%d")
            
            if day_str in self.historical_data["days"]:
                day_data = self.historical_data["days"][day_str]
                monthly_data["total_usage_time"] += day_data.get("total_session_time", 0)
                monthly_data["fatigue_events"] += len(day_data.get("fatigue_events", []))
                
                avg_blink_rate = sum(day_data.get("blink_rates", [])) / len(day_data.get("blink_rates", [1])) if day_data.get("blink_rates", []) else 0
                if avg_blink_rate > 0:
                    monthly_data["avg_blink_rate"].append(avg_blink_rate)
                    
                monthly_data["posture_warnings"] += day_data.get("posture_warnings", 0)
                monthly_data["distance_warnings"] += day_data.get("distance_warnings", 0)
                monthly_data["break_reminders"] += day_data.get("break_reminders", 0)
                monthly_data["days_active"] += 1
            
            current_day += timedelta(days=1)
        
        if monthly_data["avg_blink_rate"]:
            monthly_data["avg_blink_rate"] = sum(monthly_data["avg_blink_rate"]) / len(monthly_data["avg_blink_rate"])
        else:
            monthly_data["avg_blink_rate"] = 0
        
        hours, remainder = divmod(monthly_data["total_usage_time"], 3600)
        minutes, _ = divmod(remainder, 60)
        monthly_data["formatted_usage_time"] = f"{int(hours)}h {int(minutes)}m"
        
        month_key = f"{current_year}-{current_month:02d}"
        self.historical_data["monthly_summary"][month_key] = monthly_data
        
        self.save_historical_data()
        
        return monthly_data
    
    def export_data(self, export_path=None):
        """Exporta os dados históricos para um arquivo JSON."""
        if export_path is None:
            export_path = os.path.join("data", "export_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
        
        try:
            export_data = {
                "user_data": self.historical_data,
                "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "software_version": "1.0.0",
                "settings": {
                    "ear_threshold": self.ear_threshold,
                    "consecutive_frames": self.consecutive_frames,
                    "break_interval": self.break_interval,
                    "min_face_distance": self.min_face_distance,
                    "max_head_tilt": self.max_head_tilt
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=4)
                
            return True, export_path
        except Exception as e:
            print(f"Erro ao exportar dados: {e}")
            return False, str(e)
    
    def import_data(self, import_path):
        """Importa dados históricos de um arquivo JSON."""
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            if "user_data" in import_data:
                for day, day_data in import_data["user_data"].get("days", {}).items():
                    if day not in self.historical_data["days"]:
                        self.historical_data["days"][day] = day_data
                    else:
                        self.historical_data["days"][day]["total_session_time"] += day_data.get("total_session_time", 0)
                        self.historical_data["days"][day]["fatigue_events"].extend(day_data.get("fatigue_events", []))
                        self.historical_data["days"][day]["blink_rates"].extend(day_data.get("blink_rates", []))
                        self.historical_data["days"][day]["posture_warnings"] += day_data.get("posture_warnings", 0)
                        self.historical_data["days"][day]["distance_warnings"] += day_data.get("distance_warnings", 0)
                        self.historical_data["days"][day]["break_reminders"] += day_data.get("break_reminders", 0)
                
                for week, week_data in import_data["user_data"].get("weekly_summary", {}).items():
                    if week not in self.historical_data["weekly_summary"]:
                        self.historical_data["weekly_summary"][week] = week_data
                
                for month, month_data in import_data["user_data"].get("monthly_summary", {}).items():
                    if month not in self.historical_data["monthly_summary"]:
                        self.historical_data["monthly_summary"][month] = month_data
                
                self.save_historical_data()
                return True, "Dados importados com sucesso"
            else:
                return False, "Formato de arquivo inválido"
        except Exception as e:
            print(f"Erro ao importar dados: {e}")
            return False, str(e)

    def get_recommendations(self):
        """Gera recomendações personalizadas com base nos dados históricos."""
        recommendations = []
        
        if not self.historical_data["days"]:
            recommendations.append("Continue usando o aplicativo para receber recomendações personalizadas.")
            return recommendations
        
        blink_rates = []
        for day_data in self.historical_data["days"].values():
            blink_rates.extend(day_data.get("blink_rates", []))
        
        if blink_rates:
            avg_blink_rate = sum(blink_rates) / len(blink_rates)
            if avg_blink_rate < 12:
                recommendations.append("Sua taxa média de piscadas está abaixo do recomendado. Pratique a regra 20-20-20: a cada 20 minutos, olhe para algo a 20 pés (6 metros) de distância por 20 segundos.")
        
        fatigue_count = sum(len(day_data.get("fatigue_events", [])) for day_data in self.historical_data["days"].values())
        total_usage_time = sum(day_data.get("total_session_time", 0) for day_data in self.historical_data["days"].values())
        
        if total_usage_time > 0:
            if fatigue_rate > 2:
                recommendations.append("Você apresenta sinais frequentes de fadiga ocular. Considere aumentar o brilho do ambiente e reduzir o brilho da tela.")
        
        posture_warnings = sum(day_data.get("posture_warnings", 0) for day_data in self.historical_data["days"].values())
        if posture_warnings > 20:
            recommendations.append("Você recebe muitos avisos de postura. Considere ajustar a altura da cadeira/mesa e manter uma postura ergonômica.")
        
        distance_warnings = sum(day_data.get("distance_warnings", 0) for day_data in self.historical_data["days"].values())
        if distance_warnings > 20:
            recommendations.append("Você tende a se aproximar muito da tela. Tente manter uma distância de pelo menos 50-70cm do monitor.")
        
        days_count = len(self.historical_data["days"])
        if days_count > 0:
            avg_daily_time = total_usage_time / days_count
            if avg_daily_time > 5 * 3600:  
                recommendations.append("Seu tempo médio diário de uso de tela é elevado. Tente fazer pausas mais frequentes e alternar entre tarefas digitais e não-digitais.")
        
        if not recommendations:
            recommendations.append("Seus hábitos de uso de tela parecem adequados. Continue mantendo bons hábitos para a saúde ocular!")
        
        return recommendations