import tkinter as tk
from tkinter import messagebox
from src.pdf_report import generate_pdf_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
from datetime import datetime
import os

class EyeCareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EyeCareAI - Monitoramento Ocular")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f4f8")
        
        self.fatigue_events = []
        self.ear_history = []
        self.blink_rates = []
        self.max_ear_points = 100  
        self.running = True

        self.main_frame = tk.Frame(self.root, bg="#f0f4f8")
        self.main_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.metrics_frame = tk.Frame(self.main_frame, bg="#ffffff", bd=2, relief="groove")
        self.metrics_frame.pack(pady=5, fill="x")

        self.label_status = tk.Label(self.metrics_frame, text="Status: Aguardando...", font=("Helvetica", 14, "bold"), bg="#ffffff", fg="#333333")
        self.label_status.pack(pady=5)

        self.label_ear = tk.Label(self.metrics_frame, text="EAR: 0.00", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_ear.pack(pady=5)

        self.label_blink_rate = tk.Label(self.metrics_frame, text="Piscadas/min: 0.0", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_blink_rate.pack(pady=5)

        self.label_events = tk.Label(self.metrics_frame, text="Eventos de Fadiga: 0", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_events.pack(pady=5)

        self.label_session_time = tk.Label(self.metrics_frame, text="Tempo de Sessão: 00:00:00", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_session_time.pack(pady=5)

        self.label_fps = tk.Label(self.metrics_frame, text="FPS: 0", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_fps.pack(pady=5)

        self.button_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.button_frame.pack(pady=5)

        self.button_report = tk.Button(
            self.button_frame,
            text="Gerar Relatório PDF",
            command=self.generate_report,
            font=("Helvetica", 10),
            bg="#4CAF50",
            fg="white",
            width=18
        )
        self.button_report.pack(side="left", padx=5)
        self.button_report.pack(side="left", padx=5)

        self.button_export_csv = tk.Button(self.button_frame, text="Exportar CSV", command=self.export_csv, font=("Helvetica", 10), bg="#2196F3", fg="white", width=15)
        self.button_export_csv.pack(side="left", padx=5)

        self.button_pause = tk.Button(self.button_frame, text="Pausar", command=self.toggle_pause, font=("Helvetica", 10), bg="#FF9800", fg="white", width=15)
        self.button_pause.pack(side="left", padx=5)

        self.button_weekly_report = tk.Button(self.button_frame, text="Relatório Semanal", command=self.generate_weekly_report, font=("Helvetica", 10), bg="#9c27b0", fg="white", width=15)
        self.button_weekly_report.pack(side="left", padx=5)

        self.lang_var = tk.StringVar(value="pt-BR")
        lang_menu = tk.OptionMenu(self.button_frame, self.lang_var, "pt-BR", "en-US", "es-ES", command=self.change_language)
        lang_menu.pack(side="right", padx=5)

        self.graph_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.graph_frame.pack(pady=10, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 2))
        self.ax.set_title("EAR ao Longo do Tempo", fontsize=10)
        self.ax.set_xlabel("Tempo (s)", fontsize=8)
        self.ax.set_ylabel("EAR", fontsize=8)
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.set_ylim(0, 0.5)
        self.line_ear, = self.ax.plot([], [], color="#2196F3", label="EAR")
        self.ax.legend()

        self.canvas_ear = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas_ear.get_tk_widget().pack(fill="both", expand=True)

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 2))
        self.ax2.set_title("Taxa de Piscadas", fontsize=10)
        self.ax2.set_xlabel("Tempo", fontsize=8)
        self.ax2.set_ylabel("Piscadas/min", fontsize=8)
        self.ax2.grid(True)
        self.line_blink, = self.ax2.plot([], [], color="#4CAF50", label="Piscadas/min")
        self.ax2.legend()

        self.canvas_blink = FigureCanvasTkAgg(self.fig2, master=self.graph_frame)
        self.canvas_blink.get_tk_widget().pack(fill="both", expand=True)

        self.start_time = datetime.now()

    def apply_theme(self, dark_mode):
        if dark_mode:
            self.root.configure(bg="#1e1e1e")
            self.main_frame.configure(bg="#1e1e1e")
            self.metrics_frame.configure(bg="#2d2d2d")
            for widget in [self.label_status, self.label_ear, self.label_blink_rate,
                           self.label_events, self.label_session_time, self.label_fps]:
                widget.configure(bg="#2d2d2d", fg="#ffffff")
            self.button_frame.configure(bg="#1e1e1e")
        else:
            self.root.configure(bg="#f0f4f8")
            self.main_frame.configure(bg="#f0f4f8")
            self.metrics_frame.configure(bg="#ffffff")
            for widget in [self.label_status, self.label_ear, self.label_blink_rate,
                           self.label_events, self.label_session_time, self.label_fps]:
                widget.configure(bg="#ffffff", fg="#333333")
            self.button_frame.configure(bg="#f0f4f8")

    def change_language(self, lang):
        self.video_processor.translate_ui(lang)
        translations = self.video_processor.translations
        self.label_status.config(text=f"Status: {translations['normal_status']}")
        self.button_report.config(text="Gerar Relatório PDF")
        self.button_export_csv.config(text="Exportar CSV")
        self.button_pause.config(text="Pausar")

    def update_all_metrics(self, ear, event_count, blink_rate, face_distance, posture_warning, distance_warning, break_reminder, fps):
        if not self.running:
            return

        self.label_ear.config(text=f"EAR: {ear:.2f}")
        self.label_events.config(text=f"Eventos de Fadiga: {event_count}")
        self.label_blink_rate.config(text=f"Piscadas/min: {blink_rate:.1f}")
        self.label_fps.config(text=f"FPS: {fps:.1f}")

        elapsed_time = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.label_session_time.config(text=f"Tempo de Sessão: {hours:02d}:{minutes:02d}:{seconds:02d}")

        self.ear_history.append(ear)
        if len(self.ear_history) > self.max_ear_points:
            self.ear_history.pop(0)
        times = np.arange(len(self.ear_history))

        self.line_ear.set_data(times, self.ear_history)
        self.ax.set_xlim(0, max(len(self.ear_history), 1))
        self.canvas_ear.draw()

        self.blink_rates.append(blink_rate)
        if len(self.blink_rates) > self.max_ear_points:
            self.blink_rates.pop(0)
        times_blink = np.arange(len(self.blink_rates))

        self.line_blink.set_data(times_blink, self.blink_rates)
        self.ax2.set_xlim(0, max(len(self.blink_rates), 1))
        self.ax2.set_ylim(0, max(self.blink_rates + [15]))
        self.canvas_blink.draw()

    def update_fatigue_status(self, status):
        self.label_status.config(text=f"Status: {status}")

    def generate_report(self):
        try:
            fatigue_events = self.video_processor.fatigue_events
            report_path = generate_pdf_report(fatigue_events, output_dir="reports")
            
            if report_path and os.path.exists(report_path):
                messagebox.showinfo("Sucesso", f"Relatório gerado:\n{report_path}")
            else:
                messagebox.showerror("Erro", "Falha ao gerar o relatório.")
        except Exception as e:
            print(f"[ERRO NA GERAÇÃO DO RELATÓRIO] {e}")
            messagebox.showerror("Erro", "Não foi possível gerar o relatório.")

    def export_csv(self):
        try:
            fatigue_events = self.video_processor.fatigue_events

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("reports", f"fatigue_events_{timestamp}.csv")

            df = pd.DataFrame({
                "Evento": [f"Evento {i+1}" for i in range(len(fatigue_events))],
                "Horário": [datetime.fromtimestamp(t).strftime('%d/%m/%Y %H:%M:%S') for t in fatigue_events]
            })
            df.to_csv(output_path, index=False)
            messagebox.showinfo("Sucesso", f"CSV exportado: {output_path}")
        except Exception as e:
            print(f"[ERRO AO EXPORTAR CSV] {e}")
            messagebox.showerror("Erro", "Não foi possível exportar o CSV.")
            
    def generate_weekly_report(self):
        weekly_data = self.video_processor.generate_weekly_report()
        messagebox.showinfo("Relatório Semanal", f"Total de uso: {weekly_data['formatted_usage_time']}\nFadiga: {weekly_data['fatigue_events']} eventos")

    def toggle_pause(self):
        self.running = not self.running
        self.button_pause.config(text="Continuar" if not self.running else "Pausar")
        self.label_status.config(text="Status: Pausado" if not self.running else "Status: Monitorando")