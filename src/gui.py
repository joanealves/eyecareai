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
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f4f8")
        self.fatigue_events = []
        self.ear_history = []
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

        self.label_events = tk.Label(self.metrics_frame, text="Eventos de Fadiga: 0", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_events.pack(pady=5)

        self.label_session_time = tk.Label(self.metrics_frame, text="Tempo de Sessão: 00:00:00", font=("Helvetica", 12), bg="#ffffff", fg="#333333")
        self.label_session_time.pack(pady=5)

        self.button_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.button_frame.pack(pady=5)

        self.button_report = tk.Button(self.button_frame, text="Gerar Relatório PDF", command=self.generate_report, font=("Helvetica", 10), bg="#4CAF50", fg="white", width=15)
        self.button_report.pack(side="left", padx=5)

        self.button_export_csv = tk.Button(self.button_frame, text="Exportar CSV", command=self.export_csv, font=("Helvetica", 10), bg="#2196F3", fg="white", width=15)
        self.button_export_csv.pack(side="left", padx=5)

        self.button_pause = tk.Button(self.button_frame, text="Pausar", command=self.toggle_pause, font=("Helvetica", 10), bg="#FF9800", fg="white", width=15)
        self.button_pause.pack(side="left", padx=5)

        self.graph_frame = tk.Frame(self.main_frame, bg="#f0f4f8")
        self.graph_frame.pack(pady=10, fill="both", expand=True)

        self.fig, self.ax = plt.subplots(figsize=(6, 3))
        self.ax.set_title("EAR ao Longo do Tempo", fontsize=12)
        self.ax.set_xlabel("Tempo (s)", fontsize=10)
        self.ax.set_ylabel("EAR", fontsize=10)
        self.ax.grid(True, linestyle="--", alpha=0.7)
        self.ax.set_ylim(0, 0.5)
        self.line, = self.ax.plot([], [], color="#2196F3", label="EAR")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.fig2, self.ax2 = plt.subplots(figsize=(6, 2))
        self.ax2.set_title("Eventos de Fadiga por Hora", fontsize=12)
        self.ax2.set_xlabel("Hora", fontsize=10)
        self.ax2.set_ylabel("Eventos", fontsize=10)
        self.ax2.set_xlim(0, 23)
        self.ax2.set_ylim(0, 10)  

        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self.graph_frame)
        self.canvas2.get_tk_widget().pack(fill="both", expand=True)

        self.start_time = datetime.now()

    def update_fatigue_status(self, status):
        """Atualiza o status de fadiga na interface."""
        self.label_status.config(text=f"Status: {status}")

    def update_metrics(self, ear, event_count):
        """Atualiza as métricas e os gráficos na interface."""
        if not self.running:
            return
        self.label_ear.config(text=f"EAR: {ear:.2f}")
        self.label_events.config(text=f"Eventos de Fadiga: {event_count}")
        self.fatigue_events = self.fatigue_events[:event_count]

        elapsed_time = datetime.now() - self.start_time
        hours, remainder = divmod(elapsed_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.label_session_time.config(text=f"Tempo de Sessão: {hours:02d}:{minutes:02d}:{seconds:02d}")

        self.ear_history.append(ear)
        if len(self.ear_history) > self.max_ear_points:
            self.ear_history.pop(0)
        times = np.arange(len(self.ear_history))
        self.line.set_data(times, self.ear_history)
        self.ax.set_xlim(0, max(len(self.ear_history), 1))
        self.canvas.draw()

        hours = [datetime.fromtimestamp(t).hour for t in self.fatigue_events]
        hour_counts = np.bincount(hours, minlength=24)
        self.ax2.clear()
        self.ax2.bar(range(24), hour_counts, color="#FF9800")
        self.ax2.set_title("Eventos de Fadiga por Hora")
        self.ax2.set_xlabel("Hora")
        self.ax2.set_ylabel("Eventos")
        self.ax2.set_xlim(0, 23)
        self.ax2.set_ylim(0, max(hour_counts.max(), 1))
        self.canvas2.draw()

    def generate_report(self):
        """Gera um relatório em PDF."""
        report_path = generate_pdf_report(self.fatigue_events)
        messagebox.showinfo("Sucesso", f"Relatório gerado: {report_path}")

    def export_csv(self):
        """Exporta os eventos de fadiga como CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("..", "reports", f"fatigue_events_{timestamp}.csv")
        df = pd.DataFrame({
            "Evento": [f"Evento {i+1}" for i in range(len(self.fatigue_events))],
            "Horário": [datetime.fromtimestamp(t).strftime('%d/%m/%Y %H:%M:%S') for t in self.fatigue_events]
        })
        df.to_csv(output_path, index=False)
        messagebox.showinfo("Sucesso", f"CSV exportado: {output_path}")

    def toggle_pause(self):
        """Pausa ou continua o monitoramento."""
        self.running = not self.running
        self.button_pause.config(text="Continuar" if not self.running else "Pausar")
        self.label_status.config(text="Status: Pausado" if not self.running else "Status: Monitorando")