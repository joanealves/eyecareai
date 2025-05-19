import tkinter as tk
from src.pdf_report import generate_pdf_report

class EyeCareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EyeCareAI - Monitoramento Ocular")
        self.fatigue_events = []
        
        self.label_status = tk.Label(root, text="Status: Aguardando...", font=("Arial", 14))
        self.label_status.pack(pady=10)
        
        self.label_ear = tk.Label(root, text="EAR: 0.00", font=("Arial", 12))
        self.label_ear.pack(pady=5)
        
        self.label_events = tk.Label(root, text="Eventos de Fadiga: 0", font=("Arial", 12))
        self.label_events.pack(pady=5)
        
        self.button_report = tk.Button(root, text="Gerar Relatório PDF", command=self.generate_report)
        self.button_report.pack(pady=10)
    
    def update_fatigue_status(self, status):
        """Atualiza o status de fadiga na interface."""
        self.label_status.config(text=f"Status: {status}")
    
    def update_metrics(self, ear, event_count):
        """Atualiza as métricas na interface."""
        self.label_ear.config(text=f"EAR: {ear:.2f}")
        self.label_events.config(text=f"Eventos de Fadiga: {event_count}")
        self.fatigue_events = self.fatigue_events[:event_count]
    
    def generate_report(self):
        """Gera um relatório em PDF."""
        report_path = generate_pdf_report(self.fatigue_events)
        self.label_status.config(text=f"Relatório gerado: {report_path}")