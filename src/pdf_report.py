from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from datetime import datetime

def generate_pdf_report(fatigue_events, output_dir=os.path.join("..", "reports")):
    """Gera um relatório em PDF com os eventos de fadiga."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"report_{timestamp}.pdf")
    
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    
    c.drawString(100, 750, "Relatório de Monitoramento Ocular - EyeCareAI")
    c.drawString(100, 730, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    c.drawString(100, 700, f"Número de eventos de fadiga: {len(fatigue_events)}")
    
    if fatigue_events:
        c.drawString(100, 680, "Horários dos eventos de fadiga:")
        y = 660
        for i, event_time in enumerate(fatigue_events, 1):
            event_str = datetime.fromtimestamp(event_time).strftime('%d/%m/%Y %H:%M:%S')
            c.drawString(120, y, f"Evento {i}: {event_str}")
            y -= 20
    else:
        c.drawString(100, 680, "Nenhum evento de fadiga detectado.")
    
    c.save()
    return output_path