from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from datetime import datetime

def generate_pdf_report(fatigue_events, output_dir="reports"):
    """Gera um relatório em PDF com os eventos de fadiga."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"report_{timestamp}.pdf")

    try:
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica", 12)

        c.drawString(50, 750, "Relatório de Monitoramento Ocular - EyeCareAI")
        c.drawString(50, 730, f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        c.drawString(50, 700, f"Número de eventos de fadiga: {len(fatigue_events)}")

        if fatigue_events:
            c.drawString(50, 680, "Horários dos eventos de fadiga:")
            y = 660
            for i, event_time in enumerate(fatigue_events, 1):
                event_str = datetime.fromtimestamp(event_time).strftime('%d/%m/%Y %H:%M:%S')
                c.drawString(70, y, f"Evento {i}: {event_str}")
                y -= 15
                if y < 50:  
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750
        else:
            c.drawString(50, 680, "Nenhum evento de fadiga detectado.")

        c.save()
        return output_path
    except Exception as e:
        print(f"[ERRO AO GERAR RELATÓRIO]: {e}")
        return None