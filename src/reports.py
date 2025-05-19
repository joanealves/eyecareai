def generate_report(fatigue_times, start_times, output_path="reports/report.pdf"):
    os.makedirs("reports", exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Relatório EyeCareAI")
    c.drawString(100, 730, f"Data: {time.strftime('%d/%m/%Y %H:%M:%S')}")
    
    total_time = start_times[-1] if start_times else 0
    fatigue_duration = sum([t for t in fatigue_times if t > 3.0])  
    c.drawString(100, 710, f"Tempo Total de Monitoramento: {total_time:.2f} segundos")
    c.drawString(100, 690, f"Tempo de Fadiga Detectada: {fatigue_duration:.2f} segundos")
    
    c.save()