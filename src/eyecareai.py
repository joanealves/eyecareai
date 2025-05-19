import sys
import os
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import tkinter as tk
from src.video_processing import VideoProcessor
from src.gui import EyeCareGUI
from src.pdf_report import generate_pdf_report

def main():
    logger.info("Iniciando o EyeCareAI")
    os.makedirs("reports", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    logger.info("Pastas criadas")

    root = tk.Tk()
    gui = EyeCareGUI(root)
    logger.info("Interface gráfica inicializada")

    video_processor = VideoProcessor(gui)
    logger.info("Processador de vídeo inicializado")

    def on_closing():
        logger.info("Encerrando o programa")
        video_processor.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    video_processor.start()
    logger.info("Iniciando loop principal")
    root.mainloop()

if __name__ == "__main__":
    main()