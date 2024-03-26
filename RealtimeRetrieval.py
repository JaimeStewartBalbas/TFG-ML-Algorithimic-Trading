import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
import threading
import yfinance as yf

class RealTimeCandlestick:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Candlestick Chart")
        self.create_widgets()
        self.update_graph()  # Initial graph update

    def create_widgets(self):
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

    def update_graph(self):
        ibex_data = yf.download("^IBEX", period="2h", interval="1m",progress=False)
        mpf.plot(ibex_data, type='candle', style='charles', ax=self.ax, volume=False, tight_layout=True)
        self.canvas.draw()

        # Schedule next update after 10 seconds
        self.root.after(10000, self.update_graph)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeCandlestick(root)
    app.run()
