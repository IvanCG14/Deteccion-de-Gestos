import myo
import tkinter as tk
from tkinter import ttk
import threading
import collections
import numpy as np
import time

import os
from pathlib import Path

# 1. Obtener la ruta de la carpeta raíz del proyecto (subiendo un nivel desde 'getdata')
# __file__ es la ubicación de dataset_creator_myo.py
BASE_DIR = Path(__file__).resolve().parent.parent 

# 2. Construir la ruta al SDK de forma relativa
sdk_path = os.path.join(BASE_DIR, "MYO_armband_SDK", "myo-sdk-win-0.9.0")

class Listener(myo.DeviceListener):
    def __init__(self, n_samples=100):
        super().__init__()
        self.n_samples = n_samples
        self.lock = threading.Lock()

        self.orientation_data = collections.deque(np.zeros((n_samples, 4)), maxlen=n_samples)
        self.acceleration_data = collections.deque(np.zeros((n_samples, 3)), maxlen=n_samples)
        self.gyroscope_data = collections.deque(np.zeros((n_samples, 3)), maxlen=n_samples)
        self.emg_data = [collections.deque(np.zeros(n_samples), maxlen=n_samples) for _ in range(8)]

        self.connected = False

    def on_connected(self, event):
        print(f"Myo conectado: {event.device_name}")
        event.device.vibrate(myo.VibrationType.short)
        event.device.stream_emg(True)  # Solo activa EMG, IMU viene por defecto
        self.connected = True

    def on_disconnected(self, event):
        print("Myo desconectado.")
        self.connected = False

    def on_paired(self, event):
        print(f"Emparejado con {event.device_name}")

    def on_unpaired(self, event):
        return False

    def on_orientation(self, event):
        with self.lock:
            self.orientation_data.append(event.orientation)
            self.acceleration_data.append(event.acceleration)
            self.gyroscope_data.append(event.gyroscope)

    def on_emg(self, event):
        with self.lock:
            for i in range(8):
                self.emg_data[i].append(event.emg[i])

    def get_data_snapshot(self):
        with self.lock:
            return (
                list(self.orientation_data),
                list(self.acceleration_data),
                list(self.gyroscope_data),
                [list(ch) for ch in self.emg_data]
            )

class MyoApp(tk.Tk):
    def __init__(self, listener, hub):
        super().__init__()
        self.title("Myo Sensor Visualizer")
        self.listener = listener
        self.hub = hub

        self.show_emg = tk.BooleanVar(value=True)
        self.show_orientation = tk.BooleanVar(value=True)
        self.show_accelerometer = tk.BooleanVar(value=True)
        self.show_gyroscope = tk.BooleanVar(value=True)

        ttk.Checkbutton(self, text="EMG", variable=self.show_emg).grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(self, text="Orientation", variable=self.show_orientation).grid(row=0, column=1, sticky='w')
        ttk.Checkbutton(self, text="Accelerometer", variable=self.show_accelerometer).grid(row=0, column=2, sticky='w')
        ttk.Checkbutton(self, text="Gyroscope", variable=self.show_gyroscope).grid(row=0, column=3, sticky='w')

        self.labels = {
            'emg': ttk.Label(self, text="EMG: []"),
            'orientation': ttk.Label(self, text="Orientation: ()"),
            'accelerometer': ttk.Label(self, text="Accelerometer: ()"),
            'gyroscope': ttk.Label(self, text="Gyroscope: ()"),
        }
        self.labels['emg'].grid(row=1, column=0, columnspan=4, sticky='w')
        self.labels['orientation'].grid(row=2, column=0, columnspan=4, sticky='w')
        self.labels['accelerometer'].grid(row=3, column=0, columnspan=4, sticky='w')
        self.labels['gyroscope'].grid(row=4, column=0, columnspan=4, sticky='w')

        self.update_gui()
        threading.Thread(target=self.run_hub, daemon=True).start()

    def run_hub(self):
        while self.hub.run(self.listener.on_event, 500):
            time.sleep(0.01)

    def update_gui(self):
        orientation, acceleration, gyroscope, emg = self.listener.get_data_snapshot()

        if self.show_emg.get():
            # Muestra último valor EMG de 8 canales
            last_emg = [ch[-1] if ch else 0 for ch in emg]
            self.labels['emg'].config(text=f"EMG: {last_emg}")
        else:
            self.labels['emg'].config(text="EMG: [hidden]")

        if self.show_orientation.get():
            if orientation:
                x, y, z, w = orientation[-1]
                self.labels['orientation'].config(text=f"Orientation: x={x:.3f} y={y:.3f} z={z:.3f} w={w:.3f}")
            else:
                self.labels['orientation'].config(text="Orientation: no data")
        else:
            self.labels['orientation'].config(text="Orientation: [hidden]")

        if self.show_accelerometer.get():
            if acceleration:
                x, y, z = acceleration[-1]
                self.labels['accelerometer'].config(text=f"Accelerometer: x={x:.3f} y={y:.3f} z={z:.3f}")
            else:
                self.labels['accelerometer'].config(text="Accelerometer: no data")
        else:
            self.labels['accelerometer'].config(text="Accelerometer: [hidden]")

        if self.show_gyroscope.get():
            if gyroscope:
                x, y, z = gyroscope[-1]
                self.labels['gyroscope'].config(text=f"Gyroscope: x={x:.3f} y={y:.3f} z={z:.3f}")
            else:
                self.labels['gyroscope'].config(text="Gyroscope: no data")
        else:
            self.labels['gyroscope'].config(text="Gyroscope: [hidden]")

        self.after(100, self.update_gui)

if __name__ == '__main__':
    myo.init(sdk_path=sdk_path)

    hub = myo.Hub()
    listener = Listener()

    app = MyoApp(listener, hub)
    app.mainloop()
