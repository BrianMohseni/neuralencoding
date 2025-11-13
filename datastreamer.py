from collections import deque
import time
import threading


class DataStreamer:
    def __init__(self, window_size=512):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.lock = threading.Lock()

    def update(self, addr, *args):
        with self.lock:
            self.data_buffer.append(args)

    def get_window(self):
        with self.lock:
            return list(self.data_buffer)

    def get_latest(self):
        with self.lock:
            if len(self.data_buffer) > 0:
                return self.data_buffer[-1]
            return None


class DataRecorder:
    def __init__(self, record_time=2):
        self.record_time = record_time
        self.recorded_data = []
        self.is_recording = False
        self.lock = threading.Lock()
        self.start_time = None

    def start_recording(self):
        with self.lock:
            self.is_recording = True
            self.recorded_data = []
            self.start_time = time.time()

    def update(self, addr, *args):
        with self.lock:
            if self.is_recording:
                elapsed = time.time() - self.start_time
                if elapsed <= self.record_time:
                    self.recorded_data.append((elapsed, args))
                else:
                    self.is_recording = False

    def stop_recording(self):
        with self.lock:
            self.is_recording = False

    def get_recorded_data(self):
        with self.lock:
            return self.recorded_data.copy()

    def is_recording_active(self):
        with self.lock:
            return self.is_recording

