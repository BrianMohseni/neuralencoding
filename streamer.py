import argparse
import signal
import threading
import time
from typing import List, Dict

import numpy as np
import tensorflow as tf
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer


class GracefulKiller:
    def __init__(self):
        self._kill_now = False
        signal.signal(signal.SIGINT, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

    def _on_signal(self, signum, frame):
        self._kill_now = True

    @property
    def kill_now(self) -> bool:
        return self._kill_now


class StreamBuffer:
    def __init__(self, eeg_channels: int, optics_channels: int, target_rate: float = 256.0, window_seconds: float = 1.0):
        self.eeg_channels = eeg_channels
        self.optics_channels = optics_channels
        self.target_rate = target_rate
        self.window_seconds = window_seconds
        self.rows: List[Dict] = []
        self.seq = 0

    def _pad(self, values: List[float], target_len: int) -> List[float]:
        if len(values) >= target_len:
            return list(values[:target_len])
        return list(values) + [np.nan] * (target_len - len(values))

    def add_eeg(self, values: List[float]):
        ts = time.time()
        self.rows.append({
            "timestamp": ts,
            "eeg": self._pad([float(v) for v in values], self.eeg_channels),
            "optics": [np.nan] * self.optics_channels
        })
        self.seq += 1

    def add_optics(self, values: List[float]):
        ts = time.time()
        self.rows.append({
            "timestamp": ts,
            "eeg": [np.nan] * self.eeg_channels,
            "optics": self._pad([float(v) for v in values], self.optics_channels)
        })
        self.seq += 1

    def get_window(self):
        if not self.rows:
            return None

        timestamps = np.array([row["timestamp"] for row in self.rows])
        eeg_data = np.array([row["eeg"] for row in self.rows])
        optics_data = np.array([row["optics"] for row in self.rows])

        min_len = min(len(timestamps), len(eeg_data), len(optics_data))
        timestamps = timestamps[:min_len]
        eeg_data = eeg_data[:min_len]
        optics_data = optics_data[:min_len]

        timestamps, unique_idx = np.unique(timestamps, return_index=True)
        eeg_data = eeg_data[unique_idx]
        optics_data = optics_data[unique_idx]

        end_time = timestamps.max()
        start_time = end_time - self.window_seconds
        resampled_timestamps = np.arange(start_time, end_time, 1.0 / self.target_rate)

        eeg_resampled = np.zeros((len(resampled_timestamps), self.eeg_channels))
        optics_resampled = np.zeros((len(resampled_timestamps), self.optics_channels))

        for ch in range(self.eeg_channels):
            mask = ~np.isnan(eeg_data[:, ch])
            if mask.sum() > 1:
                eeg_resampled[:, ch] = np.interp(resampled_timestamps, timestamps[mask], eeg_data[mask, ch])
            else:
                eeg_resampled[:, ch] = np.nan

        for ch in range(self.optics_channels):
            mask = ~np.isnan(optics_data[:, ch])
            if mask.sum() > 1:
                optics_resampled[:, ch] = np.interp(resampled_timestamps, timestamps[mask], optics_data[mask, ch])
            else:
                optics_resampled[:, ch] = np.nan

        features = np.concatenate([eeg_resampled, optics_resampled], axis=1)
        return features


class Handler:
    def __init__(self, buffer: StreamBuffer):
        self.buffer = buffer

    def handle_eeg(self, address: str, *args):
        self.buffer.add_eeg(list(args))

    def handle_optics(self, address: str, *args):
        self.buffer.add_optics(list(args))


def run_server(listen_ip: str, listen_port: int, handler: Handler):
    dispatcher = Dispatcher()
    dispatcher.map("/muse/eeg", handler.handle_eeg)
    dispatcher.map("/muse/optics", handler.handle_optics)

    server = ThreadingOSCUDPServer((listen_ip, listen_port), dispatcher)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def parse_args():
    p = argparse.ArgumentParser(
        description="Stream EEG/optics OSC data, interpolate, and classify with model.keras."
    )
    p.add_argument("--listen-ip", type=str, default="0.0.0.0", help="IP to bind (default is local)")
    p.add_argument("--listen-port", type=int, default=8890, help="Port to bind (default: 8890)")
    p.add_argument("--eeg-channels", type=int, default=4, help="Number of EEG channels (default: 4)")
    p.add_argument("--optics-channels", type=int, default=4, help="Number of optics channels (default: 16)")
    p.add_argument("--target-rate", type=float, default=256.0, help="Resampling rate in Hz (default: 256 Hz)")
    p.add_argument("--window-seconds", type=float, default=2.0, help="Length of sliding window in seconds")
    p.add_argument("--model", type=str, default="model.keras", help="Path to Keras model file")
    p.add_argument("--interval", type=float, default=0.25, help="Seconds between predictions")
    return p.parse_args()


def main():
    args = parse_args()
    killer = GracefulKiller()
    model = tf.keras.models.load_model(args.model)

    buffer = StreamBuffer(
        eeg_channels=args.eeg_channels,
        optics_channels=args.optics_channels,
        target_rate=args.target_rate,
        window_seconds=args.window_seconds,
    )

    handler = Handler(buffer)
    server, thread = run_server(args.listen_ip, args.listen_port, handler)

    print(
        f"Streaming OSC on {args.listen_ip}:{args.listen_port}, using model {args.model}, "
        f"EEG ch={args.eeg_channels}, Optics ch={args.optics_channels}, Target rate={args.target_rate} Hz"
    )

    try:
        while not killer.kill_now:
            features = buffer.get_window()
            if features is not None and not np.isnan(features).all():
                X = np.expand_dims(features, axis=0)
                preds = model.predict(X[:, -1], verbose=0)
                print(f"[Prediction] {preds}")
            time.sleep(args.interval)
    finally:
        print("Stopping...")
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass
        try:
            thread.join(timeout=2.0)
        except Exception:
            pass


if __name__ == "__main__":
    main()
