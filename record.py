import argparse
import csv
import signal
import threading
import time
from typing import List, Dict
import numpy as np

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


class CsvRecorder:
    def __init__(self, path: str, eeg_channels: int, optics_channels: int, target_rate: float = 256.0):
        self.path = path
        self.eeg_channels = int(eeg_channels)
        self.optics_channels = int(optics_channels)
        self.target_rate = target_rate
        self.rows: List[Dict] = []
        self.seq = 0

        # Power bands: alpha, beta, gamma, delta, theta (each with 4 electrodes)
        self.power_bands = ["alpha", "beta", "gamma", "delta", "theta"]
        self.power_band_channels = len(self.power_bands) * self.eeg_channels

    def _pad(self, values: List[float], target_len: int) -> List[float]:
        if len(values) >= target_len:
            return list(values[:target_len])
        return list(values) + [np.nan] * (target_len - len(values))

    def add_eeg(self, values: List[float]):
        ts = time.time()
        self.rows.append({
            "timestamp": ts,
            "eeg": self._pad([float(v) for v in values], self.eeg_channels),
            "optics": [np.nan] * self.optics_channels,
            "bands": [[np.nan] * self.eeg_channels for _ in self.power_bands],
        })
        self.seq += 1

    def add_optics(self, values: List[float]):
        ts = time.time()
        self.rows.append({
            "timestamp": ts,
            "eeg": [np.nan] * self.eeg_channels,
            "optics": self._pad([float(v) for v in values], self.optics_channels),
            "bands": [[np.nan] * self.eeg_channels for _ in self.power_bands],
        })
        self.seq += 1

    def add_band(self, band: str, values: List[float]):
        ts = time.time()
        band_idx = self.power_bands.index(band)
        self.rows.append({
            "timestamp": ts,
            "eeg": [np.nan] * self.eeg_channels,
            "optics": [np.nan] * self.optics_channels,
            "bands": [
                self._pad([float(v) for v in values], self.eeg_channels) if i == band_idx
                else [np.nan] * self.eeg_channels
                for i in range(len(self.power_bands))
            ],
        })
        self.seq += 1

    def resample_and_save(self):
        if not self.rows:
            print("No data collected, nothing to save.")
            return

        timestamps = np.array([row["timestamp"] for row in self.rows])
        eeg_data = np.array([row["eeg"] for row in self.rows])
        optics_data = np.array([row["optics"] for row in self.rows])
        band_data = np.array([np.concatenate(row["bands"]) for row in self.rows])

        start_time, end_time = timestamps.min(), timestamps.max()
        resampled_timestamps = np.arange(start_time, end_time, 1.0 / self.target_rate)

        eeg_resampled = np.zeros((len(resampled_timestamps), self.eeg_channels))
        optics_resampled = np.zeros((len(resampled_timestamps), self.optics_channels))
        bands_resampled = np.zeros((len(resampled_timestamps), self.power_band_channels))

        for ch in range(self.eeg_channels):
            mask = ~np.isnan(eeg_data[:, ch])
            eeg_resampled[:, ch] = np.interp(resampled_timestamps, timestamps[mask], eeg_data[mask, ch]) if mask.sum() > 1 else np.nan

        for ch in range(self.optics_channels):
            mask = ~np.isnan(optics_data[:, ch])
            optics_resampled[:, ch] = np.interp(resampled_timestamps, timestamps[mask], optics_data[mask, ch]) if mask.sum() > 1 else np.nan

        for ch in range(self.power_band_channels):
            mask = ~np.isnan(band_data[:, ch])
            bands_resampled[:, ch] = np.interp(resampled_timestamps, timestamps[mask], band_data[mask, ch]) if mask.sum() > 1 else np.nan

        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            header = (
                ["timestamp", "seq"]
                + [f"eeg_{i}" for i in range(self.eeg_channels)]
                + [f"optics_{i}" for i in range(self.optics_channels)]
                + [f"{band}_{i}" for band in self.power_bands for i in range(self.eeg_channels)]
            )
            writer.writerow(header)

            for i, ts in enumerate(resampled_timestamps):
                row = (
                    [ts, i]
                    + eeg_resampled[i].tolist()
                    + optics_resampled[i].tolist()
                    + bands_resampled[i].tolist()
                )
                writer.writerow(row)


class Handler:
    def __init__(self, recorder: CsvRecorder):
        self.recorder = recorder

    def handle_eeg(self, address: str, *args):
        self.recorder.add_eeg(list(args))

    def handle_optics(self, address: str, *args):
        self.recorder.add_optics(list(args))

    def handle_band(self, band: str):
        def _inner(address: str, *args):
            self.recorder.add_band(band, list(args))
        return _inner


def run_server(listen_ip: str, listen_port: int, handler: Handler):
    dispatcher = Dispatcher()
    dispatcher.map("/muse/eeg", handler.handle_eeg)
    dispatcher.map("/muse/optics", handler.handle_optics)

    # Add power bands
    for band in handler.recorder.power_bands:
        dispatcher.map(f"/muse/elements/{band}_absolute", handler.handle_band(band))

    server = ThreadingOSCUDPServer((listen_ip, listen_port), dispatcher)
    server.allow_reuse_address = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def parse_args():
    p = argparse.ArgumentParser(
        description="Records /muse/eeg, /muse/optics, and /muse/elements/*_absolute OSC streams into a synchronized CSV at 256 Hz."
    )
    p.add_argument("--listen-ip", type=str, default="0.0.0.0", help="IP to bind (default is local)")
    p.add_argument("--listen-port", type=int, default=8890, help="Port to bind (default: 8890)")
    p.add_argument("--csv", type=str, required=False, default="data.csv", help="Output path")
    p.add_argument("--max-seconds", type=float, default=5.0, help="Max recording duration in seconds, where 0 is unlimited")
    p.add_argument("--eeg-channels", type=int, default=4, help="Number of EEG channels (default: 4)")
    p.add_argument("--optics-channels", type=int, default=4, help="Number of optics channels (default: 16)")
    p.add_argument("--target-rate", type=float, default=256.0, help="Resampling rate in Hz (default: 256 Hz)")
    return p.parse_args()


def main():
    args = parse_args()
    killer = GracefulKiller()

    recorder = CsvRecorder(
        args.csv, eeg_channels=args.eeg_channels,
        optics_channels=args.optics_channels,
        target_rate=args.target_rate
    )
    handler = Handler(recorder)

    server, thread = run_server(args.listen_ip, args.listen_port, handler)
    print(
        f"Recording OSC on {args.listen_ip}:{args.listen_port} → {args.csv} "
        f"(max {args.max_seconds or '∞'} sec, EEG ch={args.eeg_channels}, Optics ch={args.optics_channels}, Target rate={args.target_rate} Hz)"
    )

    start = time.time()
    try:
        while not killer.kill_now:
            if args.max_seconds and (time.time() - start) >= args.max_seconds:
                print("Max duration reached.")
                break
            time.sleep(0.05)
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
        recorder.resample_and_save()
        print(f"Saved synchronized resampled data at {args.target_rate} Hz to {args.csv}")


if __name__ == "__main__":
    main()
