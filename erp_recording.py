from pythonosc import dispatcher
from pythonosc import osc_server
import argparse
import time
import csv
import threading
from pynput import keyboard
from datastreamer import DataStreamer, DataRecorder

parser = argparse.ArgumentParser()
parser.add_argument("--recording_time", type=float, default=300.0)
parser.add_argument("--ip", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8890)
parser.add_argument("--output", type=str, default="eeg_recording.csv")
args = parser.parse_args()

ip = args.ip
port = args.port

streamer = DataStreamer(window_size=512)
recorder = DataRecorder(record_time=args.recording_time)

space_pressed_flag = False
lock = threading.Lock()

def on_press(key):
    global space_pressed_flag
    try:
        if key == keyboard.Key.space:
            with lock:
                space_pressed_flag = True
    except AttributeError:
        pass

def on_release(key):
    pass

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def eeg_handler(addr, *args):
    global space_pressed_flag
    eeg_data = args[:4]

    with lock:
        pressed = 1 if space_pressed_flag else 0
        space_pressed_flag = False

    streamer.update(addr, *eeg_data)
    recorder.update(addr, *eeg_data, pressed)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/muse/eeg*", eeg_handler)

server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

server_thread = threading.Thread(target=server.serve_forever)
server_thread.daemon = True
server_thread.start()

print(f"Recording Started")
recorder.start_recording()

while recorder.is_recording_active():
    time.sleep(0.1)

print("Recording has finished")
recorded_data = recorder.get_recorded_data()

with open(args.output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['timestamp', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'space_pressed'])
    for timestamp, data in recorded_data:
        if len(data) == 5:
            eeg_data, pressed = data[:4], data[4]
        else:
            eeg_data, pressed = data[:4], 0
        writer.writerow([timestamp] + list(eeg_data) + [pressed])

print(f"Data saved to {args.output}")

server.shutdown()
listener.stop()
