import cv2
import sounddevice as sd
import numpy as np
import time
import threading
import queue
import os

# Reduce OpenCV verbosity
os.environ['OPENCV_LOG_LEVEL'] = 'WARN'

# Queues: Minimal for low latency/memory
video_queue = queue.Queue(maxsize=5)   # Frames + ts
audio_queue = queue.Queue(maxsize=20)  # Chunks + ts

# Sync window (tune: tighter=more accurate, wider=forgiving)
SYNC_WINDOW = 0.03  # 30ms

# Calibration offset (tune later via clap test)
AUDIO_OFFSET = 0.0  # e.g., -0.015 if audio lags video


def video_thread(duration=60):
    """Capture video frames with timestamps."""
    # Probe cameras
    cap = None
    for i in range(5):
        cap_test = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows backend
        if cap_test.isOpened():
            print(f"Using camera index {i}")
            cap = cap_test
            break
        cap_test.release()

    if cap is None:
        print("No camera found—check hardware/connections")
        return

    # Tune properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.perf_counter()
    frame_count = 0

    while time.perf_counter() - start < duration:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed—retrying...")
            time.sleep(0.1)
            continue

        ts = time.perf_counter()
        video_queue.put((ts, frame))

        frame_count += 1
        if video_queue.qsize() > 3:
            video_queue.get()

        time.sleep(1 / 15)  # Pace to FPS

    cap.release()
    fps = frame_count / duration
    print(f"Captured {frame_count} frames successfully (avg {fps:.1f} FPS)")


def audio_thread(duration=60):
    """Capture audio chunks with timestamps."""
    def callback(indata, frames, time_info, status):
        ts = time.perf_counter() + AUDIO_OFFSET
        chunk = indata[:, 0].copy()  # Mono float32
        audio_queue.put((ts, chunk))
        if audio_queue.qsize() > 10:
            audio_queue.get()

    stream = sd.InputStream(
        samplerate=22050,
        channels=1,
        blocksize=512,
        dtype='float32',
        latency='low',
        callback=callback
    )

    with stream:
        time.sleep(duration)


def sync_loop(duration=60):
    """Synchronize audio and video streams."""
    start = time.perf_counter()
    alignments = []

    while time.perf_counter() - start < duration:
        if video_queue.empty():
            time.sleep(0.01)
            continue

        v_ts, frame = video_queue.get()
        matches = 0
        temp_audio = []

        while not audio_queue.empty():
            a_ts, a_chunk = audio_queue.get()
            diff = abs(a_ts - v_ts)
            if diff <= SYNC_WINDOW:
                matches += 1
                alignments.append((v_ts, a_ts, diff))
            else:
                temp_audio.append((a_ts, a_chunk))

        for item in temp_audio:
            audio_queue.put(item)

        print(f"Frame at t={v_ts:.3f}s: {matches} audio matches (window={SYNC_WINDOW*1000:.0f}ms)")
        cv2.imshow('Sync Test (Wave/Clap)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if alignments:
        mean_diff = np.mean([d for _, _, d in alignments]) * 1000
        print(f"Sync Accuracy: Mean diff {mean_diff:.1f}ms over {len(alignments)} alignments")
    else:
        print("No alignments—check queues or window")


if __name__ == "__main__":
    v_thread = threading.Thread(target=video_thread, args=(60,), daemon=True)
    a_thread = threading.Thread(target=audio_thread, args=(60,), daemon=True)
    s_thread = threading.Thread(target=sync_loop, args=(60,), daemon=True)

    v_thread.start()
    a_thread.start()
    s_thread.start()

    v_thread.join()
    a_thread.join()
    s_thread.join()
