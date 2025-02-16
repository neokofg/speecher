import logging

import numpy as np
import queue
import threading
import time
from transformers import pipeline
from download_and_initialize_model import load_model, global_model
if global_model is None:
    global_model = load_model()
model = global_model["model"]
processor = global_model["processor"]
device = global_model["device"]
torch_dtype = global_model["torch_dtype"]
logging.basicConfig(level=logging.DEBUG)

if hasattr(model.config, "forced_decoder_ids"):
    model.config.forced_decoder_ids = None

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)
sample_rate = 16000
block_duration = 0.5
block_size = int(sample_rate * block_duration)
audio_q = queue.Queue()
recognized_q = queue.Queue()
def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    audio_q.put(indata.copy())


def process_audio():
    buffer = np.empty((0, 1), dtype=np.float32)
    prev_result = None
    silence_duration = 0

    while True:
        try:
            data = audio_q.get(timeout=1)

            # Проверяем переполнение очереди
            if audio_q.qsize() > 10:
                logging.warning("Audio queue is building up, clearing...")
                while not audio_q.empty():
                    audio_q.get()

            buffer = np.concatenate((buffer, data))

            if np.max(np.abs(data)) < 0.005:
                silence_duration += len(data) / sample_rate
                if silence_duration > 2.0:
                    buffer = np.empty((0, 1), dtype=np.float32)
                    silence_duration = 0
            else:
                silence_duration = 0

            if len(buffer) >= block_size:
                try:
                    audio_chunk = buffer[:block_size]
                    buffer = buffer[block_size:]
                    audio_input = audio_chunk.flatten()
                    result = pipe(audio_input, generate_kwargs={
                        "language": "ru",
                        "task": "transcribe"
                    })["text"].strip()

                    if result and result not in ["Продолжение следует...", "Спасибо."] and result != prev_result:
                        recognized_q.put(result)
                        prev_result = result
                except Exception as e:
                    logging.error(f"Error processing audio: {e}")

        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            continue
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()
if __name__ == "__main__":
    import sounddevice as sd
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
