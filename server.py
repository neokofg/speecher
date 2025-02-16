import asyncio
import logging
import os
import time
import uuid

import numpy as np
from aiohttp import web
import aiohttp_cors
from baseline_pipeline import pipe, block_size
from scipy import signal
from initialize_tts import tts

from translation import TranslationModel

logging.basicConfig(level=logging.DEBUG)


class AudioProcessor:
    def __init__(self):
        self.buffer = np.empty((0, 1), dtype=np.float32)
        self.silence_duration = 0
        self.is_speaking = False
        self.last_sent_text = None
        self.min_speech_duration = 0.5
        self.silence_threshold = 0.01
        self.speech_started_at = None


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    processor = AudioProcessor()
    logging.info('WebSocket connection opened')

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.BINARY:
                try:
                    audio_chunk = np.frombuffer(msg.data, dtype=np.int16)
                    float_data = audio_chunk.astype(np.float32) / 32768.0
                    resampled_data = signal.resample(float_data, len(float_data) // 3)
                    current_amplitude = np.max(np.abs(resampled_data))

                    is_current_chunk_speech = current_amplitude > processor.silence_threshold

                    if is_current_chunk_speech:
                        if not processor.is_speaking:
                            processor.is_speaking = True
                            processor.speech_started_at = time.time()
                            logging.info("Speech started")
                        processor.silence_duration = 0
                    else:
                        processor.silence_duration += len(resampled_data) / 16000

                    processor.buffer = np.concatenate([
                        processor.buffer,
                        resampled_data.reshape(-1, 1)
                    ])

                    should_process = (
                        processor.is_speaking and
                        processor.silence_duration > 0.3 and
                        (time.time() - processor.speech_started_at) > processor.min_speech_duration
                    )

                    if should_process:
                        logging.info(f"Processing speech chunk of length {processor.buffer.shape[0]}")
                        chunk = processor.buffer.flatten()
                        processor.buffer = np.empty((0, 1), dtype=np.float32)
                        processor.is_speaking = False

                        try:
                            result = pipe(chunk, generate_kwargs={
                                "language": "ru",
                                "task": "transcribe"
                            })["text"].strip()

                            if result and result not in ["Продолжение следует...", "Спасибо."] and result != processor.last_sent_text:
                                logging.info(f"Sending result: {result}")
                                await ws.send_str(result)
                                processor.last_sent_text = result

                        except Exception as e:
                            logging.error(f"ASR Error: {e}")

                    elif processor.silence_duration > 1.0:
                        processor.buffer = np.empty((0, 1), dtype=np.float32)
                        processor.is_speaking = False

                except Exception as e:
                    logging.error(f"Error processing audio chunk: {e}")

    except Exception as e:
        logging.error(f"Error in websocket handler: {e}")
    finally:
        logging.info('WebSocket connection closed')

    return ws


async def tts_handler(request):
    try:
        data = await request.json()
        text = data.get("text")
        language = data.get("language", "russian").lower()

        if not text:
            return web.json_response({"error": "Нет текста"}, status=400)

        output_file = f"{uuid.uuid4()}_output.wav"

        if language in ["russian", "ru"]:
            tts.tts_to_file(
                text=text,
                file_path=output_file,
                speaker_wav="path/to/voice.wav",
                language="ru"
            )
        else:
            tts.tts_to_file(
                text=text,
                file_path=output_file,
                speaker_wav="path/to/voice.wav",
                language=language
            )

        if not os.path.exists(output_file):
            return web.json_response({"error": "Не удалось сгенерировать аудио"}, status=500)

        return web.FileResponse(output_file, headers={"Content-Type": "audio/wav"})

    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)


translator = TranslationModel()
async def translate_handler(request):
    try:
        data = await request.json()
        text = data.get("text")
        source_lang = data.get("source_lang", "Russian")
        target_lang = data.get("target_lang", "English")

        if not text:
            return web.json_response({"error": "Нет текста для перевода"}, status=400)

        # Запускаем перевод в отдельном потоке, чтобы не блокировать event loop
        translation_result = await asyncio.to_thread(translator.translate, text, source_lang, target_lang)

        return web.json_response({"translation": translation_result})
    except Exception as e:
        logging.error(f"Translation error: {e}")
        return web.json_response({"error": str(e)}, status=500)


app = web.Application()
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})

app.router.add_get('/ws', websocket_handler)
app.router.add_post('/api/tts', tts_handler)
app.router.add_post('/api/translate', translate_handler)

for route in list(app.router.routes()):
    cors.add(route)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8000)
