import asyncio
import logging
import os
import time

import numpy as np
from aiohttp import web
import aiohttp_cors
from baseline_pipeline import pipe, block_size
from scipy import signal
from initialize_tts import tts

logging.basicConfig(level=logging.DEBUG)


class AudioProcessor:
    def __init__(self):
        self.buffer = np.empty((0, 1), dtype=np.float32)
        self.silence_duration = 0
        self.is_speaking = False
        self.last_sent_text = None
        self.min_speech_duration = 0.5  # Минимальная длительность речи в секундах
        self.silence_threshold = 0.01  # Порог тишины
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
                    # Преобразуем входящие данные
                    audio_chunk = np.frombuffer(msg.data, dtype=np.int16)
                    float_data = audio_chunk.astype(np.float32) / 32768.0
                    resampled_data = signal.resample(float_data, len(float_data) // 3)
                    current_amplitude = np.max(np.abs(resampled_data))

                    # Определяем, говорит ли человек
                    is_current_chunk_speech = current_amplitude > processor.silence_threshold

                    if is_current_chunk_speech:
                        if not processor.is_speaking:
                            # Начало речи
                            processor.is_speaking = True
                            processor.speech_started_at = time.time()
                            logging.info("Speech started")
                        processor.silence_duration = 0
                    else:
                        # Тишина
                        processor.silence_duration += len(resampled_data) / 16000

                    # Добавляем данные в буфер в любом случае
                    processor.buffer = np.concatenate([
                        processor.buffer,
                        resampled_data.reshape(-1, 1)
                    ])

                    # Проверяем, нужно ли обработать накопленные данные
                    should_process = (
                            processor.is_speaking and  # Была речь
                            processor.silence_duration > 0.3 and  # Пауза более 0.3 секунды
                            (time.time() - processor.speech_started_at) > processor.min_speech_duration
                    # Минимальная длительность речи
                    )

                    if should_process:
                        logging.info(f"Processing speech chunk of length {processor.buffer.shape[0]}")
                        chunk = processor.buffer.flatten()
                        # Очищаем буфер
                        processor.buffer = np.empty((0, 1), dtype=np.float32)
                        processor.is_speaking = False

                        try:
                            result = pipe(chunk, generate_kwargs={
                                "language": "ru",
                                "task": "transcribe"
                            })["text"].strip()

                            if result and result not in ["Продолжение следует...",
                                                         "Спасибо."] and result != processor.last_sent_text:
                                logging.info(f"Sending result: {result}")
                                await ws.send_str(result)
                                processor.last_sent_text = result

                        except Exception as e:
                            logging.error(f"ASR Error: {e}")

                    # Если тишина слишком долгая, очищаем буфер
                    elif processor.silence_duration > 1.0:  # Более 1 секунды тишины
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

        output_file = "output.wav"  # Файл, куда будет записан результат

        # В данном примере для некоторых языков можно использовать ваш vocal_text,
        # а для остальных — напрямую TTS API. Здесь приведён пример для русского.
        if language == "russian" or language == "ru":
            # Здесь speaker_wav можно задавать динамически или из настроек
            tts.tts_to_file(
                text=text,
                file_path=output_file,
                speaker_wav="path/to/voice.wav",  # Замените на актуальный путь или параметр
                language="ru"
            )
        else:
            # Добавьте обработку для других языков по аналогии
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

app = web.Application()
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*"
    )
})

app.router.add_get('/ws', websocket_handler)
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0', port=8000)