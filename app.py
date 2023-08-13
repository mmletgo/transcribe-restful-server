from typing import BinaryIO

import numpy as np
import ffmpeg
from fastapi import FastAPI, File, UploadFile
from threading import Lock
from faster_whisper import WhisperModel
from starlette.responses import JSONResponse
import uvicorn
import io
import traceback

model_lock = Lock()
SAMPLE_RATE = 16000
model = WhisperModel("large-v2", device="cuda", compute_type="float16")

app = FastAPI()


@app.post("/asr", tags=["Endpoints"])
async def asr(file: UploadFile = File(None)):
    try:
        audio = await file.read()
        result = transcribe(load_audio(io.BytesIO(audio)))
        return JSONResponse(content={"transcription": result}, status_code=200)
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (ffmpeg.input(
                "pipe:", threads=0).output("-",
                                           format="s16le",
                                           acodec="pcm_s16le",
                                           ac=1,
                                           ar=sr).run(cmd="ffmpeg",
                                                      capture_stdout=True,
                                                      capture_stderr=True,
                                                      input=file.read()))
        except ffmpeg.Error as e:
            raise RuntimeError(
                f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def transcribe(audio):
    with model_lock:
        segments = []
        segment_generator, info = model.transcribe(audio,
                                                   beam_size=5,
                                                   language='en',
                                                   word_timestamps=True)
        for segment in segment_generator:
            segments.append(segment)

        segment_list = []
        for segment in segments:
            word_list = []
            for word in segment.words:
                word_list.append({
                    "start": word.start,
                    "end": word.end,
                    "text": word.word,
                })
            segment_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": word_list,
            })
        result = segment_list
        # print(segment_list)

        return result


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8800)
