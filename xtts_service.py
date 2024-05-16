import functools
import time
import logging
import requests
import subprocess
import sys
import os
import json
from typing import Iterator
logging.basicConfig(level = logging.INFO)

from tqdm import tqdm
from websockets.sync.server import serve


def get_speaker(ref_audio,server_url):
    files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
    response = requests.post(f"{server_url}/clone_speaker", files=files)
    return response.json()


def stream_ffplay(audio_stream, output_file, save=True):
    if not save:
        ffplay_cmd = ["ffplay", "-nodisp", "-probesize", "1024", "-autoexit", "-"]
    else:
        print("Saving to ", output_file)
        ffplay_cmd = ["ffmpeg", "-probesize", "1024", "-i", "-", output_file]

    ffplay_proc = subprocess.Popen(ffplay_cmd, stdin=subprocess.PIPE)
    for chunk in audio_stream:
        if chunk is not None:
            ffplay_proc.stdin.write(chunk)

    # close on finish
    ffplay_proc.stdin.close()
    ffplay_proc.wait()


def tts(text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
    start = time.perf_counter()
    speaker["text"] = text
    speaker["language"] = language
    speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
    res = requests.post(
        f"{server_url}/tts_stream",
        json=speaker,
        stream=True,
    )
    end = time.perf_counter()
    print(f"Time to make POST: {end-start}s", file=sys.stderr)

    if res.status_code != 200:
        print("Error:", res.text)
        sys.exit(1)

    first = True
    for chunk in res.iter_content(chunk_size=512):
        if first:
            end = time.perf_counter()
            print(f"Time to first chunk: {end-start}s", file=sys.stderr)
            first = False
        if chunk:
            yield chunk

    print("⏱️ response.elapsed:", res.elapsed)


class SpeechXTTS:
    def __init__(self):
        pass
    
    def run(self, host, port, audio_queue=None, should_send_server_ready=None):
        logging.info("\n[WhisperSpeech INFO:] Launching XTTSv2 model ...\n")
        
        with open("./default_speaker.json", "r") as file:
            self.speaker = json.load(file)
        if os.environ["ref_file"] != "":
            self.speaker = get_speaker(os.environ["ref_file"], os.environ["xtts_url"])
        self.server_url = os.environ["xtts_url"]
        
        logging.info("[WhisperSpeech INFO:] Launched XTTSv2 model. Connect to the WebGUI now.")
        should_send_server_ready.value = True

        with serve(
            functools.partial(self.start_whisperspeech_tts, audio_queue=audio_queue), 
            host, port
            ) as server:
            server.serve_forever()

    def start_whisperspeech_tts(self, websocket, audio_queue=None):
        self.eos = False
        self.output_audio = None

        while True:
            llm_response = audio_queue.get()
            if audio_queue.qsize() != 0:
                continue

            # check if this websocket exists
            try:
                websocket.ping()
            except Exception as e:
                del websocket
                audio_queue.put(llm_response)
                break
            
            llm_output = llm_response["llm_output"][0]
            self.eos = llm_response["eos"]

            def should_abort():
                if not audio_queue.empty(): raise TimeoutError()

            # only process if the output updated
            if self.last_llm_response != llm_output.strip():
                try:
                    start = time.time()
                    
                    # audio = self.pipe.generate(llm_output.strip(), step_callback=should_abort)
                    audio = tts(llm_output.strip(), self.speaker, "en", self.server_url, 20)
                    
                    inference_time = time.time() - start
                    logging.info(f"[WhisperSpeech INFO:] TTS inference done in {inference_time} ms.\n\n")
                    self.output_audio = audio
                    self.last_llm_response = llm_output.strip()
                except TimeoutError:
                    pass

            if self.eos and self.output_audio is not None:
                try:
                    websocket.send(self.output_audio)
                except Exception as e:
                    logging.error(f"[WhisperSpeech ERROR:] Audio error: {e}")

