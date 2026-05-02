"""
src/pipeline/tts_engine.py
───────────────────────────
Non-blocking TTS using Kokoro (82M params, offline).

speak() queues a string and returns immediately.
Audio renders and plays in a background thread so the pipeline
is never stalled waiting for speech to finish.
"""

import queue
import threading
import numpy as np

try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False
    print("[TTS] sounddevice not found — audio playback disabled.")

try:
    from kokoro import KPipeline
    _KOKORO_AVAILABLE = True
except ImportError:
    _KOKORO_AVAILABLE = False
    print("[TTS] kokoro not found — TTS disabled. Install with: pip install kokoro")


SAMPLE_RATE = 24_000    # Kokoro output sample rate
VOICE       = "af_heart"   # American Female, warm tone. Others: af_sky, am_adam, bf_emma


class TTSEngine:
    def __init__(self, voice: str = VOICE, speed: float = 1.15):
        """
        Parameters
        ----------
        voice : Kokoro voice ID (see kokoro docs for full list)
        speed : 1.0 = normal, 1.15 = slightly faster (better for warnings)
        """
        self._queue   = queue.Queue(maxsize=2)   # drop old messages if backed up
        self._voice   = voice
        self._speed   = speed
        self._pipeline = None

        if _KOKORO_AVAILABLE:
            self._pipeline = KPipeline(lang_code="a")   # 'a' = American English
            print(f"[TTS] Kokoro ready — voice={voice}  speed={speed}")

        t = threading.Thread(target=self._worker, daemon=True)
        t.start()

    def speak(self, text: str, priority: bool = False):
        """
        Queue a string for speech.

        priority=True clears the queue first (use for CRITICAL events
        so they aren't waiting behind an earlier INFO message).
        """
        if priority:
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass

        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass   # silently drop if the queue is already full

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker(self):
        while True:
            text = self._queue.get()
            try:
                self._synthesize_and_play(text)
            except Exception as e:
                print(f"[TTS] Error: {e}")

    def _synthesize_and_play(self, text: str):
        if not _KOKORO_AVAILABLE or self._pipeline is None:
            print(f"[TTS] (no engine) Would say: {text}")
            return

        # Kokoro returns a generator of (graphemes, phonemes, audio_chunk)
        audio_chunks = []
        for _, _, audio in self._pipeline(text, voice=self._voice, speed=self._speed):
            if audio is not None:
                audio_chunks.append(audio)

        if not audio_chunks:
            return

        full_audio = np.concatenate(audio_chunks).astype(np.float32)

        if _SD_AVAILABLE:
            sd.play(full_audio, samplerate=SAMPLE_RATE)
            sd.wait()
        else:
            print(f"[TTS] Would say: {text}")
