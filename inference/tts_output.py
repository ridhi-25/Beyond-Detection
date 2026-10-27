"""Asynchronous pyttsx3 speech output and WAV-file synthesis."""

from __future__ import annotations

import queue
import time
import uuid
from pathlib import Path
from threading import Event, Thread

QueueItem = tuple[str, int, str] | tuple[str, Event] | tuple[str]


class _QueuedTTSEngine:
    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        rate: int | None = None,
        volume: float | None = None,
    ) -> None:
        if volume is not None and not 0.0 <= volume <= 1.0:
            raise ValueError("volume must be between 0.0 and 1.0")
        self._output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
        self._rate = rate
        self._volume = volume
        self._run_id = uuid.uuid4().hex
        self._queue: queue.Queue[QueueItem] = queue.Queue()
        self._thread_error: BaseException | None = None
        self._closed = False
        self._next_index = 0
        self._audio_files: list[Path] = []
        self._worker = Thread(
            target=self._run,
            name="beyond-detection-tts",
            daemon=True,
        )
        self._worker.start()

    @property
    def audio_files(self) -> tuple[Path, ...]:
        return tuple(self._audio_files)

    def speak(self, message: str) -> float:
        """Queue speech and return its enqueue latency in seconds."""
        if self._closed:
            raise RuntimeError("TTS engine is closed")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        self._raise_worker_error()
        started = time.perf_counter()
        index = self._next_index
        self._next_index += 1
        self._queue.put(("job", index, message))
        return time.perf_counter() - started

    def flush(self) -> None:
        """Wait for queued work and raise if the background worker failed."""
        self._queue.join()
        self._raise_worker_error()

    def close(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return
        try:
            self.flush()
        finally:
            self._closed = True
            self._queue.put(("stop",))
            self._worker.join()
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._thread_error is not None:
            raise RuntimeError("Background TTS worker failed") from self._thread_error

    def _run(self) -> None:
        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            if self._rate is not None:
                engine.setProperty("rate", self._rate)
            if self._volume is not None:
                engine.setProperty("volume", self._volume)
        except BaseException as error:
            self._thread_error = error

        pending: list[tuple[int, str]] = []
        while True:
            item = self._queue.get()
            try:
                if item[0] == "job":
                    job = (item[1], item[2])
                    if self._output_dir is None:
                        self._synthesize(engine, [job])
                    else:
                        pending.append(job)
                    continue
                if item[0] == "flush":
                    self._synthesize(engine, pending)
                    pending.clear()
                    item[1].set()
                    continue
                if item[0] == "stop":
                    self._synthesize(engine, pending)
                    return
            except BaseException as error:
                if self._thread_error is None:
                    self._thread_error = error
                if item[0] == "flush":
                    item[1].set()
                if item[0] == "stop":
                    return
            finally:
                self._queue.task_done()

    def _synthesize(self, engine, pending: list[tuple[int, str]]) -> None:
        if not pending:
            return
        if engine is None or self._thread_error is not None:
            return
        if self._output_dir is not None:
            for index, message in pending:
                output_path = self._output_dir / f"alert_{self._run_id}_{index:04d}.wav"
                engine.save_to_file(message, str(output_path))
                self._audio_files.append(output_path)
            engine.runAndWait()
        else:
            for _, message in pending:
                engine.say(message)
                engine.runAndWait()



class TTSEngine(_QueuedTTSEngine):
    """Play speech locally using a dedicated background worker thread."""

    def __init__(self, rate: int | None = None, volume: float | None = None) -> None:
        super().__init__(rate=rate, volume=volume)


class FileTTSEngine(_QueuedTTSEngine):
    """Synthesize each queued message to a WAV file without playing audio."""

    def __init__(
        self,
        output_dir: str | Path,
        rate: int | None = None,
        volume: float | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir, rate=rate, volume=volume)

    def flush(self) -> None:
        if self._closed:
            self._raise_worker_error()
            return
        self._raise_worker_error()
        completed = Event()
        self._queue.put(("flush", completed))
        completed.wait()
        self._raise_worker_error()
        missing = [
            path
            for path in self.audio_files
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            raise OSError(f"TTS audio file(s) were not created: {missing}")