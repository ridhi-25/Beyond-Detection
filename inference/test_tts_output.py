from pathlib import Path
from unittest import TestCase

from inference.tts_output import FileTTSEngine


def test_file_engine_synthesizes_queued_messages(tmp_path: Path) -> None:
    engine = FileTTSEngine(tmp_path)
    try:
        engine.speak("First queued validation alert")
        engine.speak("Second queued validation alert")
        engine.flush()

        assert len(engine.audio_files) == 2
        assert all(path.is_file() and path.stat().st_size > 44 for path in engine.audio_files)
        assert all(path.parent == tmp_path for path in engine.audio_files)
    finally:
        engine.close()


def test_file_engine_rejects_empty_messages(tmp_path: Path) -> None:
    engine = FileTTSEngine(tmp_path)
    try:
        with TestCase().assertRaisesRegex(ValueError, "non-empty"):
            engine.speak(" ")
    finally:
        engine.close()


def test_file_engine_cannot_enqueue_after_close(tmp_path: Path) -> None:
    engine = FileTTSEngine(tmp_path)
    engine.close()

    with TestCase().assertRaisesRegex(RuntimeError, "closed"):
        engine.speak("Too late")