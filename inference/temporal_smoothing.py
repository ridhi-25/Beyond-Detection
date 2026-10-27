"""Stateful temporal smoothing for Sprint 10 alert decisions."""

from collections import deque


class AlertStabilizer:
    def __init__(
        self,
        window_size: int = 5,
        min_consistent_frames: int = 3,
        cooldown_seconds: float = 2.0,
    ) -> None:
        if (
            not isinstance(window_size, int)
            or isinstance(window_size, bool)
            or window_size < 1
        ):
            raise ValueError("window_size must be an integer >= 1")
        if (
            not isinstance(min_consistent_frames, int)
            or isinstance(min_consistent_frames, bool)
            or min_consistent_frames < 1
        ):
            raise ValueError("min_consistent_frames must be an integer >= 1")
        if min_consistent_frames > window_size:
            raise ValueError("min_consistent_frames must not exceed window_size")
        if (
            not isinstance(cooldown_seconds, (int, float))
            or isinstance(cooldown_seconds, bool)
            or not cooldown_seconds >= 0
        ):
            raise ValueError("cooldown_seconds must be a number >= 0")

        self.window_size = window_size
        self.min_consistent_frames = min_consistent_frames
        self.cooldown_seconds = cooldown_seconds
        self.history = deque(maxlen=window_size)
        self.last_spoken_message: str | None = None
        self.last_spoken_time: float | None = None

    def update(self, raw_alert: dict | None, current_time: float) -> dict | None:
        self.history.append(raw_alert)
        candidate = self.history[-1]

        if candidate is None:
            return None

        count = sum(
            1
            for alert in self.history
            if alert is not None
            and alert["class_name"] == candidate["class_name"]
            and alert["direction"] == candidate["direction"]
        )
        if count < self.min_consistent_frames:
            return None

        same_as_last = self.last_spoken_message == candidate["message"]
        if (
            same_as_last
            and self.last_spoken_time is not None
            and current_time - self.last_spoken_time < self.cooldown_seconds
        ):
            return None

        self.last_spoken_message = candidate["message"]
        self.last_spoken_time = current_time
        return candidate
