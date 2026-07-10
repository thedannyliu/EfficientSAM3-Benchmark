from __future__ import annotations

import cv2
import numpy as np


class TextPromptEditor:
    def __init__(self) -> None:
        self.active = False
        self.text = ""

    def start(self, initial_text: str = "") -> None:
        self.active = True
        self.text = initial_text

    def handle_key(self, key: int) -> str | None:
        if not self.active:
            return None
        if key in {10, 13}:
            prompt = self.text.strip()
            self.active = False
            return prompt or None
        if key == 27:
            self.active = False
            return None
        if key in {8, 127}:
            self.text = self.text[:-1]
        elif 32 <= key <= 126:
            self.text += chr(key)
        return None

    def draw(self, image_bgr: np.ndarray) -> None:
        if not self.active:
            return
        height, width = image_bgr.shape[:2]
        bar_height = min(54, height)
        top = height - bar_height
        cv2.rectangle(image_bgr, (0, top), (width - 1, height - 1), (24, 24, 24), -1)
        cv2.rectangle(image_bgr, (0, top), (width - 1, height - 1), (120, 120, 120), 1)
        visible = self._visible_suffix(width - 32)
        cv2.putText(
            image_bgr,
            f"> {visible}_",
            (16, height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _visible_suffix(self, max_width: int) -> str:
        visible = self.text
        while (
            visible
            and cv2.getTextSize(f"> {visible}_", cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0] > max_width
        ):
            visible = visible[1:]
        return visible
