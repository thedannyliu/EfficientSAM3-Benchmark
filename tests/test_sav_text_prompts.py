from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sam_backend.sav_text_prompts import apply_prompt_file, init_prompt_file


class SAVTextPromptsTest(unittest.TestCase):
    def test_init_and_apply_prompt_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest = tmp / "sav.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "sav_video_000",
                        "video_id": "video",
                        "object_id": "000",
                        "prompt_frame_index": 0,
                        "point": [10.0, 20.0],
                        "initial_mask_area_ratio": 0.1,
                        "initial_bbox_xyxy": [1, 2, 3, 4],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prompt_file = tmp / "prompts.json"
            payload = init_prompt_file(manifest, prompt_file, tmp / "review")
            self.assertEqual(payload["samples"][0]["text_prompt"], "")
            self.assertEqual(payload["samples"][0]["instance_hint"], "")
            self.assertEqual(payload["samples"][0]["review_overlay"], str(tmp / "review" / "sav_video_000.png"))

            payload["samples"][0]["text_prompt"] = "person"
            payload["samples"][0]["instance_hint"] = "standing person in the center"
            payload["samples"][0]["review_note"] = "selected visible person"
            prompt_file.write_text(json.dumps(payload), encoding="utf-8")
            output = tmp / "sav_text.jsonl"
            summary = apply_prompt_file(manifest, prompt_file, output)

            self.assertEqual(summary["rows"], 1)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["text_prompt"], "person")
            self.assertEqual(row["text_prompt_instance_hint"], "standing person in the center")
            self.assertEqual(row["text_prompt_review_note"], "selected visible person")


if __name__ == "__main__":
    unittest.main()
