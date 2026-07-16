from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import shutil
import subprocess
import tempfile
from collections import deque
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from .backends import _import_required, _prepend_repo_path
from .sam2_stage1 import patch_stage1_forward_image


MAX_OBJECT_COUNT = 255
OBJECT_COLORS = (
    (30, 220, 80),
    (255, 80, 30),
    (40, 150, 255),
    (230, 210, 40),
    (210, 70, 230),
    (50, 220, 220),
)


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_kind: str
    checkpoint_path: str
    sam2_checkpoint_path: str = ""
    student_family: str = "auto"
    student_model_name: str = ""
    student_backbone_checkpoint: str = ""
    student_adapter_mode: str = "auto"


@dataclass(frozen=True)
class ModelArtifacts:
    result: dict[str, Any]
    mask_dir: Path
    latencies_ms: list[float]
    last_overlay: np.ndarray


@dataclass
class SaveControl:
    armed: bool = False


@dataclass(frozen=True)
class SaveDecision:
    timing_mode: str | None
    speed: float


class FFmpegVideoWriter:
    def __init__(
        self,
        output_path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        source_path: Path | None,
        codec: str,
        preset: str,
        crf: int,
    ) -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required for the video demo")
        self.output_path = output_path
        self.width = width
        self.height = height
        self.frames = 0
        self.closed = False
        self.log_path = output_path.with_suffix(output_path.suffix + ".ffmpeg.log")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            _format_fps(fps),
            "-i",
            "pipe:0",
        ]
        audio_stream_index = (
            _ffprobe_audio_stream_index(source_path)
            if source_path is not None
            else None
        )
        source_has_audio = audio_stream_index is not None
        if source_has_audio:
            command.extend(
                [
                    "-i",
                    str(source_path),
                    "-map",
                    "0:v:0",
                    "-map",
                    f"1:{audio_stream_index}?",
                ]
            )
        command.extend(
            ffmpeg_video_args(
                codec,
                preset=preset,
                crf=crf,
                width=width,
                height=height,
                fps=fps,
            )
        )
        if source_has_audio:
            command.extend(
                ["-af", "apad", "-c:a", "aac", "-b:a", "192k", "-shortest"]
            )
        else:
            command.append("-an")
        command.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)])
        self.log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_handle,
        )

    def write(self, frame_bgr: np.ndarray, repeat_count: int = 1) -> None:
        if frame_bgr.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"video frame has shape {frame_bgr.shape[:2]}, expected "
                f"{(self.height, self.width)}"
            )
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        payload = np.ascontiguousarray(frame_bgr).tobytes()
        for _ in range(max(0, repeat_count)):
            try:
                self.process.stdin.write(payload)
            except BrokenPipeError as exc:
                raise RuntimeError(
                    f"ffmpeg stopped accepting frames for {self.output_path} "
                    f"after {self.frames} frames; see {self.log_path}"
                    f"{self._failure_details()}"
                ) from exc
            self.frames += 1

    def _failure_details(self) -> str:
        self.log_handle.flush()
        details = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
        return f"\nFFmpeg output:\n{details[-4000:]}" if details else ""

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        return_code = self.process.wait()
        self.log_handle.close()
        if return_code != 0:
            details = self.log_path.read_text(encoding="utf-8", errors="replace").strip()
            detail_suffix = f"\nFFmpeg output:\n{details[-4000:]}" if details else ""
            raise RuntimeError(
                f"ffmpeg failed for {self.output_path}; see {self.log_path}"
                f"{detail_suffix}"
            )
        self.log_path.unlink(missing_ok=True)


class PromptSelector:
    def __init__(
        self,
        frame_bgr: np.ndarray,
        display_max_width: int,
        display_max_height: int,
    ) -> None:
        self.frame_bgr = frame_bgr
        self.prompts: list[dict[str, Any]] = []
        self.drag_start: tuple[float, float] | None = None
        self.display_scale = display_scale(
            frame_bgr.shape[:2],
            display_max_width,
            display_max_height,
        )
        self.window_name = "SAM2 video demo: select objects"

    def run(self) -> list[dict[str, Any]]:
        print("Select one or more objects: click for a point or drag for a box.")
        print("Enter/Space: start  U: undo  R: reset  Q/Esc: cancel")
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                display = self._render()
                cv2.imshow(self.window_name, display)
                key = cv2.waitKey(20) & 0xFF
                if key in {27, ord("q")}:
                    raise KeyboardInterrupt
                if key == ord("u") and self.prompts:
                    removed = self.prompts.pop()
                    print(f"Removed object {len(self.prompts) + 1}: {removed['prompt_mode']}")
                elif key == ord("r"):
                    self.prompts.clear()
                    print("Cleared selected objects")
                elif key in {10, 13, 32}:
                    if self.prompts:
                        return [dict(prompt) for prompt in self.prompts]
                    print("Select at least one object")
        finally:
            cv2.destroyWindow(self.window_name)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if len(self.prompts) >= MAX_OBJECT_COUNT:
            self.drag_start = None
            return
        point = self._image_point(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = point
            return
        if event != cv2.EVENT_LBUTTONUP or self.drag_start is None:
            return
        start = self.drag_start
        self.drag_start = None
        x1, x2 = sorted((start[0], point[0]))
        y1, y2 = sorted((start[1], point[1]))
        if x2 - x1 >= 8.0 and y2 - y1 >= 8.0:
            prompt = {"prompt_mode": "box", "box": [x1, y1, x2, y2]}
        else:
            prompt = {"prompt_mode": "point", "point": [point[0], point[1]], "label": 1}
        self.prompts.append(prompt)
        print(f"Selected object {len(self.prompts)}: {prompt['prompt_mode']}")

    def _image_point(self, x: int, y: int) -> tuple[float, float]:
        height, width = self.frame_bgr.shape[:2]
        image_x = max(0.0, min(float(width - 1), x / self.display_scale))
        image_y = max(0.0, min(float(height - 1), y / self.display_scale))
        return image_x, image_y

    def _render(self) -> np.ndarray:
        overlay = self.frame_bgr.copy()
        for object_index, prompt in enumerate(self.prompts):
            color = OBJECT_COLORS[object_index % len(OBJECT_COLORS)]
            if prompt["prompt_mode"] == "box":
                x1, y1, x2, y2 = [int(round(value)) for value in prompt["box"]]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4, cv2.LINE_AA)
                label_point = (x1, max(24, y1 - 8))
            else:
                x, y = [int(round(value)) for value in prompt["point"]]
                cv2.circle(overlay, (x, y), 12, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.circle(overlay, (x, y), 7, color, -1, cv2.LINE_AA)
                label_point = (x + 14, max(24, y - 10))
            _draw_label(overlay, f"ID {object_index + 1}", label_point, color)
        _draw_status(overlay, f"Objects {len(self.prompts)} | Enter/Space: start")
        if self.display_scale == 1.0:
            return overlay
        size = (
            max(1, int(round(overlay.shape[1] * self.display_scale))),
            max(1, int(round(overlay.shape[0] * self.display_scale))),
        )
        return cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Track selected first-frame objects with SAM2.1-L, TinyViT-21M, "
            "and TinyViT-5M."
        )
    )
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--prompts-json", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--timing-mode",
        choices=("both", "source_fps", "realtime"),
        default="both",
    )
    parser.add_argument("--external-repo", default="external/sam2")
    parser.add_argument(
        "--sam2-checkpoint",
        default="checkpoints/sam2/sam2.1_hiera_large.pt",
    )
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--sam2-distill-root",
        default="external/SAM2-Distillation-Pipeline",
    )
    parser.add_argument(
        "--tv21-stage1-checkpoint",
        default="checkpoints/sam2_distill/stage1/tv21m_mse_cos.pt",
    )
    parser.add_argument(
        "--tv21-backbone-checkpoint",
        default=(
            "checkpoints/sam2_distill/tinyvit/"
            "tiny_vit_21m_512.dist_in22k_ft_in1k.safetensors"
        ),
    )
    parser.add_argument(
        "--tv5-stage1-checkpoint",
        "--tinyvit-stage1-checkpoint",
        dest="tv5_stage1_checkpoint",
        default="checkpoints/sam2_distill/stage1/tv5_proj_sam21l_msehr_cos025_best.pt",
    )
    parser.add_argument(
        "--tv5-backbone-checkpoint",
        "--tinyvit-backbone-checkpoint",
        dest="tv5_backbone_checkpoint",
        default=(
            "checkpoints/sam2_distill/tinyvit/"
            "tiny_vit_5m_224.dist_in22k_ft_in1k.safetensors"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("sam2p1_l", "tv21m_mse_cos", "tv5m_projection"),
        default=("sam2p1_l", "tv21m_mse_cos", "tv5m_projection"),
        help="models to run in the requested order",
    )
    parser.add_argument("--display-max-width", type=int, default=1600)
    parser.add_argument("--display-max-height", type=int, default=900)
    parser.add_argument("--inference-max-side", type=int, default=1024)
    parser.add_argument("--codec", default="libx264")
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--save-speed", type=float, default=1.0)
    parser.add_argument("--sam2-save-speed", type=float)
    parser.add_argument("--tv21-save-speed", type=float)
    parser.add_argument("--tv5-save-speed", type=float)
    return parser


def model_specs_from_args(args: argparse.Namespace) -> tuple[ModelSpec, ...]:
    specs = {
        "sam2p1_l": ModelSpec(
            model_id="sam2p1_l",
            model_kind="sam2",
            checkpoint_path=args.sam2_checkpoint,
        ),
        "tv21m_mse_cos": ModelSpec(
            model_id="tv21m_mse_cos",
            model_kind="stage1-student",
            checkpoint_path=args.tv21_stage1_checkpoint,
            sam2_checkpoint_path=args.sam2_checkpoint,
            student_family="tinyvit",
            student_model_name="tiny_vit_21m_512.dist_in22k_ft_in1k",
            student_backbone_checkpoint=args.tv21_backbone_checkpoint,
            student_adapter_mode="auto",
        ),
        "tv5m_projection": ModelSpec(
            model_id="tv5m_projection",
            model_kind="stage1-student",
            checkpoint_path=args.tv5_stage1_checkpoint,
            sam2_checkpoint_path=args.sam2_checkpoint,
            student_family="tinyvit",
            student_model_name="tiny_vit_5m_224.dist_in22k_ft_in1k",
            student_backbone_checkpoint=args.tv5_backbone_checkpoint,
            student_adapter_mode="projection",
        ),
    }
    return tuple(specs[model_id] for model_id in args.models)


def model_save_speed_overrides(args: argparse.Namespace) -> dict[str, float]:
    return {
        model_id: speed
        for model_id, speed in (
            ("sam2p1_l", args.sam2_save_speed),
            ("tv21m_mse_cos", args.tv21_save_speed),
            ("tv5m_projection", args.tv5_save_speed),
        )
        if speed is not None
    }


def resolve_model_save_speeds(
    model_specs: tuple[ModelSpec, ...],
    *,
    timing_mode: str,
    default_speed: float,
    overrides: dict[str, float],
) -> dict[str, float]:
    if timing_mode == "source_fps":
        return {model_spec.model_id: 1.0 for model_spec in model_specs}
    return {
        model_spec.model_id: overrides.get(model_spec.model_id, default_speed)
        for model_spec in model_specs
    }


def load_prompts(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"prompts JSON does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not 1 <= len(payload) <= MAX_OBJECT_COUNT:
        raise ValueError(
            f"prompts JSON must contain between 1 and {MAX_OBJECT_COUNT} prompts"
        )
    prompts: list[dict[str, Any]] = []
    for object_index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"prompt {object_index} must be an object")
        prompt_mode = item.get("prompt_mode")
        coordinate_key = "box" if prompt_mode == "box" else "point"
        expected_coordinates = 4 if prompt_mode == "box" else 2
        if prompt_mode not in {"point", "box"}:
            raise ValueError(f"prompt {object_index} has invalid prompt_mode")
        coordinates = item.get(coordinate_key)
        if not isinstance(coordinates, list) or len(coordinates) != expected_coordinates:
            raise ValueError(
                f"prompt {object_index} must contain {expected_coordinates} "
                f"{coordinate_key} coordinates"
            )
        normalized_coordinates = [float(value) for value in coordinates]
        if not all(math.isfinite(value) for value in normalized_coordinates):
            raise ValueError(f"prompt {object_index} coordinates must be finite")
        if prompt_mode == "box":
            x1, y1, x2, y2 = normalized_coordinates
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"prompt {object_index} box must have positive area")
            prompts.append({"prompt_mode": "box", "box": normalized_coordinates})
        else:
            prompts.append(
                {
                    "prompt_mode": "point",
                    "point": normalized_coordinates,
                    "label": int(item.get("label", 1)),
                }
            )
    return prompts


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video does not exist: {video_path}")
    if args.save_speed <= 0.0:
        raise ValueError("save-speed must be positive")
    model_specs = model_specs_from_args(args)
    save_speed_overrides = model_save_speed_overrides(args)
    for model_id, speed in save_speed_overrides.items():
        if speed <= 0.0:
            raise ValueError(f"save speed for {model_id} must be positive")
    for model_spec in model_specs:
        if model_spec.model_kind == "sam2":
            _require_file(Path(model_spec.checkpoint_path), "SAM2.1-L checkpoint")
            continue
        _require_file(
            Path(model_spec.sam2_checkpoint_path),
            f"{model_spec.model_id} SAM2.1-L base checkpoint",
        )
        _require_file(
            Path(model_spec.checkpoint_path),
            f"{model_spec.model_id} Stage1 checkpoint",
        )
        _require_file(
            Path(model_spec.student_backbone_checkpoint),
            f"{model_spec.model_id} backbone checkpoint",
        )
    if not Path(args.external_repo).is_dir():
        raise FileNotFoundError(f"SAM2 source does not exist: {args.external_repo}")
    if (
        any(model_spec.model_kind == "stage1-student" for model_spec in model_specs)
        and not Path(args.sam2_distill_root).is_dir()
    ):
        raise FileNotFoundError(
            f"SAM2-Distillation-Pipeline does not exist: {args.sam2_distill_root}"
        )

    video_info, first_frame = probe_video(video_path)
    output_dir = _resolve_output_dir(args.output_dir, video_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompts_json:
        prompts = load_prompts(Path(args.prompts_json))
        print(f"Loaded {len(prompts)} prompts from {args.prompts_json}")
    else:
        prompts = PromptSelector(
            first_frame,
            args.display_max_width,
            args.display_max_height,
        ).run()
    (output_dir / "prompts.json").write_text(
        json.dumps(prompts, indent=2) + "\n",
        encoding="utf-8",
    )

    summary: dict[str, Any] = {
        "video_path": str(video_path),
        "video": asdict(video_info),
        "prompts": prompts,
        "object_count": len(prompts),
        "timing_mode": args.timing_mode,
        "models": [],
    }
    with tempfile.TemporaryDirectory(prefix="sam2_video_demo_", dir=output_dir) as work_dir:
        frame_dir = Path(work_dir) / "inference_frames"
        frame_count, inference_size = materialize_inference_frames(
            video_path,
            frame_dir,
            max_side=args.inference_max_side,
        )
        if frame_count != video_info.frame_count:
            video_info = VideoInfo(
                width=video_info.width,
                height=video_info.height,
                fps=video_info.fps,
                frame_count=frame_count,
            )
            summary["video"] = asdict(video_info)
        artifacts: list[ModelArtifacts] = []
        save_control = SaveControl()
        for model_spec in model_specs:
            model_artifacts = run_model(
                model_spec,
                video_path=video_path,
                video_info=video_info,
                frame_dir=frame_dir,
                inference_size=inference_size,
                prompts=prompts,
                mask_dir=Path(work_dir) / "masks" / model_spec.model_id,
                external_repo=args.external_repo,
                sam2_distill_root=args.sam2_distill_root,
                model_config=args.model_config,
                device=args.device,
                display_max_width=args.display_max_width,
                display_max_height=args.display_max_height,
                save_control=save_control,
            )
            artifacts.append(model_artifacts)
            summary["models"].append(model_artifacts.result)
            (output_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
        if save_control.armed:
            decision = SaveDecision(args.timing_mode, args.save_speed)
            if save_speed_overrides:
                configured_speeds = ", ".join(
                    f"{model_id}={speed:g}x"
                    for model_id, speed in save_speed_overrides.items()
                )
                print(
                    "Automatic save was armed during tracking: "
                    f"mode={decision.timing_mode} speeds={configured_speeds}"
                )
            else:
                print(
                    "Automatic save was armed during tracking: "
                    f"mode={decision.timing_mode} speed={decision.speed:g}x"
                )
        else:
            decision = choose_video_save(
                [artifact.last_overlay for artifact in artifacts],
                args.display_max_width,
                args.display_max_height,
                default_timing_mode=args.timing_mode,
                default_speed=args.save_speed,
                speed_overrides=save_speed_overrides,
            )
        if decision.timing_mode == "source_fps":
            decision = SaveDecision("source_fps", 1.0)
        summary["video_save_requested"] = decision.timing_mode is not None
        summary["videos_saved"] = False
        summary["save_timing_mode"] = decision.timing_mode
        summary["save_speed"] = decision.speed
        model_save_speeds = (
            resolve_model_save_speeds(
                model_specs,
                timing_mode=decision.timing_mode,
                default_speed=decision.speed,
                overrides=save_speed_overrides,
            )
            if decision.timing_mode is not None
            else {}
        )
        summary["model_save_speeds"] = model_save_speeds
        summary["save_output_fps"] = (
            video_info.fps
            if decision.timing_mode is not None
            else None
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        if decision.timing_mode is not None:
            for artifact in artifacts:
                artifact.result["output_paths"] = save_model_outputs(
                    artifact,
                    video_path=video_path,
                    video_info=video_info,
                    output_dir=output_dir,
                    timing_mode=decision.timing_mode,
                    speed=model_save_speeds[artifact.result["model_id"]],
                    codec=args.codec,
                    preset=args.preset,
                    crf=args.crf,
                )
            summary["videos_saved"] = True
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
    cv2.destroyAllWindows()
    print(f"Demo complete: {output_dir}")


def probe_video(video_path: Path) -> tuple[VideoInfo, np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    ok, first_frame = capture.read()
    if not ok:
        capture.release()
        raise RuntimeError(f"failed to decode first frame: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0))
    capture.release()
    probed_fps = _ffprobe_fps(video_path)
    if probed_fps > 0.0:
        fps = probed_fps
    if fps <= 0.0:
        raise RuntimeError(f"video source has no valid frame rate: {video_path}")
    height, width = first_frame.shape[:2]
    return VideoInfo(width, height, fps, frame_count), first_frame


def materialize_inference_frames(
    video_path: Path,
    frame_dir: Path,
    *,
    max_side: int,
) -> tuple[int, tuple[int, int]]:
    if max_side <= 0:
        raise ValueError("inference-max-side must be positive")
    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    frame_index = 0
    inference_size: tuple[int, int] | None = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        scale = min(1.0, max_side / float(max(height, width)))
        target_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        if target_size != (width, height):
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
        if inference_size is None:
            inference_size = target_size
        elif inference_size != target_size:
            raise RuntimeError("video frame dimensions changed during decoding")
        output_path = frame_dir / f"{frame_index:06d}.jpg"
        if not cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write inference frame: {output_path}")
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"Prepared {frame_index} inference frames")
    capture.release()
    if frame_index == 0 or inference_size is None:
        raise RuntimeError(f"video contains no decodable frames: {video_path}")
    print(
        f"Prepared {frame_index} frames at {inference_size[0]}x{inference_size[1]} "
        "for SAM2"
    )
    return frame_index, inference_size


def run_model(
    model_spec: ModelSpec,
    *,
    video_path: Path,
    video_info: VideoInfo,
    frame_dir: Path,
    inference_size: tuple[int, int],
    prompts: list[dict[str, Any]],
    mask_dir: Path,
    external_repo: str,
    sam2_distill_root: str,
    model_config: str,
    device: str,
    display_max_width: int,
    display_max_height: int,
    save_control: SaveControl,
) -> ModelArtifacts:
    print(f"Loading {model_spec.model_id}")
    predictor, torch_module, load_summary = build_predictor(
        model_spec,
        external_repo=external_repo,
        sam2_distill_root=sam2_distill_root,
        model_config=model_config,
        device=device,
    )
    inference_state: dict[str, Any] | None = None
    source_capture = cv2.VideoCapture(str(video_path))
    if not source_capture.isOpened():
        raise RuntimeError(f"failed to reopen video: {video_path}")
    mask_dir.mkdir(parents=True, exist_ok=True)
    latencies_ms: list[float] = []
    completion_times: deque[float] = deque(maxlen=60)
    first_completion_time: float | None = None
    last_completion_time: float | None = None
    last_overlay: np.ndarray | None = None
    try:
        inference_state = _init_state(predictor, frame_dir)
        scaled_prompts = scale_prompts(
            prompts,
            source_size=(video_info.width, video_info.height),
            inference_size=inference_size,
        )
        _sync(torch_module)
        prompt_start = perf_counter()
        for object_id, prompt in enumerate(scaled_prompts, start=1):
            _add_prompt(predictor, inference_state, object_id, prompt)
        _sync(torch_module)
        prompt_ms = (perf_counter() - prompt_start) * 1000.0
        source_frame_index = -1
        source_frame: np.ndarray | None = None
        iterator = predictor.propagate_in_video(
            inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=video_info.frame_count,
        )
        while True:
            _sync(torch_module)
            started = perf_counter()
            try:
                frame_index, object_ids, mask_logits = next(iterator)
                _sync(torch_module)
            except StopIteration:
                break
            latency_ms = (perf_counter() - started) * 1000.0
            frame_index = int(frame_index)
            while source_frame_index < frame_index:
                ok, decoded = source_capture.read()
                if not ok:
                    raise RuntimeError(
                        f"source video ended before SAM2 frame {frame_index}"
                    )
                source_frame = decoded
                source_frame_index += 1
            if source_frame is None:
                raise RuntimeError("source frame is unavailable")
            masks = resize_masks(
                mask_logits,
                target_size=(video_info.width, video_info.height),
            )
            object_id_values = [int(object_id) for object_id in object_ids]
            label_map = masks_to_label_map(masks, object_id_values)
            mask_path = mask_dir / f"{frame_index:06d}.png"
            if not cv2.imwrite(
                str(mask_path),
                label_map,
                [cv2.IMWRITE_PNG_COMPRESSION, 3],
            ):
                raise RuntimeError(f"failed to write temporary mask: {mask_path}")
            completed_at = perf_counter()
            completion_times.append(completed_at)
            if first_completion_time is None:
                first_completion_time = completed_at
            last_completion_time = completed_at
            overlay = overlay_masks(
                source_frame,
                masks,
                object_id_values,
                model_id=model_spec.model_id,
                frame_index=frame_index,
                latency_ms=latency_ms,
                display_fps=rolling_fps(completion_times),
            )
            last_overlay = overlay
            latencies_ms.append(latency_ms)
            _show_tracking_frame(
                overlay,
                display_max_width,
                display_max_height,
                save_control,
            )
            if frame_index % 30 == 0 or frame_index + 1 == video_info.frame_count:
                print(
                    f"{model_spec.model_id}: {frame_index + 1}/{video_info.frame_count} "
                    f"frames, {latency_ms:.1f} ms"
                )
    finally:
        source_capture.release()
        if inference_state is not None:
            try:
                predictor.reset_state(inference_state)
            except (AttributeError, RuntimeError):
                pass
        del inference_state
        del predictor
        gc.collect()
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

    if not latencies_ms or last_overlay is None:
        raise RuntimeError(f"{model_spec.model_id} produced no tracking frames")
    latency_array = np.asarray(latencies_ms, dtype=np.float64)
    mean_display_fps = None
    if (
        len(latencies_ms) > 1
        and first_completion_time is not None
        and last_completion_time is not None
        and last_completion_time > first_completion_time
    ):
        mean_display_fps = (
            (len(latencies_ms) - 1) / (last_completion_time - first_completion_time)
        )
    result = {
        "model_id": model_spec.model_id,
        "model_kind": model_spec.model_kind,
        "frames": int(latency_array.size),
        "object_count": len(prompts),
        "prompt_ms": prompt_ms,
        "mean_latency_ms": float(latency_array.mean()),
        "p50_latency_ms": float(np.percentile(latency_array, 50)),
        "p95_latency_ms": float(np.percentile(latency_array, 95)),
        "mean_display_fps": mean_display_fps,
        "output_paths": {},
        **load_summary,
    }
    print(
        f"{model_spec.model_id}: average tracking latency "
        f"{result['mean_latency_ms']:.1f} ms over {result['frames']} frames"
    )
    if mean_display_fps is not None:
        print(
            f"{model_spec.model_id}: average interactive display rate "
            f"{mean_display_fps:.1f} FPS"
        )
    print(json.dumps(result, indent=2))
    return ModelArtifacts(
        result=result,
        mask_dir=mask_dir,
        latencies_ms=latencies_ms,
        last_overlay=last_overlay,
    )


def masks_to_label_map(masks: np.ndarray, object_ids: list[int]) -> np.ndarray:
    if masks.ndim != 3:
        raise ValueError(f"masks must have shape [objects, height, width], got {masks.shape}")
    label_map = np.zeros(masks.shape[1:], dtype=np.uint8)
    for mask, object_id in zip(masks, object_ids, strict=False):
        if not 0 < object_id < 256:
            raise ValueError(f"object ID must fit uint8 label map: {object_id}")
        label_map[mask] = object_id
    return label_map


def masks_from_label_map(label_map: np.ndarray, object_count: int) -> np.ndarray:
    if label_map.ndim != 2:
        raise ValueError(f"label map must have shape [height, width], got {label_map.shape}")
    if not 1 <= object_count <= MAX_OBJECT_COUNT:
        raise ValueError(
            f"object count must be between 1 and {MAX_OBJECT_COUNT}, got {object_count}"
        )
    return np.stack(
        [label_map == object_id for object_id in range(1, object_count + 1)],
        axis=0,
    )


def choose_video_save(
    previews: list[np.ndarray],
    display_max_width: int,
    display_max_height: int,
    *,
    default_timing_mode: str,
    default_speed: float,
    speed_overrides: dict[str, float] | None = None,
) -> SaveDecision:
    if not previews:
        return SaveDecision(None, default_speed)
    preview_width = max(1, display_max_width // len(previews))
    displays: list[np.ndarray] = []
    for preview in previews:
        scale = display_scale(
            preview.shape[:2],
            preview_width,
            display_max_height,
        )
        if scale < 1.0:
            preview = cv2.resize(
                preview,
                (
                    max(1, int(round(preview.shape[1] * scale))),
                    max(1, int(round(preview.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        displays.append(preview)
    min_height = min(display.shape[0] for display in displays)
    displays = [
        cv2.resize(
            display,
            (
                max(1, int(round(display.shape[1] * min_height / display.shape[0]))),
                min_height,
            ),
            interpolation=cv2.INTER_AREA,
        )
        if display.shape[0] != min_height
        else display
        for display in displays
    ]
    combined = np.hstack(displays)
    window_name = "SAM2 video demo complete"
    speed = default_speed
    speed_overrides = speed_overrides or {}
    print("Tracking complete. F: without latency  L: with latency  B: both")
    if speed_overrides:
        configured_speeds = ", ".join(
            f"{model_id}={model_speed:g}x"
            for model_id, model_speed in speed_overrides.items()
        )
        print(f"Per-model latency speeds: {configured_speeds}")
        print("Enter/Space/S uses command settings. N/Q/Esc: discard")
    else:
        print(
            "Set latency-video speed with 1/2/4/8. "
            "Enter/Space/S uses command defaults. N/Q/Esc: discard"
        )
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            display = combined.copy()
            speed_label = (
                "Per-model latency speeds configured"
                if speed_overrides
                else f"Default {default_timing_mode} | Latency-video speed {speed:g}x"
            )
            _draw_label(
                display,
                speed_label,
                (18, max(60, display.shape[0] - 24)),
                (40, 220, 255),
            )
            cv2.imshow(window_name, display)
            key = cv2.waitKey(20) & 0xFF
            if not speed_overrides and key in {ord("1"), ord("2"), ord("4"), ord("8")}:
                speed = float(chr(key))
                print(f"Latency-video speed set to {speed:g}x")
                continue
            if key in {ord("f"), ord("F")}:
                return SaveDecision("source_fps", 1.0)
            if key in {ord("l"), ord("L")}:
                return SaveDecision("realtime", speed)
            if key in {ord("b"), ord("B")}:
                return SaveDecision("both", speed)
            if key in {10, 13, 32, ord("s"), ord("S"), ord("y"), ord("Y")}:
                return SaveDecision(default_timing_mode, speed)
            if key in {27, ord("n"), ord("N"), ord("q"), ord("Q")}:
                return SaveDecision(None, speed)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                return SaveDecision(None, speed)
    finally:
        cv2.destroyWindow(window_name)


def save_model_outputs(
    artifacts: ModelArtifacts,
    *,
    video_path: Path,
    video_info: VideoInfo,
    output_dir: Path,
    timing_mode: str,
    speed: float,
    codec: str,
    preset: str,
    crf: int,
) -> dict[str, str]:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to save the overlay videos")
    if speed <= 0.0:
        raise ValueError("save speed must be positive")
    codec = resolve_ffmpeg_codec(codec)
    modes = (
        ("source_fps",) if timing_mode == "source_fps"
        else ("realtime",) if timing_mode == "realtime"
        else ("source_fps", "realtime")
    )
    output_paths: dict[str, str] = {}
    for mode in modes:
        mode_speed = speed if mode == "realtime" else 1.0
        speed_suffix = (
            ""
            if mode_speed == 1.0
            else f"_{_format_speed(mode_speed)}x"
        )
        output_path = output_dir / (
            f"{artifacts.result['model_id']}_{mode}{speed_suffix}.mp4"
        )
        _encode_model_output(
            artifacts,
            video_path=video_path,
            video_info=video_info,
            output_path=output_path,
            mode=mode,
            speed=mode_speed,
            codec=codec,
            preset=preset,
            crf=crf,
        )
        output_paths[mode] = str(output_path)
    return output_paths


def _encode_model_output(
    artifacts: ModelArtifacts,
    *,
    video_path: Path,
    video_info: VideoInfo,
    output_path: Path,
    mode: str,
    speed: float,
    codec: str,
    preset: str,
    crf: int,
) -> None:
    writer = FFmpegVideoWriter(
        output_path,
        width=video_info.width,
        height=video_info.height,
        fps=video_info.fps,
        source_path=video_path if mode == "source_fps" else None,
        codec=codec,
        preset=preset,
        crf=crf,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        writer.close()
        raise RuntimeError(f"failed to reopen video for encoding: {video_path}")
    object_count = int(artifacts.result["object_count"])
    previous_overlay: np.ndarray | None = None
    print(f"Encoding {output_path}")
    try:
        for frame_index, latency_ms in enumerate(artifacts.latencies_ms):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"source video ended before frame {frame_index}")
            label_map = cv2.imread(
                str(artifacts.mask_dir / f"{frame_index:06d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )
            if label_map is None:
                raise RuntimeError(f"missing temporary mask for frame {frame_index}")
            masks = masks_from_label_map(label_map, object_count)
            overlay = overlay_masks(
                frame,
                masks,
                list(range(1, object_count + 1)),
                model_id=str(artifacts.result["model_id"]),
                frame_index=frame_index,
                latency_ms=latency_ms,
            )
            repeat_count = 1
            if mode == "realtime":
                effective_latency_ms = latency_ms
                if frame_index == 0:
                    effective_latency_ms += float(artifacts.result["prompt_ms"])
                repeat_count = realtime_repeat_count(
                    effective_latency_ms,
                    video_info.fps,
                    speed,
                )
                for output_frame in smooth_transition_frames(
                    previous_overlay,
                    overlay,
                    repeat_count,
                ):
                    writer.write(output_frame)
            else:
                writer.write(overlay)
            previous_overlay = overlay
            if (frame_index + 1) % 100 == 0:
                print(f"Encoded {frame_index + 1}/{len(artifacts.latencies_ms)} frames")
    finally:
        capture.release()
        writer.close()


def build_predictor(
    model_spec: ModelSpec,
    *,
    external_repo: str,
    sam2_distill_root: str,
    model_config: str,
    device: str,
) -> tuple[Any, Any, dict[str, Any]]:
    _prepend_repo_path(external_repo)
    torch_module = _import_required("torch")
    builder = _import_required("sam2.build_sam")
    full_checkpoint = (
        model_spec.sam2_checkpoint_path
        if model_spec.model_kind == "stage1-student"
        else model_spec.checkpoint_path
    )
    predictor = builder.build_sam2_video_predictor(
        config_file=model_config,
        ckpt_path=full_checkpoint,
        device=device,
        apply_postprocessing=False,
        hydra_overrides_extra=["++model.non_overlap_masks=false"],
    )
    predictor.eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    load_summary: dict[str, Any] = {
        "checkpoint_path": model_spec.checkpoint_path,
        "sam2_checkpoint_path": full_checkpoint,
        "model_config": model_config,
    }
    if model_spec.model_kind == "stage1-student":
        load_summary.update(
            patch_stage1_forward_image(
                predictor,
                torch_module,
                sam2_distill_root=sam2_distill_root,
                student_checkpoint_path=model_spec.checkpoint_path,
                sam2_checkpoint_path=full_checkpoint,
                device=device,
                requested_family=model_spec.student_family,
                requested_model_name=model_spec.student_model_name,
                requested_backbone_checkpoint=model_spec.student_backbone_checkpoint,
                legacy_tinyvit_checkpoint="",
                requested_adapter_mode=model_spec.student_adapter_mode,
                fallback_model_name=model_spec.student_model_name,
            )
        )
    return predictor, torch_module, load_summary


def scale_prompts(
    prompts: list[dict[str, Any]],
    *,
    source_size: tuple[int, int],
    inference_size: tuple[int, int],
) -> list[dict[str, Any]]:
    source_width, source_height = source_size
    inference_width, inference_height = inference_size
    scale_x = inference_width / float(source_width)
    scale_y = inference_height / float(source_height)
    scaled: list[dict[str, Any]] = []
    for prompt in prompts:
        if prompt["prompt_mode"] == "box":
            x1, y1, x2, y2 = prompt["box"]
            scaled.append(
                {
                    "prompt_mode": "box",
                    "box": [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y],
                }
            )
        else:
            x, y = prompt["point"]
            scaled.append(
                {
                    "prompt_mode": "point",
                    "point": [x * scale_x, y * scale_y],
                    "label": int(prompt.get("label", 1)),
                }
            )
    return scaled


def resize_masks(mask_logits: Any, *, target_size: tuple[int, int]) -> np.ndarray:
    values = mask_logits.detach().float().cpu().numpy() if hasattr(mask_logits, "detach") else np.asarray(mask_logits)
    masks = values > 0
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    target_width, target_height = target_size
    if masks.shape[-2:] == (target_height, target_width):
        return masks.astype(bool)
    return np.stack(
        [
            cv2.resize(
                mask.astype(np.uint8),
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            for mask in masks
        ],
        axis=0,
    )


def overlay_masks(
    frame_bgr: np.ndarray,
    masks: np.ndarray,
    object_ids: list[int],
    *,
    model_id: str,
    frame_index: int,
    latency_ms: float,
    display_fps: float | None = None,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    for mask, object_id in zip(masks, object_ids, strict=False):
        color = OBJECT_COLORS[(object_id - 1) % len(OBJECT_COLORS)]
        overlay[mask] = (
            overlay[mask].astype(np.float32) * 0.55
            + np.asarray(color, dtype=np.float32) * 0.45
        ).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        ys, xs = np.nonzero(mask)
        if xs.size:
            _draw_label(
                overlay,
                f"ID {object_id}",
                (int(xs.min()), max(24, int(ys.min()) - 8)),
                color,
            )
    status = (
        f"{model_id}  frame {frame_index}  objects {len(object_ids)}  "
        f"{latency_ms:.1f} ms"
    )
    if display_fps is not None:
        status += f"  display {display_fps:.1f} FPS"
    _draw_status(overlay, status)
    return overlay


def realtime_repeat_count(
    latency_ms: float,
    fps: float,
    speed: float = 1.0,
) -> int:
    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    return max(
        1,
        int(math.ceil(max(0.0, latency_ms) * fps / (1000.0 * speed))),
    )


def smooth_transition_frames(
    previous_frame: np.ndarray | None,
    current_frame: np.ndarray,
    frame_count: int,
) -> Iterator[np.ndarray]:
    if frame_count < 1:
        raise ValueError("frame count must be positive")
    if previous_frame is None:
        for _ in range(frame_count):
            yield current_frame
        return
    if previous_frame.shape != current_frame.shape:
        raise ValueError("transition frames must have matching shapes")
    if frame_count == 1:
        yield current_frame
        return
    for step in range(1, frame_count):
        yield cv2.addWeighted(
            previous_frame,
            1.0 - step / frame_count,
            current_frame,
            step / frame_count,
            0.0,
        )
    yield current_frame


def rolling_fps(completion_times: deque[float]) -> float | None:
    if len(completion_times) < 2:
        return None
    duration = completion_times[-1] - completion_times[0]
    if duration <= 0.0:
        return None
    return (len(completion_times) - 1) / duration


def ffmpeg_video_args(
    codec: str,
    *,
    preset: str,
    crf: int,
    width: int,
    height: int,
    fps: float,
) -> list[str]:
    if codec in {"libx264", "libx265"}:
        return ["-c:v", codec, "-preset", preset, "-crf", str(crf)]
    if codec == "mpeg4":
        return ["-c:v", codec, "-q:v", "2"]
    bitrate_mbps = max(12, int(math.ceil(width * height * fps * 0.16 / 1_000_000.0)))
    return ["-c:v", codec, "-b:v", f"{bitrate_mbps}M"]


def resolve_ffmpeg_codec(requested_codec: str) -> str:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=False,
        capture_output=True,
        text=True,
    )
    encoder_text = result.stdout + result.stderr
    available = {
        line.split()[1]
        for line in encoder_text.splitlines()
        if len(line.split()) >= 2 and line.lstrip().startswith("V")
    }
    if requested_codec in available:
        return requested_codec
    if requested_codec != "libx264":
        raise RuntimeError(
            f"requested FFmpeg encoder is unavailable: {requested_codec}"
        )
    for candidate in ("h264_nvmpi", "h264_v4l2m2m", "h264_nvenc", "mpeg4"):
        if candidate in available:
            print(
                f"FFmpeg encoder libx264 is unavailable; using {candidate} instead"
            )
            return candidate
    raise RuntimeError(
        "no supported FFmpeg video encoder is available; checked libx264, "
        "h264_nvmpi, h264_v4l2m2m, h264_nvenc, and mpeg4"
    )


def _init_state(predictor: Any, frame_dir: Path) -> dict[str, Any]:
    parameters = inspect.signature(predictor.init_state).parameters
    kwargs: dict[str, Any] = {"video_path": str(frame_dir)}
    if "offload_video_to_cpu" in parameters:
        kwargs["offload_video_to_cpu"] = True
    if "offload_state_to_cpu" in parameters:
        kwargs["offload_state_to_cpu"] = True
    if "async_loading_frames" in parameters:
        kwargs["async_loading_frames"] = True
    return predictor.init_state(**kwargs)


def _add_prompt(
    predictor: Any,
    inference_state: dict[str, Any],
    object_id: int,
    prompt: dict[str, Any],
) -> None:
    if prompt["prompt_mode"] == "box":
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            box=np.asarray(prompt["box"], dtype=np.float32),
        )
    else:
        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=object_id,
            points=np.asarray([prompt["point"]], dtype=np.float32),
            labels=np.asarray([prompt.get("label", 1)], dtype=np.int32),
        )


def _sync(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _show_tracking_frame(
    frame_bgr: np.ndarray,
    display_max_width: int,
    display_max_height: int,
    save_control: SaveControl,
) -> None:
    display_frame = frame_bgr.copy()
    if save_control.armed:
        _draw_label(
            display_frame,
            "SAVE ARMED",
            (18, max(60, display_frame.shape[0] - 24)),
            (40, 220, 255),
        )
    scale = display_scale(
        display_frame.shape[:2],
        display_max_width,
        display_max_height,
    )
    if scale < 1.0:
        display = cv2.resize(
            display_frame,
            (
                max(1, int(round(display_frame.shape[1] * scale))),
                max(1, int(round(display_frame.shape[0] * scale))),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = display_frame
    cv2.imshow("SAM2 video demo tracking", display)
    key = cv2.waitKey(1) & 0xFF
    if key in {27, ord("q"), ord("Q")}:
        raise KeyboardInterrupt
    if key in {ord("s"), ord("S")}:
        save_control.armed = not save_control.armed
        state = "armed" if save_control.armed else "disarmed"
        print(f"Automatic video save {state}")


def display_scale(
    frame_hw: tuple[int, int],
    max_width: int,
    max_height: int,
) -> float:
    height, width = frame_hw
    width_scale = max_width / float(width) if max_width > 0 else 1.0
    height_scale = max_height / float(height) if max_height > 0 else 1.0
    return min(1.0, width_scale, height_scale)


def _draw_label(
    frame_bgr: np.ndarray,
    text: str,
    point: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame_bgr,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_status(frame_bgr: np.ndarray, text: str) -> None:
    cv2.putText(
        frame_bgr,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 0),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr,
        text,
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _ffprobe_fps(video_path: Path) -> float:
    if shutil.which("ffprobe") is None:
        return 0.0
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0.0
    try:
        return float(Fraction(result.stdout.strip()))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _ffprobe_audio_stream_index(video_path: Path) -> int | None:
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index,codec_name",
            "-of",
            "json",
            str(video_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return select_audio_stream_index(payload)


def select_audio_stream_index(payload: dict[str, Any]) -> int | None:
    streams = payload.get("streams", [])
    if not isinstance(streams, list):
        return None
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        codec_name = str(stream.get("codec_name", "")).strip().lower()
        if codec_name in {"", "none", "unknown"}:
            continue
        try:
            return int(stream["index"])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def _format_fps(fps: float) -> str:
    return f"{fps:.8f}".rstrip("0").rstrip(".")


def _format_speed(speed: float) -> str:
    return f"{speed:g}".replace(".", "p")


def _resolve_output_dir(raw_output_dir: str, video_path: Path) -> Path:
    if raw_output_dir:
        return Path(raw_output_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("overlays/thor/video_demo") / f"{run_id}_{video_path.stem}"


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


if __name__ == "__main__":
    main()
