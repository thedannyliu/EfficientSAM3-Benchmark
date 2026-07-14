from __future__ import annotations

from copy import deepcopy
from typing import Any


class OnlineFrameStore:
    """Append-only frame storage compatible with SAM3 tensor indexing."""

    def __init__(self, frames: list[Any] | None = None) -> None:
        self._frames = list(frames or [])
        self._pruned_before = 0

    def append(self, frame: Any) -> None:
        self._frames.append(frame)

    def prune_before(self, frame_idx: int) -> None:
        stop = min(frame_idx, len(self._frames))
        for idx in range(self._pruned_before, stop):
            self._frames[idx] = None
        self._pruned_before = max(self._pruned_before, stop)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: Any) -> Any:
        if hasattr(index, "detach"):
            index = index.detach().cpu()
            if index.ndim == 0:
                index = int(index.item())
            else:
                index = index.tolist()
        if isinstance(index, slice):
            indices = list(range(len(self._frames)))[index]
            return self._stack(indices)
        if isinstance(index, (list, tuple)):
            return self._stack([int(value) for value in index])
        frame = self._frames[int(index)]
        if frame is None:
            raise RuntimeError(f"SAM3 online frame {int(index)} has been pruned")
        return frame

    def _stack(self, indices: list[int]) -> Any:
        frames = [self[index] for index in indices]
        if not frames:
            raise RuntimeError("SAM3 online frame selection is empty")
        import torch

        return torch.stack(frames, dim=0)


def initialize_sam3_online_state(inference_state: dict[str, Any]) -> None:
    """Convert a one-frame SAM3 state into an appendable online state."""
    input_batch = inference_state["input_batch"]
    images = input_batch.img_batch
    if len(images) != 1:
        raise ValueError("SAM3 online state must be initialized with exactly one frame")
    input_batch.img_batch = OnlineFrameStore([images[0]])
    inference_state["_online_find_stage_template"] = deepcopy(
        input_batch.find_inputs[0]
    )
    inference_state["is_image_only"] = False


def append_sam3_online_frame(
    inference_state: dict[str, Any], frame_tensor: Any
) -> int:
    """Append one preprocessed frame and extend SAM3's per-frame state."""
    input_batch = inference_state["input_batch"]
    if not isinstance(input_batch.img_batch, OnlineFrameStore):
        raise TypeError("initialize_sam3_online_state must be called first")

    frame_idx = int(inference_state["num_frames"])
    input_batch.img_batch.append(frame_tensor)

    stage = deepcopy(inference_state["_online_find_stage_template"])
    stage.img_ids = stage.img_ids.new_tensor([frame_idx])
    stage.text_ids = input_batch.find_inputs[-1].text_ids.clone()
    input_batch.find_inputs.append(stage)
    input_batch.find_targets.append(None)
    input_batch.find_metadatas.append(None)

    for key in (
        "previous_stages_out",
        "per_frame_raw_point_input",
        "per_frame_raw_box_input",
        "per_frame_visual_prompt",
        "per_frame_geometric_prompt",
    ):
        inference_state[key].append(None)
    inference_state["per_frame_cur_step"].append(0)
    inference_state["num_frames"] = frame_idx + 1

    for tracker_state in inference_state["tracker_inference_states"]:
        tracker_state["num_frames"] = frame_idx + 1
    return frame_idx


def run_sam3_online_step(
    model: Any,
    inference_state: dict[str, Any],
    frame_tensor: Any,
    torch_module: Any,
) -> tuple[int, dict[str, Any]]:
    """Run one native SAM3 frame update without enabling autograd."""
    with torch_module.inference_mode():
        frame_idx = append_sam3_online_frame(inference_state, frame_tensor)
        out = model._run_single_frame_inference(
            inference_state, frame_idx, reverse=False
        )
        outputs = model._postprocess_output(
            inference_state,
            out,
            removed_obj_ids=out.get("removed_obj_ids"),
            suppressed_obj_ids=out.get("suppressed_obj_ids"),
            unconfirmed_obj_ids=out.get("unconfirmed_obj_ids"),
        )
    return frame_idx, outputs


def prune_sam3_online_state(
    inference_state: dict[str, Any], frame_idx: int, history_size: int
) -> None:
    """Release old frame tensors and non-conditioning outputs."""
    min_keep = max(0, frame_idx - max(1, history_size) + 1)
    frame_store = inference_state["input_batch"].img_batch
    if isinstance(frame_store, OnlineFrameStore):
        frame_store.prune_before(min_keep)

    input_batch = inference_state["input_batch"]
    pruned_before = int(inference_state.get("_online_pruned_before", 0))
    for old_idx in range(pruned_before, min_keep):
        input_batch.find_inputs[old_idx] = None
        input_batch.find_targets[old_idx] = None
        input_batch.find_metadatas[old_idx] = None
    inference_state["_online_pruned_before"] = max(pruned_before, min_keep)

    for cache_name in ("cached_frame_outputs",):
        cache = inference_state.get(cache_name, {})
        for old_idx in [key for key in cache if isinstance(key, int) and key < min_keep]:
            cache.pop(old_idx, None)

    feature_cache = inference_state.get("feature_cache", {})
    for old_idx in [key for key in feature_cache if isinstance(key, int) and key < min_keep]:
        feature_cache.pop(old_idx, None)

    for tracker_state in inference_state.get("tracker_inference_states", []):
        output_dict = tracker_state.get("output_dict", {})
        non_cond = output_dict.get("non_cond_frame_outputs", {})
        for old_idx in [key for key in non_cond if key < min_keep]:
            non_cond.pop(old_idx, None)
        frames_tracked = tracker_state.get("frames_already_tracked", {})
        for old_idx in [key for key in frames_tracked if key < min_keep]:
            frames_tracked.pop(old_idx, None)
