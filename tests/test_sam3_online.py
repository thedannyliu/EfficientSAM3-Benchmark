from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from sam_backend.sam3_online import (
    OnlineFrameStore,
    append_sam3_online_frame,
    initialize_sam3_online_state,
    prune_sam3_online_state,
    run_sam3_online_step,
)


class Sam3OnlineStateTest(unittest.TestCase):
    def _state(self) -> dict:
        stage = SimpleNamespace(
            img_ids=torch.tensor([0]),
            text_ids=torch.tensor([0]),
            payload=torch.tensor([7]),
        )
        return {
            "input_batch": SimpleNamespace(
                img_batch=torch.arange(12).reshape(1, 3, 2, 2),
                find_inputs=[stage],
                find_targets=[None],
                find_metadatas=[None],
            ),
            "num_frames": 1,
            "is_image_only": True,
            "tracker_inference_states": [{"num_frames": 1}],
            "previous_stages_out": [None],
            "per_frame_raw_point_input": [None],
            "per_frame_raw_box_input": [None],
            "per_frame_visual_prompt": [None],
            "per_frame_geometric_prompt": [None],
            "per_frame_cur_step": [0],
            "cached_frame_outputs": {},
            "feature_cache": {},
        }

    def test_frame_store_supports_tensor_indices_and_pruning(self) -> None:
        store = OnlineFrameStore([torch.tensor([1]), torch.tensor([2])])
        self.assertEqual(store[0].item(), 1)
        self.assertEqual(store[torch.tensor(1)].item(), 2)
        self.assertEqual(store[torch.tensor([0, 1])].tolist(), [[1], [2]])
        store.prune_before(1)
        with self.assertRaisesRegex(RuntimeError, "has been pruned"):
            _ = store[0]

    def test_append_extends_every_per_frame_container(self) -> None:
        state = self._state()
        initialize_sam3_online_state(state)
        state["input_batch"].find_inputs[0].text_ids[...] = 1

        frame_idx = append_sam3_online_frame(state, torch.ones(3, 2, 2))

        self.assertEqual(frame_idx, 1)
        self.assertEqual(state["num_frames"], 2)
        self.assertFalse(state["is_image_only"])
        self.assertEqual(state["tracker_inference_states"][0]["num_frames"], 2)
        self.assertEqual(state["input_batch"].find_inputs[1].img_ids.tolist(), [1])
        self.assertEqual(state["input_batch"].find_inputs[1].text_ids.tolist(), [1])
        for key in (
            "previous_stages_out",
            "per_frame_raw_point_input",
            "per_frame_raw_box_input",
            "per_frame_visual_prompt",
            "per_frame_geometric_prompt",
            "per_frame_cur_step",
        ):
            self.assertEqual(len(state[key]), 2)

    def test_prune_releases_old_frames_and_non_conditioning_outputs(self) -> None:
        state = self._state()
        initialize_sam3_online_state(state)
        for _ in range(3):
            append_sam3_online_frame(state, torch.ones(3, 2, 2))
        state["cached_frame_outputs"] = {0: "old", 3: "new"}
        state["feature_cache"] = {0: "old", 3: "new", "text": "keep"}
        state["tracker_inference_states"] = [
            {
                "num_frames": 4,
                "output_dict": {"non_cond_frame_outputs": {0: "old", 3: "new"}},
                "frames_already_tracked": {0: {}, 3: {}},
            }
        ]

        prune_sam3_online_state(state, frame_idx=3, history_size=2)

        with self.assertRaisesRegex(RuntimeError, "has been pruned"):
            _ = state["input_batch"].img_batch[0]
        self.assertIsNone(state["input_batch"].find_inputs[0])
        self.assertIsNotNone(state["input_batch"].find_inputs[2])
        self.assertEqual(state["input_batch"].img_batch[2].shape, (3, 2, 2))
        self.assertEqual(state["cached_frame_outputs"], {3: "new"})
        self.assertEqual(state["feature_cache"], {3: "new", "text": "keep"})
        tracker_state = state["tracker_inference_states"][0]
        self.assertEqual(
            tracker_state["output_dict"]["non_cond_frame_outputs"], {3: "new"}
        )
        self.assertEqual(tracker_state["frames_already_tracked"], {3: {}})
        self.assertEqual(
            append_sam3_online_frame(state, torch.ones(3, 2, 2)),
            4,
        )

    def test_online_step_runs_model_inside_inference_mode(self) -> None:
        state = self._state()
        initialize_sam3_online_state(state)

        class FakeModel:
            def _run_single_frame_inference(
                self, inference_state, frame_idx, reverse
            ):
                self.inference_mode_during_forward = torch.is_inference_mode_enabled()
                self.state_identity = id(inference_state)
                return {"obj_id_to_mask": {}, "frame_idx": frame_idx}

            def _postprocess_output(self, inference_state, out, **kwargs):
                self.inference_mode_during_postprocess = (
                    torch.is_inference_mode_enabled()
                )
                return {"out_binary_masks": [], "frame_idx": out["frame_idx"]}

        model = FakeModel()
        state_identity = id(state)

        frame_idx, outputs = run_sam3_online_step(
            model, state, torch.ones(3, 2, 2), torch
        )

        self.assertEqual(frame_idx, 1)
        self.assertEqual(outputs["frame_idx"], 1)
        self.assertEqual(state["num_frames"], 2)
        self.assertEqual(model.state_identity, state_identity)
        self.assertTrue(model.inference_mode_during_forward)
        self.assertTrue(model.inference_mode_during_postprocess)


if __name__ == "__main__":
    unittest.main()
