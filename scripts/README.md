# Script guide

Prefer installed `sam-*` CLIs for reusable benchmark operations. `data/`
prepares inputs, `setup/` installs environments, `pace/` owns cluster recipes,
and `thor/` owns device workflows. Run shell recipes from the repository root.

All in-repository callers have been updated. External automation must update
its paths using the map below; argument names and model behavior are unchanged.
Research outputs belong outside source control.

## Path migration

| Previous path | Current path |
| --- | --- |
| `scripts/check_pace_qos.sh` | `scripts/pace/check_pace_qos.sh` |
| `scripts/check_storage_budget.sh` | `scripts/data/check_storage_budget.sh` |
| `scripts/download_coco_val2017.sh` | `scripts/data/download_coco_val2017.sh` |
| `scripts/download_efficientsam3_checkpoints.sh` | `scripts/data/download_efficientsam3_checkpoints.sh` |
| `scripts/download_hf_checkpoints_via_git.sh` | `scripts/data/download_hf_checkpoints_via_git.sh` |
| `scripts/download_instinctsam_compressed_checkpoints.sh` | `scripts/data/download_instinctsam_compressed_checkpoints.sh` |
| `scripts/download_instinctsam_vitb_checkpoint.sh` | `scripts/data/download_instinctsam_vitb_checkpoint.sh` |
| `scripts/download_saco_sav_media.sh` | `scripts/data/download_saco_sav_media.sh` |
| `scripts/download_saco_stream_assets.sh` | `scripts/data/download_saco_stream_assets.sh` |
| `scripts/download_sam2_distill_tinyvit_init_weights.sh` | `scripts/data/download_sam2_distill_tinyvit_init_weights.sh` |
| `scripts/download_sam2_family_checkpoints.sh` | `scripts/data/download_sam2_family_checkpoints.sh` |
| `scripts/download_sam3_checkpoint.sh` | `scripts/data/download_sam3_checkpoint.sh` |
| `scripts/download_sav_valtest_subset.sh` | `scripts/data/download_sav_valtest_subset.sh` |
| `scripts/download_yoloe_edgetam_mobilesam_assets.sh` | `scripts/data/download_yoloe_edgetam_mobilesam_assets.sh` |
| `scripts/pace_gpu_tinyvit_trt_attention_block_sensitivity.sbatch` | `scripts/pace/pace_gpu_tinyvit_trt_attention_block_sensitivity.sbatch` |
| `scripts/pace_gpu_tinyvit_trt_candidate_mask_parity.sbatch` | `scripts/pace/pace_gpu_tinyvit_trt_candidate_mask_parity.sbatch` |
| `scripts/pace_gpu_tinyvit_trt_layer_sensitivity.sbatch` | `scripts/pace/pace_gpu_tinyvit_trt_layer_sensitivity.sbatch` |
| `scripts/pace_gpu_tinyvit_trt_role_sensitivity.sbatch` | `scripts/pace/pace_gpu_tinyvit_trt_role_sensitivity.sbatch` |
| `scripts/pace_l40s_benchmark.sbatch` | `scripts/pace/pace_l40s_benchmark.sbatch` |
| `scripts/pace_l40s_coco_suite.sbatch` | `scripts/pace/pace_l40s_coco_suite.sbatch` |
| `scripts/pace_l40s_mobilesam_coco.sbatch` | `scripts/pace/pace_l40s_mobilesam_coco.sbatch` |
| `scripts/pace_l40s_profile_efficientsam3.sbatch` | `scripts/pace/pace_l40s_profile_efficientsam3.sbatch` |
| `scripts/pace_l40s_profile_sam3.sbatch` | `scripts/pace/pace_l40s_profile_sam3.sbatch` |
| `scripts/pace_l40s_sampled_camera_frame_smoke.sbatch` | `scripts/pace/pace_l40s_sampled_camera_frame_smoke.sbatch` |
| `scripts/pace_l40s_sav_video_sam2_family.sbatch` | `scripts/pace/pace_l40s_sav_video_sam2_family.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_aux_streams.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_aux_streams.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_compare_engines.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_compare_engines.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_custom_mask_parity.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_custom_mask_parity.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_e2e.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_e2e.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_encoder_smoke.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_encoder_smoke.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_mask_parity.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_mask_parity.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_mixed_precision.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_mixed_precision.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_optimization_matrix.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_optimization_matrix.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_pipeline_pair.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_pipeline_pair.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_pipeline_sweep.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_pipeline_sweep.sbatch` |
| `scripts/pace_l40s_tinyvit_trt_quantization.sbatch` | `scripts/pace/pace_l40s_tinyvit_trt_quantization.sbatch` |
| `scripts/pace_l40s_tv21_patch_fp8_calibration.sbatch` | `scripts/pace/pace_l40s_tv21_patch_fp8_calibration.sbatch` |
| `scripts/pace_l40s_yoloe_edgetam_poc.sbatch` | `scripts/pace/pace_l40s_yoloe_edgetam_poc.sbatch` |
| `scripts/pace_l40s_yoloe_edgetam_sav_text.sbatch` | `scripts/pace/pace_l40s_yoloe_edgetam_sav_text.sbatch` |
| `scripts/pace_prepare_sav_salient_subset.sbatch` | `scripts/pace/pace_prepare_sav_salient_subset.sbatch` |
| `scripts/pace_tinyvit_trt_compare_engines.py` | `scripts/pace/pace_tinyvit_trt_compare_engines.py` |
| `scripts/pace_tinyvit_trt_encoder_smoke.py` | `scripts/pace/pace_tinyvit_trt_encoder_smoke.py` |
| `scripts/pace_tinyvit_trt_mask_parity.py` | `scripts/pace/pace_tinyvit_trt_mask_parity.py` |
| `scripts/prepare_benchmark_datasets.sh` | `scripts/data/prepare_benchmark_datasets.sh` |
| `scripts/prepare_coco_fixed_subset.sh` | `scripts/data/prepare_coco_fixed_subset.sh` |
| `scripts/prepare_sa1b_fixed_subset.sh` | `scripts/data/prepare_sa1b_fixed_subset.sh` |
| `scripts/prepare_sav_fixed10_subset.sh` | `scripts/data/prepare_sav_fixed10_subset.sh` |
| `scripts/prepare_sav_salient_subset.sh` | `scripts/data/prepare_sav_salient_subset.sh` |
| `scripts/run_pace_thor_pipeline_smoke.sh` | `scripts/pace/run_pace_thor_pipeline_smoke.sh` |
| `scripts/run_pipeline_bottleneck_matrix.sh` | `scripts/thor/run_pipeline_bottleneck_matrix.sh` |
| `scripts/run_sampled_camera_frame_smoke.sh` | `scripts/thor/run_sampled_camera_frame_smoke.sh` |
| `scripts/run_thor_coco_all_benchmarks.sh` | `scripts/thor/run_thor_coco_all_benchmarks.sh` |
| `scripts/run_thor_formal_full_matrix.sh` | `scripts/thor/run_thor_formal_full_matrix.sh` |
| `scripts/run_thor_formal_smoke_matrix.sh` | `scripts/thor/run_thor_formal_smoke_matrix.sh` |
| `scripts/run_thor_multi_prompt_image_benchmark.sh` | `scripts/thor/run_thor_multi_prompt_image_benchmark.sh` |
| `scripts/run_thor_ros_saco_stream_suite.sh` | `scripts/thor/run_thor_ros_saco_stream_suite.sh` |
| `scripts/run_thor_sa1b_image_benchmarks.sh` | `scripts/thor/run_thor_sa1b_image_benchmarks.sh` |
| `scripts/run_thor_saco_video_and_image_per_frame.sh` | `scripts/thor/run_thor_saco_video_and_image_per_frame.sh` |
| `scripts/run_thor_sam2_distill_sav_suite.sh` | `scripts/thor/run_thor_sam2_distill_sav_suite.sh` |
| `scripts/run_thor_sav_image_box_benchmarks.sh` | `scripts/thor/run_thor_sav_image_box_benchmarks.sh` |
| `scripts/run_thor_yolo_coco_suite.sh` | `scripts/thor/run_thor_yolo_coco_suite.sh` |
| `scripts/setup_5090_offline_benchmark.sh` | `scripts/setup/setup_5090_offline_benchmark.sh` |
| `scripts/setup_model_repos.sh` | `scripts/setup/setup_model_repos.sh` |
| `scripts/setup_pace_ros_conda.sh` | `scripts/setup/setup_pace_ros_conda.sh` |
| `scripts/setup_pace_venv.sh` | `scripts/setup/setup_pace_venv.sh` |
| `scripts/setup_thor_saco_stream_benchmark.sh` | `scripts/setup/setup_thor_saco_stream_benchmark.sh` |
| `scripts/source_thor_ros_env.sh` | `scripts/thor/source_thor_ros_env.sh` |
| `scripts/summarize_thor_saco_model_results.sh` | `scripts/thor/summarize_thor_saco_model_results.sh` |
| `scripts/thor_env_probe.sh` | `scripts/thor/thor_env_probe.sh` |
