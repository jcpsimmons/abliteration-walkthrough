# Methodology notes

This is a reproducible small-model experiment inspired by Arditi et al. It is not a reproduction of every experiment or evaluator in the paper.

1. Both complete public datasets are vendored at pinned revisions and verified against SHA-256 checksums. Upstream splits are pooled and repartitioned for this demonstration. The full mode requires every harmful-source row to survive deduplication, cross-label overlap and actual chat-template length checks; otherwise it stops. Seed 42 selects a matched 520-row benign sample. Each group reserves 104 previously unseen test rows, then 104 validation rows, leaving 312 training candidates. Test rows exclude all IDs processed in the earlier pilot, recorded in `datasets/pilot-exposure.json`. All training candidates are screened; only observed harmful refusals and benign non-refusals enter the mean difference. Validation and test are unfiltered. No external judge is used.
2. Every forward pass uses the same explicit system prompt, tokenizer chat template, assistant generation prefix, left padding, and greedy decoding settings. Activations are post-block residual values at the final prompt position. Long prompts are rejected rather than silently truncating the assistant prefix.
3. Candidate directions are normalized mean differences. Runtime interventions project the embedding output and both residual-writing outputs per block. Those placements are chosen to match the permanent weight edit. Candidate selection uses validation response-prefix markers plus first-token KL on benign validation prompts.
4. Weight editing preserves the output head by untying it from the embedding. The actual model checks hook/weight logit agreement on a benign probe. Unit tests also verify a complete save/reload roundtrip. Post-edit states are captured from the reloaded FP32 model.
5. The held-out test split reports refusal markers, benign refusal markers, first-token KL, and eight exact-answer smoke checks. Raw outputs are saved locally for human review. A separate all-data diagnostic evaluates all 520 assigned rows per group, including training and validation; it must not be presented as held-out evidence. No harmful-compliance score, general intelligence score, healing result, or universal safety-removal claim is implied.

## Corrections to the simplified story

- The difference of means identifies a candidate, not a guaranteed pure refusal feature. Subtraction does not automatically cancel systematic dataset confounds.
- The six-row toy table gives an exact x difference of `-1/30`, approximately `-0.0333`, not `-0.04`. Its unit direction is approximately `(-0.0476, 0.9989)`. The GPS examples deliberately round it to `(0,1)`.
- Individual real hidden coordinates are not named concepts. A two-axis chart throws away most of the vector's dimensions.
- Layer choice is empirical. Do not promise the middle block wins or that every architecture stores refusal in a single direction.
- This Qwen uses SiLU-gated MLPs. A description of simply zeroing negative values describes a different activation function.
- Precision is about significant digits and relative resolution, not a fixed number of decimal places. Converting BF16 to FP32 creates room for more precise arithmetic; it cannot recover already-rounded weights.
- Residual writer projection does not make every normalized or transformed activation orthogonal. Final RMSNorm changes coordinates. The near-zero chart samples the residual before that normalization.
- This implementation changes storage: FP32 and an untied output head produce a larger saved model. Q8_0 export is an additional quantization step.
- Weight arithmetic time is different from download time, direction search, evaluation, export, and total preparation time.

## Reproducibility

`config.json` pins all source revisions. `uv.lock` pins the dependency resolution. Per-step records retain the configuration hash, timestamps, and wall time. The report identifies the device and torch version. CPU, MPS, and CUDA can yield small numerical and generation differences, especially around close candidate scores. Record your own measured result rather than copying the production run's numbers.

The earlier 48/12/24 configuration is retained as `config.demo.json`. Its legacy split and candidate-cap behavior are separate from the new full run. The full run changes the sample and selection data; its results can differ from the pilot. The held-out prompts are fresh to this experiment, not guaranteed absent from the base model's pretraining.
