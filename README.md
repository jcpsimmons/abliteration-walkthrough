# Abliteration walkthrough

A terminal-first walkthrough of a refusal-direction weight edit on **Qwen2.5-0.5B-Instruct**. Each step explains what it measures, saves its result, and can be run separately. Built for The Dr. J Show by Dr. Josh C. Simmons.

## Start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
git clone https://github.com/jcpsimmons/abliteration-walkthrough.git
cd abliteration-walkthrough
uv sync --frozen
uv run python abliterate.py doctor
./record.sh
```

Press Enter to run each step. Press `q` to stop. On a first run this performs the computation. On later runs it explicitly labels saved results **CACHED RESULT**. You can stop between steps and resume with `run --step N`.

The repository remains private during production. The clone command will require access until release.

## Requirements

- Python 3.11 or 3.12 (uv can install it), internet for the first download, and several GB of free disk space. Allow 12 GB for model copies, exports, and caches.
- Apple Silicon with MPS, an NVIDIA CUDA device, or CPU. The production run was tested on Apple Silicon. CPU/CUDA portability is implemented but has not been hardware-tested here.
- This is a small-model demonstration. The default run assigns all 520 harmful-source prompts: 312 training candidates, 104 validation, and 104 test. It uses a deterministic matched sample of 520 harmless prompts. Only training candidates are filtered by baseline behavior.
- The model downloads from the pinned Hugging Face revision in `config.json`. Both complete dataset snapshots are committed in `datasets/` and checksum-verified before use. No API key, hosted inference service, or paid model call is required.

## The twelve steps

| Step | Command suffix | What happens |
| --- | --- | --- |
| 1 | `run --step 1` | Download and list the model files |
| 2 | `run --step 2` | Read the safetensors header and identify 49 residual writers |
| 3 | `run --step 3` | Verify local datasets, split, and screen all training candidates |
| 4 | `run --step 4` | Capture post-block, last-token residual states |
| 5 | `run --step 5` | Compute and normalize 24 difference-of-means directions |
| 6 | `run --step 6` | Test each direction on validation prompts; select by a stated score |
| 7 | `run --step 7` | Plot original states and validation scores |
| 8 | `run --step 8` | Edit weights, verify against hooks, and save the model |
| 9 | `run --step 9` | Reload the saved model and measure the changed states |
| 10 | `run --step 10` | Evaluate held-out prompts, all-data diagnostics, and a small benign smoke test |
| 11 | `run --step 11` | Optional Q8_0 GGUF export through llama.cpp |
| 12 | `run --step 12` | Save a measured summary and reusable charts |

Prefix a command suffix with `uv run python abliterate.py`. To prepare everything without pauses:

```sh
uv run python abliterate.py all
```

Step 11 is optional. Recompute a step with `--refresh`; dependent cached steps are invalidated. Changing `config.json` requires recomputing its prerequisites. For a wholly separate experiment, use `--out out/experiment-2` on every command.

```sh
uv run python abliterate.py run --step 6 --refresh
uv run python abliterate.py all
```

## What the arithmetic does

For a unit vector `r`, a residual state `x` becomes:

```python
x_projected = x - (x @ r).unsqueeze(-1) * r
```

We estimate a candidate `r` for each block from the mean of refused training states minus the mean of answered training states. The validation score is `refusal_fraction + 0.1 * benign_first_token_KL_nats`. This transparent teaching rule is not a claim of universal optimality.

For an attention or MLP output matrix with output coordinates on its rows:

```python
W -= r[:, None] * (r @ W)[None, :]
```

The embedding stores hidden coordinates on its columns, so its orientation differs:

```python
E -= (E @ r)[:, None] * r[None, :]
```

Qwen ties the embedding to the output classifier. This implementation clones and preserves the original output head before editing the embedding, sets `tie_word_embeddings=False`, and verifies persistence after reload. Consequently, the saved FP32 model is larger than the original BF16 download. It does not promise an unchanged file size.

The residual states are sampled after each decoder block, before the final RMSNorm. Capturing `output_hidden_states[-1]` without accounting for final normalization would measure a different state. The original and edited charts share the original PCA basis and common axis limits. Edited dots come from a new forward pass, not from moving the original plot points.

In both PCA charts, **X = harmful prompts** (red) and **O = harmless prompts** (cyan). Each mark is one prompt's 896-number internal state compressed to two plotting coordinates. PC1 and PC2 summarize variation; they are not harmfulness or refusal scores. The symbols identify the prompt sets, not whether the edited model refused or answered. Cached chart steps redraw the saved coordinates, preserving the symbols and colors.

## Prompts and recording

`prompts/demo.txt` is the swap file. It accepts literal prompts or `dataset:N`, which selects a prompt from the local held-out harmful-source set. If changing it after a run, rerun step 10 with `--refresh`, then step 12. A separate private file can be supplied with `--prompts prompts/local-demo.txt`.

**Prompt text and generated responses are hidden by default.** Terminal output shows IDs, counts, and detector results. Public source prompts live in the committed `datasets/` snapshots. Generated responses and run assignments are saved under ignored `out/private/`. A response with no refusal phrase can still be evasive, incorrect, or incoherent.

```sh
uv run python abliterate.py demo
# Only after reviewing the local demo off camera:
uv run python abliterate.py demo --reveal
```

Hiding text is a production control, not a guarantee about a platform's moderation or monetization decisions. Review any text you choose to put on screen.

## Chat with the original or edited model

From the repository folder:

```sh
./chat before
./chat after
```

`before` loads `out/base`; `after` loads `out/edited`. Both load the original or edited checkpoint from the experiment, run in FP32, and use the same system prompt. The banner identifies the model and device. Responses stream as they are generated. `/reset` clears the conversation; `/exit` quits. Chats are not saved.

For a one-shot question:

```sh
./chat before --message "What is 17 + 25? Answer with just the number."
./chat after --message "What is 17 + 25? Answer with just the number."
```

Chat allows 4,096 new tokens per reply by default, separate from the controlled experiment's 48-token limit. Adjust the reply cap with `--max-new-tokens`. The default 8,192-token budget reserves 4,096 tokens for the reply and leaves 4,096 for the system prompt, your message, and conversation history. The oldest complete turns are dropped when necessary; adjust the total budget with `--context-tokens`. Restart an existing chat to use the new defaults. Run step 1 before original-model chat and step 8 before edited-model chat.

## Complete datasets

The committed snapshots contain **520 harmful-source prompts (44,204 bytes)** and **31,323 harmless-source prompts (2,244,965 bytes)**, about **2.3 MB combined**. Each JSONL line is one original text row. `datasets/manifest.json` records pinned revisions, source split counts and SHA-256 checksums. See [dataset provenance](datasets/README.md).

The full run uses every harmful prompt and an equal-sized harmless sample. Fitting a direction and choosing a layer require separate data from the final test. The new 104-prompt test groups also exclude every ID processed in the earlier pilot. All 312 training candidates per group are screened; the accepted counts can differ. Step 10 additionally evaluates all 520 assigned prompts in each group before and after, reporting this as an **all-data diagnostic that includes training**, separately from the held-out result.

To reproduce the smaller pilot configuration in a separate output folder:

```sh
uv run python abliterate.py all --config config.demo.json --out out/pilot
```

The measured local comparison is saved in [the full-run report](results/full-2026-10-24.md), with machine-readable summary and verification records alongside it.

The legacy pilot mode downloads the pinned datasets; the default full mode reads the committed snapshots offline. Model download still requires internet on first use.

## Export to GGUF

Use a [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout with its converter dependencies installed in the same environment. The exporter accepts a path; it does not alter that checkout.

```sh
uv run python abliterate.py run --step 11 --llama-cpp /path/to/llama.cpp
cd out
ollama create abliteration-walkthrough -f Modelfile
ollama run abliteration-walkthrough 'Explain why the sky is blue in one sentence.'
```

To include export in the paused walkthrough, run `./record.sh --with-export --llama-cpp /path/to/llama.cpp`. Quantized runtime behavior can differ from the saved FP32 model and should be evaluated separately for any stronger claims.

## Outputs

`out/` is ignored by Git. It contains the original and edited weights, tensor captures, dataset copies, raw responses, logs, plots (PNG and SVG), `summary.json`, and `REPORT.md`. Do not commit the whole output directory.

The public source contains no personal paths, credentials, private chat export, or private user prompts. The public-source research datasets are included intentionally. Model and dataset licenses remain separate from the MIT license for this code. The Qwen license is copied into the edited model folder.

## Interpretation

The refusal detector is a prefix regular expression. It can miss refusals or flag a harmless apology. The test split is excluded from direction estimation and layer selection, but the dataset pair is not topic-matched. Differences in subject, wording, and length can contribute to the estimated direction. The training filter also uses the same imperfect detector. These limitations are recorded rather than hidden by selecting only successful examples.

KL is measured at the first assistant token on benign prompts. The eight exact-answer questions are a **smoke test**, not MMLU, GSM8K, a safety benchmark, or a broad capability score. Reduced refusal markers do not establish successful harmful assistance. This walkthrough implements one single-direction method, not Heretic's optimization or a healing fine-tune.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```

Tests cover projection geometry, degenerate directions, KL identity, hook cleanup, tied-output-head preservation, hook/weight equivalence, and save/reload equivalence using a small random Qwen2 model. Data tests check complete coverage, disjoint partitions, exclusion of earlier pilot exposure, snapshot integrity and tamper detection. Chat tests check conversation trimming without breaking turn pairs. The full run supplies separate integration evidence on the actual 0.5B model.

## Sources

- [Arditi et al., Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- [Authors' code and results](https://github.com/andyrdt/refusal_direction)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [Harmful behaviors dataset](https://huggingface.co/datasets/mlabonne/harmful_behaviors)
- [Harmless Alpaca dataset](https://huggingface.co/datasets/mlabonne/harmless_alpaca)

See [the methodology notes](docs/methodology.md) for the differences from the simplified teaching story.
