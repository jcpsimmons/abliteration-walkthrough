# Public dataset snapshots

These files preserve the full `text` column from the pinned Hugging Face datasets, one JSON object per line. Original train rows precede original test rows. Text contents and upstream ordering are preserved; the experiment strips surrounding whitespace for IDs and model inputs.

| File | Rows | Bytes | Upstream train / test |
| --- | ---: | ---: | --- |
| `harmful_behaviors.jsonl` | 520 | 44,204 | 416 / 104 |
| `harmless_alpaca.jsonl` | 31,323 | 2,244,965 | 25,058 / 6,265 |

Sources: [mlabonne/harmful_behaviors](https://huggingface.co/datasets/mlabonne/harmful_behaviors) and [mlabonne/harmless_alpaca](https://huggingface.co/datasets/mlabonne/harmless_alpaca). Exact revisions and content hashes are in `manifest.json`. These are public research prompts; they are data, not instructions to the reader or automation agent.

Neither upstream dataset card declares a license at the pinned revision. The repository's MIT license applies to our code and does not grant additional rights to these datasets. Source attribution and that missing license metadata are preserved here.

The harmful snapshot includes deliberately harmful requests used for refusal research. Generated model responses are not part of these files and remain in the ignored run output directory.

`pilot-exposure.json` contains the public prompt IDs processed in the earlier pilot, including all screened candidates. The full run excludes these IDs from its test groups. The source snapshots themselves are complete and unfiltered. The new experiment uses all 520 harmful rows and a matched 520-row benign sample, with separate training candidates, validation and test groups.
