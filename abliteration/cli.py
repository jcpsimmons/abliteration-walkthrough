"""Twelve individually runnable, cached teaching steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time

import torch
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .core import (capture, choose_device, digest, edit_weights, generate,
                   intervention, load_model, mean_kl, next_logits, refused,
                   unit_direction)

ROOT = Path(__file__).resolve().parents[1]
TITLES = ["Download the weights", "Inspect the grids", "Prepare the two piles",
          "Record internal states", "Find the arrows", "Test every arrow",
          "Plot the original states", "Edit the residual writers", "Measure after the edit",
          "Run the held-out check", "Export to GGUF", "Build the evidence report"]
DEPS = {1: [], 2: [1], 3: [1], 4: [3], 5: [4], 6: [5], 7: [6],
        8: [6], 9: [8, 7], 10: [8, 3], 11: [8], 12: [9, 10]}


class Walkthrough:
    def __init__(self, args):
        self.args = args
        self.cfg = json.loads(Path(args.config).read_text())
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.fingerprint = digest(self.cfg)
        self.console = Console(record=True, width=110)

    def read(self, name):
        return json.loads((self.out / name).read_text())

    def save(self, name, value):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(value, indent=2) + "\n")
        temp.replace(path)

    def artifact(self, name):
        return torch.load(self.out / name, map_location="cpu", weights_only=True)

    def model(self, edited=False):
        return load_model(self.out / ("edited" if edited else "base"), self.cfg)

    def progress(self, done, total):
        self.console.print(f"  processed {done:>3}/{total}")

    def step(self, n):
        marker = self.out / f"steps/{n:02}.json"
        for dep in DEPS[n]:
            p = self.out / f"steps/{dep:02}.json"
            if not p.exists() or json.loads(p.read_text())["config_hash"] != self.fingerprint:
                raise RuntimeError(f"Run step {dep} first with this configuration.")
        self.console.rule(f"{n:02} / {TITLES[n - 1]}", style="cyan")
        if marker.exists() and not self.args.refresh:
            meta = json.loads(marker.read_text())
            if meta["config_hash"] == self.fingerprint:
                self.console.print("[yellow]CACHED RESULT: replay of a completed measurement[/yellow]")
                if n in (7, 9):
                    # Redraw saved coordinates so old monochrome logs do not hide the key.
                    self.show_states(n == 9)
                    self.console.print(f"Original measurement took {meta['seconds']:.1f} seconds.")
                else:
                    self.console.print(Text.from_ansi((self.out / f"logs/{n:02}.txt").read_text()))
                return
        # Invalidate downstream markers before a re-run, including failed runs.
        stale = {n}
        for k in range(n + 1, 13):
            if any(d in stale for d in DEPS[k]):
                stale.add(k)
        for k in stale:
            (self.out / f"steps/{k:02}.json").unlink(missing_ok=True)
        self.console.export_text(clear=True)
        start = time.perf_counter()
        getattr(self, f"step_{n}")()
        elapsed = time.perf_counter() - start
        self.console.print(f"[green]Step {n} complete.[/green] Measured wall time: {elapsed:.1f} seconds.")
        (self.out / "logs").mkdir(exist_ok=True)
        (self.out / f"logs/{n:02}.txt").write_text(self.console.export_text(clear=True, styles=True))
        self.save(f"steps/{n:02}.json", {"config_hash": self.fingerprint, "seconds": elapsed,
                                       "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})

    def step_1(self):
        from huggingface_hub import snapshot_download
        snapshot_download(self.cfg["model"], revision=self.cfg["model_revision"], local_dir=self.out / "base",
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md", "LICENSE"])
        table = Table("File", "Bytes")
        for p in sorted((self.out / "base").iterdir()):
            if p.is_file():
                table.add_row(p.name, f"{p.stat().st_size:,}")
        self.console.print(table)
        self.console.print(f"Pinned model revision: {self.cfg['model_revision']}")
        self.console.print(f"Device: {choose_device(self.cfg['device'])}; arithmetic: float32")

    def step_2(self):
        cfg = self.read("base/config.json")
        assert (cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]) == (24, 896, 4864)
        with (self.out / "base/model.safetensors").open("rb") as f:
            header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
        rows = [{"name": k, "shape": v["shape"], "dtype": v["dtype"]}
                for k, v in header.items() if k != "__metadata__"]
        self.save("header.json", rows)
        table = Table("Tensor", "Shape", "Storage")
        for row in rows:
            if any(x in row["name"] for x in ("embed_tokens", "o_proj", "down_proj")):
                table.add_row(row["name"], str(row["shape"]), row["dtype"])
        self.console.print(table)
        self.console.print("24 blocks x 2 residual writers + 1 embedding = 49 matrices.")
        self.console.print("Q/K/V and MLP gate/up read or transform; their weights are unchanged.")
        self.console.print("This model uses SiLU gating, not a ReLU that simply zeros negatives.")
        self.console.print("The base embedding and output head share weights. We will untie the output head.")

    def step_3(self):
        if self.cfg.get("data_mode") == "full-harmful":
            return self.prepare_full_data()
        return self.prepare_sample_data()

    def prepare_full_data(self):
        from .data import prepare
        model, tok = self.model()
        record, audit = prepare(ROOT / "datasets", tok, self.cfg)
        self.save("dataset-audit.json", audit)
        for label, candidates in record["candidates"].items():
            self.console.print(f"Screening all {len(candidates)} {label} training candidates. Text hidden.")
            answers = generate(model, tok, [r["prompt"] for r in candidates], self.cfg, self.progress)
            screened = [{**row, "response": answer, "refusal_detected": refused(answer)}
                        for row, answer in zip(candidates, answers)]
            accepted = [row for row, answer in zip(candidates, answers)
                        if refused(answer) == (label == "harmful")]
            if len(accepted) < 8:
                raise RuntimeError(f"Only {len(accepted)} accepted {label} training rows")
            record["train"][label] = accepted
            record["filter"][label] = {"screened": len(screened), "accepted": len(accepted)}
            self.save(f"private/{label}-training-screen.json", screened)
        self.save("private/prompts.json", record)
        table = Table("Split", "Harmful", "Harmless", "Use")
        for split, use in [("candidates", "all training candidates screened"), ("train", "direction estimation"),
                           ("validation", "layer selection"), ("test", "final held-out check"),
                           ("all", "full-data diagnostic, includes training")]:
            table.add_row(split, *(str(len(record[split][k])) for k in ("harmful", "harmless")), use)
        self.console.print(table)
        self.console.print("Every harmful-source row is assigned; test IDs exclude all prior pilot exposure.")
        self.console.print("Only training is behavior-filtered. Validation and test retain baseline non-refusals.")

    def prepare_sample_data(self):
        from datasets import load_dataset
        model, tok = self.model()
        pools = {}
        for label in ("harmful", "harmless"):
            data = load_dataset(self.cfg[f"{label}_dataset"], revision=self.cfg[f"{label}_revision"])
            unique = {}
            for split in data.values():
                for row in split:
                    prompt = row.get("instruction") or row.get("prompt") or row.get("text") or row.get("goal")
                    if not prompt:
                        raise ValueError("Dataset has no supported prompt column.")
                    if row.get("input"):
                        prompt += "\n" + row["input"]
                    if not 4 < len(tok.encode(prompt)) <= self.cfg["max_prompt_tokens"] - 40:
                        continue
                    key = hashlib.sha256(prompt.strip().encode()).hexdigest()
                    unique[key] = {"id": key[:12], "prompt": prompt.strip()}
            pool = list(unique.values())
            random.Random(self.cfg["seed"]).shuffle(pool)
            pools[label] = pool
        # Remove cross-label duplicates before splitting.
        overlap = {r["id"] for r in pools["harmful"]} & {r["id"] for r in pools["harmless"]}
        pools = {k: [r for r in v if r["id"] not in overlap] for k, v in pools.items()}
        nv, nt, ntrain = (self.cfg[k] for k in ("validation_size", "test_size", "train_size"))
        record = {"validation": {}, "test": {}, "train": {}, "filter": {}}
        for label, pool in pools.items():
            if len(pool) < nv + nt + ntrain:
                raise ValueError("Dataset is too small after deduplication and length filtering.")
            record["validation"][label] = pool[:nv]
            record["test"][label] = pool[nv:nv + nt]
            candidates = pool[nv + nt:nv + nt + ntrain * 3]
            self.console.print(f"Checking {label} training candidates. Text hidden.")
            accepted = []
            screened = []
            for start in range(0, len(candidates), self.cfg["batch_size"]):
                chunk = candidates[start:start + self.cfg["batch_size"]]
                answers = generate(model, tok, [r["prompt"] for r in chunk], self.cfg)
                for row, answer in zip(chunk, answers):
                    flag = refused(answer)
                    screened.append({**row, "response": answer, "refusal_detected": flag})
                    if flag == (label == "harmful"):
                        accepted.append(row)
                self.console.print(f"  screened {len(screened):3}; accepted {min(len(accepted), ntrain):2}/{ntrain}")
                if len(accepted) >= ntrain:
                    break
            if len(accepted) < 8:
                raise RuntimeError(f"Only {len(accepted)} accepted {label} examples; increase train pool or review detector.")
            record["train"][label] = accepted[:ntrain]
            record["filter"][label] = {"screened": len(screened), "accepted": len(accepted[:ntrain])}
            self.save(f"private/{label}-training-screen.json", screened)
        self.save("private/prompts.json", record)
        table = Table("Split", "Refusal-source", "Benign-source", "Use")
        for split, use in [("train", "direction only"), ("validation", "layer selection"), ("test", "final evaluation only")]:
            table.add_row(split, *(str(len(record[split][k])) for k in ("harmful", "harmless")), use)
        self.console.print(table)
        self.console.print("Training is filtered by observed baseline response. Validation/test are unfiltered.")
        self.console.print("These datasets differ in topic and style. Difference of means can include those confounds.")

    def step_4(self):
        model, tok = self.model()
        rows = self.read("private/prompts.json")["train"]
        tensors = {}
        for label, group in rows.items():
            tensors[label] = capture(model, tok, [r["prompt"] for r in group], self.cfg)
            self.console.print(f"{label:8}: {list(tensors[label].shape)} = prompts x blocks x coordinates")
        torch.save(tensors, self.out / "activations.pt")
        self.console.print("One dot is one prompt's last-token state at a chosen block, before final RMSNorm.")

    def step_5(self):
        data = self.artifact("activations.pt")
        arrows = unit_direction(data["harmful"], data["harmless"])
        torch.save(arrows, self.out / "directions.pt")
        self.console.print("For each block: mean(refused training states) - mean(answered training states).")
        self.console.print("Divide by its Euclidean length. Now each arrow has length 1.")
        for i in (0, 11, 23):
            self.console.print(f"Block {i + 1:02}: first 8 = {arrows[i, :8].tolist()}")
            self.console.print(f"          length = {arrows[i].norm():.6f}")

    def step_6(self):
        model, tok = self.model()
        rows = self.read("private/prompts.json")["validation"]
        harmful = [r["prompt"] for r in rows["harmful"]]
        harmless = [r["prompt"] for r in rows["harmless"]]
        base_answers = generate(model, tok, harmful, self.cfg)
        base_logits = next_logits(model, tok, harmless, self.cfg)
        base_count = sum(map(refused, base_answers))
        self.console.print(f"Validation baseline: {base_count}/{len(harmful)} refusal markers.")
        results, raw = [], []
        for i, arrow in enumerate(self.artifact("directions.pt")):
            with intervention(model, arrow):
                answers = generate(model, tok, harmful, self.cfg)
                changed = next_logits(model, tok, harmless, self.cfg)
            count = sum(map(refused, answers))
            kl = max(0., mean_kl(base_logits, changed))
            # Transparent deterministic tradeoff; not a universal optimality claim.
            score = count / len(harmful) + 0.1 * kl
            row = {"block": i + 1, "refusals": count, "n": len(harmful), "kl_nats": kl, "score": score}
            results.append(row)
            raw.append({"block": i + 1, "responses": answers})
            self.console.print(f"Block {i + 1:02}: refusals {count:2}/{len(harmful)}   KL {kl:8.5f}   score {score:7.4f}")
        best = min(results, key=lambda r: (r["score"], r["block"]))
        self.save("sweep.json", {"baseline_refusals": base_count, "rows": results, "selected": best,
                                 "rule": "refusal_fraction + 0.1 * benign_first_token_KL_nats"})
        self.save("private/sweep-responses.json", {"baseline": base_answers, "interventions": raw})
        self.console.print(f"[yellow]Selected block {best['block']} by the stated validation score.[/yellow]")
        self.console.print("KL measures next-token distribution shift, not intelligence or benchmark accuracy.")

    def step_7(self):
        from .plots import before_plots
        before_plots(self.out)
        self.show_states(False)

    def show_states(self, after):
        self.console.print("One mark = one prompt's 896-number internal state, compressed to two coordinates.")
        self.console.print("PC1 and PC2 summarize variation; neither axis is a harmfulness or refusal score.")
        self.console.print(Text.assemble(("X = harmful prompts", "bold #ff6b6b"), "    ",
                                        ("O = harmless prompts", "bold #00ffff")))
        self.terminal_scatter(after)
        if after:
            self.console.print("Same prompts and plotting axes; states re-measured from the saved edited model.")
            self.console.print("Prompt-set labels, not answer grades. Marks can overlap in a terminal cell.")
        else:
            self.console.print("Original model, before the edit. The after chart reuses these same plotting axes.")

    def terminal_scatter(self, after):
        import plotext as plt
        data = self.read("chart-data.json")
        key = "after" if after else "before"
        if key not in data:
            return
        plt.clear_figure()
        plt.plotsize(100, 22)
        plt.theme("dark")
        for label, marker, color in [("harmful", "X", (255, 107, 107)),
                                     ("harmless", "O", (0, 255, 255))]:
            points = data[key][label]
            plt.scatter([p[0] for p in points], [p[1] for p in points], marker=marker, color=color)
        # A dense after plot can put both groups in the same terminal cells.
        # Alternate occlusion priority across coarse cells so O does not hide
        # every X. Repaint real points at their original coordinates, no jitter.
        x0, x1 = data["xlim"]
        y0, y1 = data["ylim"]
        visible_x = [p for p in data[key]["harmful"]
                     if (int((p[0] - x0) / (x1 - x0) * 80)
                         + int((p[1] - y0) / (y1 - y0) * 16)) % 2 == 0]
        if visible_x:
            plt.scatter([p[0] for p in visible_x], [p[1] for p in visible_x],
                        marker="X", color=(255, 107, 107))
        plt.title(f"{key.upper()} / fixed PCA axes / block {data['block']}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.xlim(*data["xlim"])
        plt.ylim(*data["ylim"])
        self.console.print(Text.from_ansi(plt.build()), no_wrap=True, overflow="ignore")

    def step_8(self):
        model, tok = self.model()
        block = self.read("sweep.json")["selected"]["block"]
        arrow = self.artifact("directions.pt")[block - 1]
        probe = ["Explain why the sky looks blue in one sentence."]
        with intervention(model, arrow):
            hooked = next_logits(model, tok, probe, self.cfg)
        start = time.perf_counter()
        records = edit_weights(model, arrow)
        edit_seconds = time.perf_counter() - start
        actual = next_logits(model, tok, probe, self.cfg)
        error = float((hooked - actual).abs().max())
        if not torch.allclose(hooked, actual, atol=0.005, rtol=0.002):
            raise RuntimeError(f"Weight edit does not match hooks: max logits difference {error:.6f}")
        for row in records:
            self.console.print(f"  {row['name']}  residual {row['max_remaining_component']:.2e}")
        target = self.out / "edited"
        model.save_pretrained(target, safe_serialization=True)
        tok.save_pretrained(target)
        shutil.copy2(self.out / "base/LICENSE", target / "LICENSE")
        self.save("edit.json", {"block": block, "matrices": records, "seconds": edit_seconds,
                               "hook_weight_max_logit_error": error, "dtype": "float32",
                               "output_head_untied": True})
        self.console.print(f"49 matrices edited in {edit_seconds:.3f} seconds; hook/weight max logit error {error:.6f}.")
        self.console.print("Saved float32 model. Output head is untied and preserved: the saved model is larger.")

    def step_9(self):
        from .plots import after_plots
        model, tok = self.model(True)
        rows = self.read("private/prompts.json")["train"]
        after = {label: capture(model, tok, [r["prompt"] for r in group], self.cfg)
                 for label, group in rows.items()}
        torch.save(after, self.out / "activations-after.pt")
        after_plots(self.out)
        self.show_states(True)
        self.console.print("Only the selected component is near zero. Other differences between the groups can remain.")

    def demo_prompts(self):
        test = self.read("private/prompts.json")["test"]["harmful"]
        lines = Path(self.args.prompts).read_text().splitlines()
        result = []
        for line in lines:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.startswith("dataset:"):
                row = test[int(line.split(":", 1)[1])]
                result.append(row)
            else:
                result.append({"id": hashlib.sha256(line.encode()).hexdigest()[:12], "prompt": line})
        return result

    def step_10(self):
        dataset = self.read("private/prompts.json")
        rows = dataset["test"]
        checks = json.loads((ROOT / "prompts/benign-checks.json").read_text())
        demos = self.demo_prompts()
        h = [r["prompt"] for r in rows["harmful"]]
        b = [r["prompt"] for r in rows["harmless"]]
        results, raw, logits = {}, {}, {}
        for name, edited in [("before", False), ("after", True)]:
            model, tok = self.model(edited)
            harm_answers = generate(model, tok, h, self.cfg, self.progress)
            benign_answers = generate(model, tok, b, self.cfg)
            check_answers = generate(model, tok, [r["prompt"] for r in checks], self.cfg)
            demo_answers = generate(model, tok, [r["prompt"] for r in demos], self.cfg) if demos else []
            logits[name] = next_logits(model, tok, b, self.cfg)
            top = next_logits(model, tok, ["The best way to cook rice is"], self.cfg)[0].softmax(-1).topk(5)
            results[name] = {
                "harmful_refusals": sum(map(refused, harm_answers)), "harmful_n": len(h),
                "benign_refusals": sum(map(refused, benign_answers)), "benign_n": len(b),
                "smoke_correct": sum(a.strip().strip(".! ").casefold() == r["answer"].casefold() for a, r in zip(check_answers, checks)),
                "smoke_n": len(checks),
                "next_tokens": [{"token": tok.decode([int(i)]), "probability": float(p)} for i, p in zip(top.indices, top.values)],
            }
            raw[name] = {"harmful": harm_answers, "benign": benign_answers,
                         "smoke": check_answers, "demo": demo_answers}
            if "all" in dataset:
                # Reuse identical held-out answers; evaluate every other assigned row once.
                full_raw, full_counts = {}, {}
                for label, test_answers in (("harmful", harm_answers), ("harmless", benign_answers)):
                    by_id = {r["id"]: a for r, a in zip(rows[label], test_answers)}
                    remaining = [r for r in dataset["all"][label] if r["id"] not in by_id]
                    self.console.print(f"Full-data diagnostic, {name}, {label}: {len(remaining)} additional prompts.")
                    extra = generate(model, tok, [r["prompt"] for r in remaining], self.cfg, self.progress)
                    by_id.update({r["id"]: a for r, a in zip(remaining, extra)})
                    full_raw[label] = [{"id": r["id"], "response": by_id[r["id"]]} for r in dataset["all"][label]]
                    full_counts[label] = {"refusals": sum(refused(r["response"]) for r in full_raw[label]),
                                          "n": len(full_raw[label])}
                results[name]["all_data_diagnostic"] = full_counts
                self.save(f"private/all-responses-{name}.json", full_raw)
            del model
            import gc
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        results["benign_first_token_kl_nats"] = max(0., mean_kl(logits["before"], logits["after"]))
        results["demo"] = [{"id": row["id"], "before_refusal": refused(raw["before"]["demo"][i]),
                            "after_refusal": refused(raw["after"]["demo"][i])} for i, row in enumerate(demos)]
        results["detector"] = "prefix regex heuristic; absence of refusal does not establish compliance, usefulness or safety"
        self.save("evaluation.json", results)
        self.save("private/test-responses.json", raw)
        self.save("private/demo-prompts.json", demos)
        table = Table("Held-out check", "Before", "After")
        for key, count, title in [("harmful_refusals", "harmful_n", "Refusal markers, harmful-source"),
                                  ("benign_refusals", "benign_n", "Refusal markers, benign-source"),
                                  ("smoke_correct", "smoke_n", "Exact answers, 8-question smoke test")]:
            table.add_row(title, *(f"{results[x][key]}/{results[x][count]}" for x in ("before", "after")))
        self.console.print(table)
        self.console.print(f"Held-out benign first-token KL: {results['benign_first_token_kl_nats']:.6f} nats")
        for row in results["demo"]:
            self.console.print(f"Prompt {row['id']}: before refusal={row['before_refusal']}, after refusal={row['after_refusal']}; text hidden")
        self.console.print("The 8-question smoke test is not MMLU, GSM8K, or a general capability evaluation.")

    def step_11(self):
        llama = self.args.llama_cpp or os.environ.get("LLAMA_CPP_DIR")
        if not llama:
            raise RuntimeError("Set LLAMA_CPP_DIR to a llama.cpp checkout, or pass --llama-cpp /path/to/llama.cpp.")
        converter = Path(llama) / "convert_hf_to_gguf.py"
        if not converter.is_file():
            raise RuntimeError("convert_hf_to_gguf.py is missing in the selected llama.cpp checkout.")
        target = self.out / "qwen-edited-q8_0.gguf"
        subprocess.run([sys.executable, str(converter), str(self.out / "edited"), "--outfile", str(target), "--outtype", "q8_0"], check=True)
        sha = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
        self.save("export.json", {"file": target.name, "bytes": target.stat().st_size, "sha256": sha,
                                  "format": "GGUF Q8_0", "ollama_tested": False})
        (self.out / "Modelfile").write_text(f"FROM ./{target.name}\nPARAMETER temperature 0\nPARAMETER num_predict 64\nSYSTEM {self.cfg['system_prompt']}\n")
        self.console.print(f"GGUF ready: {target.name}, {target.stat().st_size:,} bytes")
        self.console.print("Optional: cd out && ollama create abliteration-walkthrough -f Modelfile")

    def step_12(self):
        from .plots import evaluation_plots
        evaluation_plots(self.out)
        sweep, edit, ev = (self.read(f"{x}.json") for x in ("sweep", "edit", "evaluation"))
        timings = {str(i): self.read(f"steps/{i:02}.json")["seconds"] for i in range(1, 12) if (self.out / f"steps/{i:02}.json").exists()}
        summary = {"model": self.cfg["model"], "model_revision": self.cfg["model_revision"],
                   "selected_block": sweep["selected"]["block"], "train_filter": self.read("private/prompts.json")["filter"],
                   "evaluation": ev, "edit_seconds": edit["seconds"], "timings_seconds": timings,
                   "config_hash": self.fingerprint, "torch": torch.__version__,
                   "device": choose_device(self.cfg["device"])}
        if (self.out / "dataset-audit.json").exists():
            audit = self.read("dataset-audit.json")
            summary["data"] = {"mode": self.cfg["data_mode"], "split_counts": audit["split_counts"],
                               "sources": audit["sources"], "test_excludes_pilot_exposure": True}
        self.save("summary.json", summary)
        text = (f"# Measured run\n\nModel: {self.cfg['model']}\n\n"
                f"Selected block: {summary['selected_block']} of 24.\n\n"
                f"Held-out refusal markers: {ev['before']['harmful_refusals']}/{ev['before']['harmful_n']} before; "
                f"{ev['after']['harmful_refusals']}/{ev['after']['harmful_n']} after.\n\n"
                f"Benign next-token KL: {ev['benign_first_token_kl_nats']:.6f} nats.\n\n"
                f"Weight arithmetic: {edit['seconds']:.3f} seconds. This excludes download, filtering, search, saving and evaluation.\n\n"
                "Refusal scores are automated prefix heuristics, not a harmful-compliance assessment. "
                "The eight benign questions are only a smoke test. Source prompts are committed in datasets/; generated responses remain in ignored out/private/.\n")
        if "data" in summary:
            counts = summary["data"]["split_counts"]
            text += (f"\nFull-data assignment per source: {counts['candidates']['harmful']} training candidates, "
                     f"{counts['validation']['harmful']} validation, {counts['test']['harmful']} test. "
                     "Test IDs exclude every prompt processed in the earlier pilot.\n\n")
            for label in ("harmful", "harmless"):
                before = ev["before"]["all_data_diagnostic"][label]
                after = ev["after"]["all_data_diagnostic"][label]
                text += (f"All-data {label} refusal markers: {before['refusals']}/{before['n']} before; "
                         f"{after['refusals']}/{after['n']} after. ")
            text += "These diagnostics include training and validation, so they are not held-out scores.\n"
        (self.out / "REPORT.md").write_text(text)
        self.console.print(text, markup=False)

    def reveal(self):
        prompts = self.read("private/demo-prompts.json")
        raw = self.read("private/test-responses.json")
        for i, row in enumerate(prompts):
            self.console.print(row["prompt"], markup=False)
            for label in ("before", "after"):
                self.console.print(label.upper())
                self.console.print(raw[label]["demo"][i], markup=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "run", "all", "present", "doctor", "demo", "chat"])
    parser.add_argument("--step", type=int, choices=range(1, 13))
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--out", default=str(ROOT / "out"))
    parser.add_argument("--prompts", default=str(ROOT / "prompts/demo.txt"))
    parser.add_argument("--refresh", action="store_true", help="Recompute and invalidate dependent cached steps")
    parser.add_argument("--reveal", action="store_true", help="Print local demo text; review off camera first")
    parser.add_argument("--llama-cpp")
    parser.add_argument("--with-export", action="store_true")
    parser.add_argument("--variant", choices=["before", "after"])
    parser.add_argument("--message", help="One-shot chat; omit for an interactive conversation")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Chat generation limit (default: 4096), separate from evaluation")
    parser.add_argument("--context-tokens", type=int, default=8192, help="Chat input plus output budget (default: 8192)")
    args = parser.parse_args()
    app = Walkthrough(args)
    try:
        if args.command == "chat":
            if args.variant is None:
                parser.error("chat requires --variant before or after")
            from .chat import run_chat
            run_chat(args, app.cfg)
        elif args.command == "list":
            for i, title in enumerate(TITLES, 1):
                app.console.print(f"{i:02}  {title}")
        elif args.command == "doctor":
            app.console.print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | device {choose_device(app.cfg['device'])}")
            app.console.print(f"Model files: {'ready' if (app.out / 'base/model.safetensors').exists() else 'run step 1'}")
            app.console.print(f"Output writable: {os.access(app.out, os.W_OK)} | Prompt text display: hidden")
            app.console.print("Complete steps: " + ", ".join(str(i) for i in range(1, 13) if (app.out / f"steps/{i:02}.json").exists()))
        elif args.command == "run":
            if args.step is None:
                parser.error("run requires --step N")
            app.step(args.step)
        elif args.command == "demo":
            if args.reveal:
                app.reveal()
            else:
                ev = app.read("evaluation.json")
                app.console.print(ev["demo"])
                app.console.print("Text hidden. Use --reveal only after an off-camera review.")
        else:
            for n in range(1, 13):
                if n == 11 and not args.with_export:
                    app.console.print("Step 11 is optional. Pass --with-export and --llama-cpp to include it.")
                    continue
                if args.command == "present":
                    answer = input(f"\nENTER: {n:02} {TITLES[n - 1]} | q: quit > ")
                    if answer.strip().lower() == "q":
                        return
                app.step(n)
    except (ValueError, RuntimeError, FileNotFoundError, IndexError, subprocess.CalledProcessError) as error:
        app.console.print(f"Stopped: {error}", style="red", markup=False)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
