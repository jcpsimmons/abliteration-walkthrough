"""Small, explicit Qwen2 intervention primitives. No remote model code executes."""
from contextlib import contextmanager
from pathlib import Path
import gc
import hashlib
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def choose_device(requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(path, cfg):
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(min(8, torch.get_num_threads()))
    # transformers 4.57.6 misidentifies locally saved Qwen tokenizers as Mistral.
    # Preserve Qwen's original regex; the before/after token IDs must agree.
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                             fix_mistral_regex=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.float32, local_files_only=True,
        trust_remote_code=False, attn_implementation="eager",
    ).eval().to(choose_device(cfg["device"]))
    if model.config.model_type != "qwen2":
        raise ValueError("This teaching implementation supports Qwen2 only.")
    return model, tokenizer


def release(model):
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def encode(tokenizer, prompts, cfg, device):
    strings = [tokenizer.apply_chat_template([
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": p},
    ], tokenize=False, add_generation_prompt=True) for p in prompts]
    # Reject long prompts rather than truncating away the assistant prefix.
    batch = tokenizer(strings, padding=True, return_tensors="pt")
    if batch.input_ids.shape[1] > cfg["max_prompt_tokens"]:
        raise ValueError("Prompt exceeds max_prompt_tokens; raise the limit in config.json.")
    return batch.to(device)


REFUSAL = re.compile(
    r"\b(?:i(?:'m| am) sorry|i apologi[sz]e|i (?:cannot|can't|can not|won't|will not)|"
    r"i(?:'m| am) (?:unable|not able)|as an ai|i (?:must|have to) (?:decline|refuse)|"
    r"i do not (?:provide|support|assist)|i don't (?:provide|support|assist)|"
    r"not (?:appropriate|ethical) (?:to|for)|against (?:my|ethical) guidelines)\b", re.I,
)


def refused(text):
    """Prefix heuristic, NOT a human safety/correctness judgment."""
    return bool(REFUSAL.search(text[:500]))


@torch.inference_mode()
def generate(model, tokenizer, prompts, cfg, progress=None):
    answers = []
    for start in range(0, len(prompts), cfg["batch_size"]):
        batch = encode(tokenizer, prompts[start:start + cfg["batch_size"]], cfg, model.device)
        result = model.generate(
            **batch, do_sample=False, max_new_tokens=cfg["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            use_cache=True, temperature=None, top_p=None, top_k=None,
        )
        answers.extend(tokenizer.batch_decode(result[:, batch.input_ids.shape[1]:], skip_special_tokens=True))
        if progress:
            progress(min(start + cfg["batch_size"], len(prompts)), len(prompts))
    return answers


@torch.inference_mode()
def capture(model, tokenizer, prompts, cfg):
    """[prompt, block, hidden], post-block residual BEFORE final RMSNorm."""
    chunks = []
    for start in range(0, len(prompts), cfg["batch_size"]):
        values = {}
        handles = []
        def hook(index):
            def collect(module, args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                # Left padding ensures -1 is the final assistant-prefix token.
                values[index] = hidden[:, -1, :].detach().float().cpu()
            return collect
        try:
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.register_forward_hook(hook(i)))
            batch = encode(tokenizer, prompts[start:start + cfg["batch_size"]], cfg, model.device)
            model(**batch, use_cache=False, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        chunks.append(torch.stack([values[i] for i in range(len(model.model.layers))], dim=1))
    return torch.cat(chunks)


@torch.inference_mode()
def next_logits(model, tokenizer, prompts, cfg):
    parts = []
    for start in range(0, len(prompts), cfg["batch_size"]):
        batch = encode(tokenizer, prompts[start:start + cfg["batch_size"]], cfg, model.device)
        parts.append(model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float().cpu())
    return torch.cat(parts)


def mean_kl(base_logits, changed_logits):
    logp = base_logits.log_softmax(-1)
    logq = changed_logits.log_softmax(-1)
    return float((logp.exp() * (logp - logq)).sum(-1).mean())


def unit_direction(positive, negative):
    delta = positive.mean(0) - negative.mean(0)
    norm = delta.norm(dim=-1, keepdim=True)
    if (norm < 1e-8).any():
        raise ValueError("Degenerate direction: the two means coincide.")
    return delta / norm


def project(x, direction):
    return x - (x @ direction).unsqueeze(-1) * direction


def writers(model):
    yield "model.embed_tokens", model.model.embed_tokens, 1
    for i, layer in enumerate(model.model.layers):
        yield f"model.layers.{i}.self_attn.o_proj", layer.self_attn.o_proj, 0
        yield f"model.layers.{i}.mlp.down_proj", layer.mlp.down_proj, 0


@contextmanager
def intervention(model, direction):
    """Project each residual writer's output; equivalent to our weight edit."""
    direction = direction.to(model.device, dtype=torch.float32)
    handles = []
    def hook(module, args, output):
        return project(output, direction)
    try:
        for _, module, _ in writers(model):
            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def edit_weights(model, direction):
    """Edit residual writers; preserve the original tied output classifier."""
    if model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr():
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
    model.config.tie_word_embeddings = False
    direction = direction.to(model.device, dtype=torch.float32)
    records = []
    for name, module, hidden_axis in writers(model):
        weight = module.weight
        # Embedding rows are token vectors; linear output vectors are columns.
        if hidden_axis == 1:
            for start in range(0, weight.shape[0], 4096):
                weight[start:start + 4096] -= (weight[start:start + 4096] @ direction).unsqueeze(-1) * direction
            remaining = (weight @ direction).abs().max().item()
        else:
            weight -= direction[:, None] * (direction @ weight)[None, :]
            remaining = (direction @ weight).abs().max().item()
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(project(module.bias, direction))
        records.append({"name": name, "shape": list(weight.shape), "max_remaining_component": remaining})
    return records
