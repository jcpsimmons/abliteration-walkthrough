"""Interactive chat with the experiment checkpoints, using FP32 arithmetic."""
from pathlib import Path

import torch
from transformers import TextStreamer

from .core import load_model


def chat_messages(tokenizer, history, prompt, system_prompt, budget):
    """Trim oldest complete turns, preserving the system message and newest user input."""
    history = list(history)
    while True:
        messages = [{'role': 'system', 'content': system_prompt}, *history,
                    {'role': 'user', 'content': prompt}]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) <= budget:
            return messages, ids
        if not history:
            raise ValueError(f'Message exceeds the {budget}-token input budget; shorten it or increase --context-tokens.')
        history = history[2:]


def run_chat(args, cfg):
    target = Path(args.out).resolve() / ('base' if args.variant == 'before' else 'edited')
    if not (target / 'model.safetensors').exists():
        raise FileNotFoundError(f'{target}: run walkthrough step {1 if args.variant == "before" else 8} first.')
    if args.max_new_tokens < 1 or args.context_tokens <= args.max_new_tokens:
        raise ValueError('Context size must exceed a positive generation limit.')
    print(f'Loading {args.variant.upper()}: {target}', flush=True)
    model, tok = load_model(target, cfg)
    if args.context_tokens > model.config.max_position_embeddings:
        raise ValueError('Requested context exceeds the model context limit.')
    print(f'{args.variant.upper()} | FP32 | {model.device} | greedy | up to {args.max_new_tokens} new tokens', flush=True)
    print('/reset clears the conversation; /exit quits. Chat is not saved.', flush=True)
    history = []
    while True:
        try:
            prompt = args.message if args.message is not None else input('\nYou> ')
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if prompt.strip() == '/exit':
            return
        if prompt.strip() == '/reset':
            history = []
            print('Conversation cleared.')
            if args.message is not None:
                return
            continue
        if not prompt.strip():
            if args.message is not None:
                return
            continue
        try:
            messages, ids = chat_messages(tok, history, prompt, cfg['system_prompt'],
                                          args.context_tokens - args.max_new_tokens)
        except ValueError as error:
            if args.message is not None:
                raise
            print(error)
            continue
        print(f'{args.variant.capitalize()}> ', end='', flush=True)
        input_ids = torch.tensor([ids], device=model.device)
        try:
            with torch.inference_mode():
                result = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                        do_sample=False, max_new_tokens=args.max_new_tokens,
                                        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                                        temperature=None, top_p=None, top_k=None, use_cache=True,
                                        streamer=TextStreamer(tok, skip_prompt=True, skip_special_tokens=True))
        except KeyboardInterrupt:
            print('\nGeneration interrupted. Conversation unchanged.')
            if args.message is not None:
                return
            continue
        answer = tok.decode(result[0, len(ids):], skip_special_tokens=True)
        history = [*messages[1:], {'role': 'assistant', 'content': answer}]
        if args.message is not None:
            return
