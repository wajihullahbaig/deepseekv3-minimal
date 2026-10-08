"""Compare decoding strategies on a trained checkpoint.

    python run_text_generation.py --model_path checkpoints/best_model.pt --device cuda
    python run_text_generation.py --model_path checkpoints/best_model.pt --prompt "this video is" --prompt "I think"
"""
import argparse
import io
import json
import logging
import os
import sys
import time

import torch
import yaml

from models.deepseek_v3 import DeepSeekV3
from seeding import set_seed
from text_generation import GenerationConfig, TextGenerator
from training.checkpointing import load_tokenizer_from_checkpoint

if sys.platform == "win32":  # make the console UTF-8 safe
    os.system("chcp 65001 > NUL")
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/generation_test.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("TEST RUNNER")

PROMPTS = {
    "yt": ["this video is", "i think the best part", "thanks for sharing", "can someone explain why"],
    "wiki": ["April comes between March and May, making it the", "The Hubble Space Telescope is an",
             "Artificial intelligence makes computers", "Atoms are small particles made of"],
}


def load_model(model_path, model_yaml=None, device="cuda"):
    """Model + tokenizer from a checkpoint; the model config is taken from the checkpoint when present."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if "model_config" in checkpoint:
        model_config = dict(checkpoint["model_config"])
    else:
        with open(model_yaml, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
    model_config["vocab_size"] = state_dict["embedding.weight"].shape[0]
    model = DeepSeekV3(model_config)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    tokenizer = load_tokenizer_from_checkpoint(checkpoint, model_path)
    logger.info(f"Loaded {model_path} (vocab {model_config['vocab_size']}, tokenizer {type(tokenizer).__name__})")
    return model, tokenizer


def decoding_configs(tokenizer, max_length):
    ids = dict(eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return {
        "Greedy": GenerationConfig(max_length=max_length, do_sample=False, use_mtp=False, **ids),
        "Sampling (temp=0.7)": GenerationConfig(max_length=max_length, temperature=0.7, top_k=50, top_p=0.95,
                                                repetition_penalty=1.2, use_mtp=False, **ids),
        "Sampling (temp=1.2)": GenerationConfig(max_length=max_length, temperature=1.2, top_k=50, top_p=0.95,
                                                repetition_penalty=1.2, use_mtp=False, **ids),
        "Beam Search (beams=4)": GenerationConfig(max_length=max_length, do_sample=False, num_beams=4,
                                                  length_penalty=1.0, early_stopping=True, **ids),
        "MTP Speculation (greedy)": GenerationConfig(max_length=max_length, do_sample=False, use_mtp=True,
                                                     mtp_speculation_mode=True, **ids),
        "MTP Speculation (temp=0.7)": GenerationConfig(max_length=max_length, temperature=0.7, top_k=50, top_p=0.95,
                                                       repetition_penalty=1.2, use_mtp=True, mtp_speculation_mode=True, **ids),
    }


def run_generation_comparison(model_path, model_yaml=None, device="cuda", prompts=None, max_length=50):
    model, tokenizer = load_model(model_path, model_yaml, device)
    generator = TextGenerator(model, tokenizer, device)
    prompts = prompts or PROMPTS["yt"]
    for c, prompt in enumerate(prompts):
        logger.info("=" * 80)
        logger.info(f"PROMPT: {prompt}")
        results = {}
        for name, config in decoding_configs(tokenizer, max_length).items():
            t0 = time.time()
            try:
                text, error = generator.generate(prompt, config), None
            except Exception as exc:  # keep comparing the other methods
                text, error = None, str(exc)
                logger.exception(f"{name} failed")
            seconds = time.time() - t0
            results[name] = {"text": text, "error": error, "generation_time_seconds": round(seconds, 3),
                             "config": {k: v for k, v in config.__dict__.items()}}
            logger.info(f"{name} ({seconds:.2f}s): {text if error is None else 'ERROR ' + error}")
        with open(f"logs/prompt_{c}_comparison.json", "w", encoding="utf-8") as f:
            json.dump({"prompt": prompt, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "model_path": model_path,
                       "device": device, "results": results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Comparison saved to logs/prompt_{c}_comparison.json")


def main():
    parser = argparse.ArgumentParser(description="Compare text generation methods on a checkpoint")
    parser.add_argument("--model_path", required=True, help="checkpoint .pt (e.g. checkpoints/best_model.pt)")
    parser.add_argument("--model_yaml", default=None, help="only needed for checkpoints without an embedded model config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", action="append", help="custom prompt (repeatable); default: YouTube prompts")
    parser.add_argument("--prompts", choices=sorted(PROMPTS), default="yt", help="built-in prompt set")
    parser.add_argument("--max_length", type=int, default=50)
    args = parser.parse_args()
    set_seed(args.seed)
    run_generation_comparison(args.model_path, args.model_yaml, args.device,
                              prompts=args.prompt or PROMPTS[args.prompts], max_length=args.max_length)


if __name__ == "__main__":
    main()
