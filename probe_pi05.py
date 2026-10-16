"""
probe_pi05.py — poke at pi0.5 on single LIBERO frames.

Usage (inside an salloc GPU session, after `source vla_env.sh`):
    cd $WS/lerobot
    python probe_pi05.py --episode 0 --offset 0
    python probe_pi05.py --episode 0 --offset 60   # a later moment in the same demo

Results are printed and saved to $WS/lerobot/outputs/probe/.
All "diff" numbers are mean absolute differences in action units.
Compare them against `seed_variation` (the noise floor): a perturbation whose
diff is about the same as seed variation had essentially no effect.
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05 import PI05Policy

WS = Path(os.environ.get("WS", "/vol/gpudata/tq422-vla"))
MODEL_DIR = WS / "models" / "pi05_libero_finetuned_v044"
OUT_DIR = WS / "lerobot" / "outputs" / "probe"
CAM_MAIN = "observation.images.image"    # third-person (agentview) camera
CAM_WRIST = "observation.images.image2"  # wrist camera


def describe(title, d):
    print(f"\n--- {title} ---")
    for k, v in d.items():
        if torch.is_tensor(v):
            print(f"{k:45s} shape={tuple(v.shape)} dtype={v.dtype}")
        else:
            print(f"{k:45s} {type(v).__name__}: {str(v)[:80]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--offset", type=int, default=0, help="frame index within the episode")
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # ---------------- Load policy + processors from the local checkpoint ----------------
    policy = PI05Policy.from_pretrained(str(MODEL_DIR)).to(device).eval()
    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        str(MODEL_DIR),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    cfg = policy.config
    for field in ["chunk_size", "n_action_steps", "num_inference_steps", "max_state_dim", "max_action_dim"]:
        print(f"config.{field} = {getattr(cfg, field, 'n/a (field not found)')}")

    # ---------------- Load one frame from the LIBERO demonstrations ----------------
    # Downloads lerobot/libero (~1.9 GB) into $HF_LEROBOT_HOME on first run.
    ds = LeRobotDataset("lerobot/libero", video_backend="pyav")
    start = int(ds.meta.episodes["dataset_from_index"][args.episode])
    end = int(ds.meta.episodes["dataset_to_index"][args.episode])
    idx = start + args.offset
    assert idx < end, f"offset too large: episode has {end - start} frames"
    frame = dict(ds[idx])
    assert CAM_MAIN in frame and CAM_WRIST in frame, [k for k in frame if "image" in k]

    describe("raw dataset frame", frame)
    print("\nTask instruction:", frame["task"])

    batch = preprocess(dict(frame))
    describe("after preprocessing (what the model actually sees)", batch)

    # pi0.5 writes the discretized robot state into the text prompt: decode it to see.
    tok_key = next((k for k in batch if "tokens" in k and "mask" not in k), None)
    if tok_key is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        print("\nDecoded prompt:\n", tok.decode(batch[tok_key][0], skip_special_tokens=True))

    # ---------------- Helpers ----------------
    def act(fr, seed=0):
        """Full predicted action chunk (unnormalized), shape (chunk, action_dim).
        Uses predict_action_chunk, NOT select_action: select_action keeps an internal
        queue of actions and would return stale actions from the previous call."""
        torch.manual_seed(seed)
        b = preprocess(dict(fr))
        with torch.inference_mode():
            a = policy.predict_action_chunk(b)
            a = postprocess(a)
        return a[0].float().cpu()

    def diff(a, b):
        return (a - b).abs().mean().item()

    def timed(fr):
        torch.cuda.synchronize()
        t0 = time.time()
        a = act(fr)
        torch.cuda.synchronize()
        return a, time.time() - t0

    results = {"episode": args.episode, "frame_index": idx, "task": frame["task"]}

    # 1) Baseline prediction + latency
    act(frame)  # warm-up (first call includes CUDA setup)
    base, results["latency_s"] = timed(frame)
    print(f"\nAction chunk shape: {tuple(base.shape)}   latency: {results['latency_s']:.3f}s")
    print("First 3 predicted actions:\n", base[:3])

    # 2) Compare against what the demonstrator actually did next
    T = min(base.shape[0], end - idx)
    rows = ds.hf_dataset.select(range(idx, idx + T))["action"]
    gt = torch.stack([torch.as_tensor(r) for r in rows]).float()
    err = (base[:T, : gt.shape[1]] - gt).abs().mean(0)
    results["mae_vs_demo_per_dim"] = err.tolist()
    print(f"\nMAE vs demonstration over next {T} steps, per action dim:\n", err)

    # 3) Stochasticity: flow matching starts from random noise -> this is the noise floor
    results["seed_variation"] = [diff(base, act(frame, seed=s)) for s in (1, 2, 3)]

    # 4) Language perturbations (same seed, so any difference comes from the prompt)
    # Edit "other_task" to an instruction that fits objects in your chosen scene.
    prompts = {
        "paraphrase": f"please {frame['task']}",
        "other_task": "pick up the bowl and place it on the plate",
        "nonsense": "sing a song about the moon",
        "empty": "",
    }
    results["language"] = {}
    for name, text in prompts.items():
        fr = dict(frame)
        fr["task"] = text
        results["language"][name] = diff(base, act(fr))

    # 5) Vision perturbations
    def with_change(**changes):
        fr = dict(frame)
        fr.update(changes)
        return fr

    vision = {
        "black_main_cam": with_change(**{CAM_MAIN: torch.zeros_like(frame[CAM_MAIN])}),
        "black_wrist_cam": with_change(**{CAM_WRIST: torch.zeros_like(frame[CAM_WRIST])}),
        "noisy_main_cam": with_change(
            **{CAM_MAIN: (frame[CAM_MAIN] + 0.2 * torch.randn_like(frame[CAM_MAIN])).clamp(0, 1)}
        ),
        "swapped_cams": with_change(**{CAM_MAIN: frame[CAM_WRIST], CAM_WRIST: frame[CAM_MAIN]}),
    }
    results["vision"] = {name: diff(base, act(fr)) for name, fr in vision.items()}

    # 6) Number of flow-matching denoising steps.
    # If every diff comes out 0.0, the model reads this value only at load time:
    # look in modeling_pi05.py for where it is used.
    if hasattr(cfg, "num_inference_steps"):
        default = cfg.num_inference_steps
        results["denoise_steps"] = {}
        for n in (1, 2, 5, 10, 20):
            cfg.num_inference_steps = n
            a, lat = timed(frame)
            results["denoise_steps"][n] = {"diff_vs_default": diff(base, a), "latency_s": lat}
        cfg.num_inference_steps = default

    # ---------------- Save ----------------
    stem = f"ep{args.episode}_f{args.offset}"
    torch.save({"pred_chunk": base, "demo_chunk": gt}, OUT_DIR / f"{stem}_chunks.pt")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(results, indent=2))
    print("\n" + json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_DIR / stem}.json")


if __name__ == "__main__":
    main()