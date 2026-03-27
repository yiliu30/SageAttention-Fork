import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, argparse, gc
import sys
from tqdm import tqdm
from diffusers import HunyuanVideo15Pipeline
from diffusers.utils import export_to_video
import torch.nn.functional as F
import time
from PIL import Image
import numpy as np

from profiling_utils import AttentionProportionProfiler, ModuleForwardTimer


def _save_frame(frame, filename):
    """Save a video frame (PIL Image or numpy array) as PNG."""
    if isinstance(frame, np.ndarray):
        # HunyuanVideo15Pipeline returns float32 numpy arrays in [0, 1]
        if frame.dtype in (np.float32, np.float64):
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(frame).save(filename)
    else:
        frame.save(filename)

prompt_path = "videos/open_sora_prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo 1.5 Inference")
    parser.add_argument(
        "--model_path",
        default="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        help="HunyuanVideo 1.5 model path (HF Hub ID or local path)",
    )
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument("-s", '--smoke', action='store_true', help='Run a smoke test')
    parser.add_argument("-p", '--profile', action='store_true', help='Run a profiling test')
    parser.add_argument("-q", '--quick_e2e', action='store_true', help='Run a quick end-to-end test')
    parser.add_argument("-n", '--num_frames', type=int, default=None, help='Number of frames to process')
    parser.add_argument('--proportion', action='store_true',
                        help='Measure attention kernel time proportion in the whole pipeline')
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sdpa",
        choices=["sdpa", "sage3"],
        help="Attention type",
    )
    parser.add_argument(
        "-i", "--save_frames", action="store_true",
        help="Save individual frames as PNG images",
    )
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=None, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args


def _patch_dispatch_attention(sage3_fn):
    """Monkey-patch diffusers dispatch_attention_fn to use sage3 attention.

    In diffusers >= 0.37, HunyuanVideo15 uses dispatch_attention_fn instead of
    F.scaled_dot_product_attention. The dispatch function receives tensors in
    [B, N, H, D] format. We permute to [B, H, N, D] for sage3, then back.

    HunyuanVideo15AttnProcessor2_0 always constructs an attention_mask (padding
    mask for text tokens). For profiling we ignore the mask and use sage3 for
    all attention calls — the mask only affects padding tokens and doesn't
    meaningfully change video quality.
    """
    import diffusers.models.attention_dispatch as _attn_dispatch_module
    _orig_dispatch = _attn_dispatch_module.dispatch_attention_fn

    def _sage3_dispatch(query, key, value, attn_mask=None, dropout_p=0.0,
                        is_causal=False, scale=None, enable_gqa=False,
                        attention_kwargs=None, *, backend=None, parallel_config=None):
        # Input: [B, N, H, D] -> sage3 expects [B, H, N, D]
        q = query.permute(0, 2, 1, 3).contiguous()
        k = key.permute(0, 2, 1, 3).contiguous()
        v = value.permute(0, 2, 1, 3).contiguous()
        out = sage3_fn(q, k, v, is_causal=is_causal)
        # Output: [B, H, N, D] -> back to [B, N, H, D]
        return out.permute(0, 2, 1, 3)

    _attn_dispatch_module.dispatch_attention_fn = _sage3_dispatch
    # Also patch the imported reference in both transformer modules
    import diffusers.models.transformers.transformer_hunyuan_video as _hvt_module
    _hvt_module.dispatch_attention_fn = _sage3_dispatch
    import diffusers.models.transformers.transformer_hunyuan_video15 as _hvt15_module
    _hvt15_module.dispatch_attention_fn = _sage3_dispatch
    return _orig_dispatch, _sage3_dispatch


if __name__ == "__main__":
    args = parse_args()

    model_path = args.model_path
    num_frames = 121
    height = 320
    width = 576
    torch_dtype = torch.bfloat16
    num_inference_steps = 50
    fps = 24

    # --- Attention monkey-patching ---
    if args.attention_type == 'sage3':
        from sageattn3 import sageattn3_blackwell
        _orig_dispatch, _sage3_dispatch = _patch_dispatch_attention(sageattn3_blackwell)
        print(f"✅ Using SageAttention3 (CUTE kernel) via dispatch_attention_fn patch")

    # Wrap attention with proportion profiler if requested
    attn_profiler = None
    sage3_sub_timers = {}
    if args.proportion:
        if args.attention_type == 'sage3':
            import diffusers.models.attention_dispatch as _attn_dispatch_module
            import diffusers.models.transformers.transformer_hunyuan_video as _hvt_module
            import diffusers.models.transformers.transformer_hunyuan_video15 as _hvt15_module
            _current_dispatch = _attn_dispatch_module.dispatch_attention_fn
            attn_profiler = AttentionProportionProfiler(_current_dispatch)
            _attn_dispatch_module.dispatch_attention_fn = attn_profiler
            _hvt_module.dispatch_attention_fn = attn_profiler
            _hvt15_module.dispatch_attention_fn = attn_profiler

            # Wrap sage3 sub-functions for per-step breakdown
            import sageattn3.api as _sage3_api
            for name in ['preprocess_qkv', 'scale_and_quant_fp4',
                         'scale_and_quant_fp4_permute', 'scale_and_quant_fp4_transpose',
                         'blockscaled_fp4_attn']:
                timer = AttentionProportionProfiler(getattr(_sage3_api, name))
                sage3_sub_timers[name] = timer
                setattr(_sage3_api, name, timer)
        else:
            # For SDPA: wrap dispatch_attention_fn
            import diffusers.models.attention_dispatch as _attn_dispatch_module
            import diffusers.models.transformers.transformer_hunyuan_video as _hvt_module
            import diffusers.models.transformers.transformer_hunyuan_video15 as _hvt15_module
            attn_profiler = AttentionProportionProfiler(_attn_dispatch_module.dispatch_attention_fn)
            _attn_dispatch_module.dispatch_attention_fn = attn_profiler
            _hvt_module.dispatch_attention_fn = attn_profiler
            _hvt15_module.dispatch_attention_fn = attn_profiler

    # Wrap attention with record_function + NVTX for torch profiler / nsight
    if args.profile:
        import diffusers.models.attention_dispatch as _attn_dispatch_module
        import diffusers.models.transformers.transformer_hunyuan_video as _hvt_module
        import diffusers.models.transformers.transformer_hunyuan_video15 as _hvt15_module
        _current_attn = _attn_dispatch_module.dispatch_attention_fn
        _attn_label = f"attn_{args.attention_type}"
        def _profiled_attn(*args_inner, **kwargs_inner):
            torch.cuda.nvtx.range_push(_attn_label)
            with torch.profiler.record_function(_attn_label):
                result = _current_attn(*args_inner, **kwargs_inner)
            torch.cuda.nvtx.range_pop()
            return result
        _attn_dispatch_module.dispatch_attention_fn = _profiled_attn
        _hvt_module.dispatch_attention_fn = _profiled_attn
        _hvt15_module.dispatch_attention_fn = _profiled_attn

    # --- Directory setup ---
    prompt_path_prefix = os.path.basename(prompt_path).split(".")[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_dir = f"videos/hunyuan15-480p/{prompt_path_prefix}/{args.attention_type}/{timestamp}"
    if args.quick_e2e:
        prompt_path_prefix = "smoke"
        video_dir = f"videos/hunyuan15-480p/{prompt_path_prefix}/quick_e2e_{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    # --- Prompts ---
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    end = args.end if args.end is not None else len(prompts)
    selected_prompts = [p.strip() for p in prompts[args.start:end]]

    # --- Pipeline loading ---
    pipe = HunyuanVideo15Pipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    if args.smoke:
        selected_prompts = ["A dog is running in the park."]
        num_inference_steps = 5
        num_frames = 17  # Reduce for memory
        height = 320
        width = 512
    if args.quick_e2e:
        selected_prompts = ["A dog is running in the park."]
        num_frames = 1
        height = 320
        width = 512
        if args.num_frames is not None:
            num_frames = args.num_frames

    if not args.profile and not args.proportion:
        pipe.enable_model_cpu_offload()
    else:
        # For profiling: use model_cpu_offload (sequential offload) since the
        # full pipeline doesn't fit in 32GB VRAM. The transformer is moved to
        # GPU on demand, so transformer-level timing is still accurate.
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # --- Profile mode ---
    if args.profile:
        prompt = selected_prompts[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        global_i = f"profile_hunyuan15_{args.attention_type}_{timestamp}"
        print(f"Profiling with prompt: {prompt}")
        # warmup
        print("Warming up...")
        for _ in range(3):
            video = pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            del video
            gc.collect()
            torch.cuda.empty_cache()

        print("Warmup completed. Starting profiling...")
        trace_dir = f"{video_dir}/{global_i}_tb_trace_{timestamp}_{num_inference_steps}steps"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir, use_gzip=True),
        ) as prof:
            video = pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
        print(f"Profiling trace saved to {trace_dir}/ (view with: tensorboard --logdir {trace_dir})")

        if args.save_frames:
            frames_dir = f"{video_dir}/{global_i}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for frame_idx, frame in enumerate(video):
                filename = f"{frames_dir}/frame_{frame_idx:03d}.png"
                _save_frame(frame, filename)
                print(f"saved frame {frame_idx} to {filename}")
            print(f"Saved {len(video)} frames to {frames_dir}/")

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=fps)
        del video
        gc.collect()
        torch.cuda.empty_cache()

    # --- Proportion mode ---
    if args.proportion:
        prompt = selected_prompts[0]
        print(f"Measuring attention proportion for [{args.attention_type}] with prompt: {prompt}")

        transformer_timer = ModuleForwardTimer(pipe.transformer)
        transformer_timer.attach()

        # warmup (1 iteration to avoid OOM with CPU offload)
        print("Warming up...")
        attn_profiler.reset()
        transformer_timer.reset()
        for t in sage3_sub_timers.values():
            t.reset()
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        del video
        gc.collect()
        torch.cuda.empty_cache()

        print("Warmup completed. Measuring attention proportion...")
        attn_profiler.reset()
        transformer_timer.reset()
        for t in sage3_sub_timers.values():
            t.reset()
        pipe_start = torch.cuda.Event(enable_timing=True)
        pipe_end = torch.cuda.Event(enable_timing=True)

        pipe_start.record()
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]
        pipe_end.record()
        torch.cuda.synchronize()

        total_ms = pipe_start.elapsed_time(pipe_end)
        transformer_ms = transformer_timer.total_ms()
        attn_ms = attn_profiler.total_attn_ms()
        print(f"\n{'='*60}")
        print(f"Attention Proportion Report ({args.attention_type})")
        print(f"{'='*60}")
        print(f"  Total pipeline GPU time : {total_ms:>10.1f} ms")
        print(f"  Transformer GPU time    : {transformer_ms:>10.1f} ms")
        print(f"  Attention kernel time   : {attn_ms:>10.1f} ms")
        print(f"  ---- Proportions ----")
        print(f"  Attn / Pipeline         : {attn_ms / total_ms * 100:>10.1f} %")
        if transformer_ms > 0:
            print(f"  Attn / Transformer      : {attn_ms / transformer_ms * 100:>10.1f} %")
            print(f"  Transformer / Pipeline  : {transformer_ms / total_ms * 100:>10.1f} %")
        else:
            print(f"  Attn / Transformer      : N/A (CPU offload active)")
            print(f"  Transformer / Pipeline  : N/A (CPU offload active)")
        print(f"  ---- Counts ----")
        print(f"  Attention calls         : {attn_profiler.call_count():>10d}")
        print(f"  Transformer fwd calls   : {transformer_timer.call_count():>10d}")
        print(f"  Avg attn per call       : {attn_ms / max(attn_profiler.call_count(), 1):>10.2f} ms")
        if sage3_sub_timers:
            print(f"  ---- Sage3 Sub-step Breakdown ----")
            for name, timer in sage3_sub_timers.items():
                sub_ms = timer.total_attn_ms()
                print(f"  {name:30s}: {sub_ms:>8.1f} ms ({sub_ms / attn_ms * 100:>5.1f}% of attn)")
        print(f"{'='*60}\n")

        transformer_timer.detach()
        del video
        gc.collect()
        torch.cuda.empty_cache()

    # --- Normal generation loop ---
    for local_i, prompt in tqdm(
        enumerate(selected_prompts),
        total=len(selected_prompts),
        desc=f"Generating videos for hunyuan15-480p with {args.attention_type} attention",
    ):
        global_i = args.start + local_i
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        if args.save_frames:
            frames_dir = f"{video_dir}/{global_i}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for frame_idx, frame in enumerate(video):
                filename = f"{frames_dir}/frame_{frame_idx:03d}.png"
                _save_frame(frame, filename)
            print(f"Saved {len(video)} frames to {frames_dir}/")

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=fps)
        del video
        gc.collect()
        torch.cuda.empty_cache()
