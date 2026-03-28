import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import torch, argparse, gc
import sys
from tqdm import tqdm
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
# from sageattention import sageattn
import torch.nn.functional as F
import time

# Add sage3_impl_torch to path for sage3_triton imports
sage3_impl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               'tasks', 'sage3_impl_torch')
if sage3_impl_path not in sys.path:
    sys.path.insert(0, sage3_impl_path)

# Add standalone to path for standalone SageAttention3 imports
standalone_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'standalone')
if standalone_path not in sys.path:
    sys.path.insert(0, standalone_path)

from profiling_utils import AttentionProportionProfiler, ModuleForwardTimer

prompt_path = "videos/testing_prompts.txt"
prompt_path = "videos/open_sora_prompts.txt"


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Inference")
    parser.add_argument("--model",choices=["cogvideox-2b", "cogvideox1.5-5b"], default="cogvideox-2b", help="CogVideoX model")
    parser.add_argument('--compile', action='store_true', help='Compile the model')
    parser.add_argument("-s",'--smoke', action='store_true', help='Run a smoke test')
    parser.add_argument("-p",'--profile', action='store_true', help='Run a profiling test')
    parser.add_argument("-q",'--quick_e2e', action='store_true', help='Run a quick end-to-end test')
    parser.add_argument("-n",'--num_frames', type=int, default=None, help='Number of frames to process')
    parser.add_argument('--proportion', action='store_true', help='Measure attention kernel time proportion in the whole pipeline')
    parser.add_argument(
        "--attention_type",
        type=str,
        default="sdpa",
        choices=[
            "sdpa",
            "sage",
            "sage3",
            "sage3_triton",
            "sage3_standalone",
            "sage3_standalone_mxfp4",
            "sage3_standalone_mxfp4_s1",
            "sage3_standalone_mxfp8_s1",
            "sage3_refactored",
            "sage3_refactored_mxfp4",
            "sage3_refactored_mxfp4_s1",
            "sage3_refactored_mxfp8_s1",
            "fa3",
            "fa3_fp8",
            "escalate",
        ],
        help="Attention type",
    )
    parser.add_argument(
        "-i",
        "--save_frames",
        action="store_true",
        help="Save individual frames as PNG images",
    )
    parser.add_argument("--start", type=int, default=0, help="Starting prompt id of this run.")
    parser.add_argument("--end", type=int, default=None, help="Ending prompt id of this run.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.model == "cogvideox-2b":
        model_path = "/storage/yiliu7/THUDM/CogVideoX-2b"
        model_path = "/mnt/disk1/yiliu7/models/zai-org/CogVideoX-2b"
        num_frames = 49
        torch_dtype = torch.float16
    else:
        model_path = "THUDM/CogVideoX1.5-5B"
        num_frames = 81
        torch_dtype = torch.bfloat16

    if args.attention_type == 'sage':
        from sageattention import sageattn
        F.scaled_dot_product_attention = sageattn
    elif args.attention_type == 'sage3':
        from sageattn3 import sageattn3_blackwell
        F.scaled_dot_product_attention = sageattn3_blackwell
    elif args.attention_type == 'sage3_triton':
        from sage3_triton_wrapper import sage3_triton_sdpa_wrapper
        F.scaled_dot_product_attention = sage3_triton_sdpa_wrapper
    elif "sage3_standalone" in args.attention_type:
        # Set environment variables for optimal performance (can be overridden by user)
        if 'SAGE3_DEBUG' not in os.environ:
            os.environ['SAGE3_DEBUG'] = '0'  # Disable debug by default for performance
        if 'SAGE3_BENCHMARK' not in os.environ:
            os.environ['SAGE3_BENCHMARK'] = '1' if args.proportion else '0'  # Enable benchmarking in proportion mode
        if "mxfp8_s1" in args.attention_type:
            os.environ['SAGE3_QUANT_FORMAT'] = 'mxfp8_s1'
        elif "mxfp4_s1" in args.attention_type:
            os.environ['SAGE3_QUANT_FORMAT'] = 'mxfp4_s1'
        elif "mxfp4" in args.attention_type:
            os.environ['SAGE3_QUANT_FORMAT'] = 'mxfp4'
        from sageattention3_standalone import scaled_dot_product_attention
        print(f"✅ Using SageAttention3 Standalone implementation")
        print(f"   Location: {standalone_path}")
        print(f"   Features: Complete algorithm with QK smoothing, two-level quantization, and Triton kernels")
        print(f"   Environment: SAGE3_DEBUG={os.environ.get('SAGE3_DEBUG')}, SAGE3_BENCHMARK={os.environ.get('SAGE3_BENCHMARK')}")
        print(f"   Quantization Format: {os.environ.get('SAGE3_QUANT_FORMAT', 'nvfp4').upper()}")
        F.scaled_dot_product_attention = scaled_dot_product_attention
    elif "sage3_refactored" in args.attention_type:
        # Refactored sage3 package (composable architecture)
        if 'SAGE3_DEBUG' not in os.environ:
            os.environ['SAGE3_DEBUG'] = '0'
        if 'SAGE3_BENCHMARK' not in os.environ:
            os.environ['SAGE3_BENCHMARK'] = '1' if args.proportion else '0'
        if "sage3_refactored_" in args.attention_type:
            quant_format = args.attention_type.replace("sage3_refactored_", "")
            os.environ['SAGE3_QUANT_FORMAT'] = quant_format
        from sage3 import scaled_dot_product_attention as sage3_refactored_sdpa
        print(f"✅ Using SageAttention3 Refactored (sage3/) implementation")
        print(f"   Location: {standalone_path}/sage3/")
        print(f"   Features: Composable QuantConfig, zero-dispatch P-quant, pre-transform pipeline")
        print(f"   Environment: SAGE3_DEBUG={os.environ.get('SAGE3_DEBUG')}, SAGE3_BENCHMARK={os.environ.get('SAGE3_BENCHMARK')}")
        print(f"   Quantization Format: {os.environ.get('SAGE3_QUANT_FORMAT', 'nvfp4').upper()}")
        F.scaled_dot_product_attention = sage3_refactored_sdpa
    elif args.attention_type == 'fa3':
        from sageattention.fa3_wrapper import fa3
        F.scaled_dot_product_attention = fa3
    elif args.attention_type == 'fa3_fp8':
        from sageattention.fa3_wrapper import fa3_fp8
        F.scaled_dot_product_attention = fa3_fp8
    elif args.attention_type == 'escalate':
        from sageattention3_standalone import scaled_dot_product_attention as sage_sdpa
        import functools

        orig_sdpa = F.scaled_dot_product_attention  # save SDPA reference

        fp4_fn = functools.partial(sage_sdpa, quant_format="mxfp4")
        fp8_fn = functools.partial(sage_sdpa, quant_format="mxfp8_s1")

        # escalate_8_12: 8 SDPA + 12 FP8 + 30 FP4 (worst steps from 50-step profiling)
        worst_steps = [6, 2, 1, 4, 5, 8, 7, 9, 3, 10, 11, 14, 15, 13, 12, 16, 17, 18, 19, 49]
        critical_steps = set(worst_steps[:8])   # 8 worst -> SDPA
        moderate_steps = set(worst_steps[8:20]) # next 12 -> FP8

        calls_per_step = 30  # CogVideoX-2b: 30 attention calls per step

        escalation_call_count = [0]  # use list for mutability in closure
        def escalation_attn(*args_inner, **kwargs_inner):
            step_idx = escalation_call_count[0] // calls_per_step
            escalation_call_count[0] += 1
            if step_idx in critical_steps:
                return orig_sdpa(*args_inner, **kwargs_inner)
            elif step_idx in moderate_steps:
                return fp8_fn(*args_inner, **kwargs_inner)
            return fp4_fn(*args_inner, **kwargs_inner)

        F.scaled_dot_product_attention = escalation_attn
        print(f"✅ Using Escalation attention (escalate_8_12)")
        print(f"   8 SDPA steps: {sorted(critical_steps)}")
        print(f"   12 FP8 steps: {sorted(moderate_steps)}")
        print(f"   30 FP4 steps: remaining")

    # Wrap attention with record_function + NVTX for torch profiler / nsight
    if args.profile:
        _orig_attn = F.scaled_dot_product_attention
        _attn_label = f"attn_{args.attention_type}"
        def _profiled_attn(*args_inner, **kwargs_inner):
            torch.cuda.nvtx.range_push(_attn_label)
            with torch.profiler.record_function(_attn_label):
                result = _orig_attn(*args_inner, **kwargs_inner)
            torch.cuda.nvtx.range_pop()
            return result
        F.scaled_dot_product_attention = _profiled_attn

    # Wrap attention with proportion profiler if requested
    attn_profiler = None
    sage3_sub_timers = {}
    if args.proportion:
        attn_profiler = AttentionProportionProfiler(F.scaled_dot_product_attention)
        F.scaled_dot_product_attention = attn_profiler

        # Wrap sage3 sub-functions for per-step breakdown
        if args.attention_type == 'sage3':
            import sageattn3.api as _sage3_api
            for name in ['preprocess_qkv', 'scale_and_quant_fp4',
                         'scale_and_quant_fp4_permute', 'scale_and_quant_fp4_transpose',
                         'blockscaled_fp4_attn']:
                timer = AttentionProportionProfiler(getattr(_sage3_api, name))
                sage3_sub_timers[name] = timer
                setattr(_sage3_api, name, timer)

    
    
    # extract prefix from prompt_path
    prompt_path_prefix = os.path.basename(prompt_path).split(".")[0]
    # add timestamp for video_dir
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    video_dir = f"videos/{args.model}/{prompt_path_prefix}/{args.attention_type}/{timestamp}"
    if args.quick_e2e:
        prompt_path_prefix = "smoke"
        video_dir = f"videos/{args.model}/{prompt_path_prefix}/quick_e2e_{args.attention_type}"
    os.makedirs(video_dir, exist_ok=True)

    with open(prompt_path, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    end = args.end if args.end is not None else len(prompts)
    selected_prompts = [p.strip() for p in prompts[args.start:end]]

    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)

    if args.compile:
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    
    num_inference_steps = 50
    if args.smoke:
        selected_prompts = ["A dog is running in the park."]
        num_inference_steps = 5
    if args.quick_e2e:
        # img size
        selected_prompts = ["A dog is running in the park."]
        num_frames = 1
        if args.num_frames is not None:
            num_frames = args.num_frames
        # height=128
        # width=128

    if not args.profile and not args.proportion:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    
    if args.profile:
        prompt = selected_prompts[0]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        global_i = f"profile_{args.model}_{args.attention_type}_{timestamp}"
        print(f"Profiling with prompt: {prompt}")
        # warmup
        print("Warming up...")
        for _ in range(3):
            video = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).frames[0]
            del video
            gc.collect()
            torch.cuda.empty_cache()

        print("Warmup completed. Starting profiling...")
        # profile
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
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=6,
                generator=torch.Generator(device="cuda").manual_seed(42),
                #     # img size
                # height=height,
                # width=width,
            ).frames[0]
        print(f"Profiling trace saved to {trace_dir}/ (view with: tensorboard --logdir {trace_dir})")

        # Save individual frames as images (if requested)
        if args.save_frames:
            frames_dir = f"{video_dir}/{global_i}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for frame_idx, frame in enumerate(video):
                filename = f"{frames_dir}/frame_{frame_idx:03d}.png"
                frame.save(filename)
                print(f"saved frame {frame_idx} to {filename}")
            print(f"Saved {len(video)} frames to {frames_dir}/")

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=8)
        del video
        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(0)

    if args.proportion:
        prompt = selected_prompts[0]
        print(f"Measuring attention proportion for [{args.attention_type}] with prompt: {prompt}")

        # Attach transformer forward timer
        transformer_timer = ModuleForwardTimer(pipe.transformer)
        transformer_timer.attach()

        # warmup
        print("Warming up...")
        for _ in range(3):
            attn_profiler.reset()
            transformer_timer.reset()
            for t in sage3_sub_timers.values():
                t.reset()
            video = pipe(
                prompt=prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                guidance_scale=6,
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
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=6,
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
        print(f"  Attn / Transformer      : {attn_ms / transformer_ms * 100:>10.1f} %")
        print(f"  Transformer / Pipeline  : {transformer_ms / total_ms * 100:>10.1f} %")
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


    for local_i, prompt in tqdm(
        enumerate(selected_prompts),
        total=len(selected_prompts),
        desc=f"Generating videos for {args.model} with {args.attention_type} attention",
    ):
        global_i = args.start + local_i
        # Reset escalation counter for each video
        if args.attention_type == 'escalate':
            escalation_call_count[0] = 0
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=6,
            generator=torch.Generator(device="cuda").manual_seed(42),
        ).frames[0]

        # Save individual frames as images (if requested)
        if args.save_frames:
            frames_dir = f"{video_dir}/{global_i}_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for frame_idx, frame in enumerate(video):
                frame.save(f"{frames_dir}/frame_{frame_idx:03d}.png")
            print(f"Saved {len(video)} frames to {frames_dir}/")

        export_to_video(video, f"{video_dir}/{global_i}.mp4", fps=8)
        del video
        gc.collect()
        torch.cuda.empty_cache()
