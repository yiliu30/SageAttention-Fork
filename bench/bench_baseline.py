from torch.nn.functional import scaled_dot_product_attention as sdpa
import torch
# from flash_attn.utils.benchmark import benchmark_forward

import torch
import torch.utils.benchmark as benchmark


def benchmark_forward(
    fn, *inputs, repeats=10, desc="", verbose=True, amp=False, amp_dtype=torch.float16, **kwinputs
):
    """Use Pytorch Benchmark on the forward pass of an arbitrary function."""
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m

import argparse

parser = argparse.ArgumentParser(description='Benchmark Baseline')
parser.add_argument('--method', type=str, default='fa2', choices=['fa2', 'torch', 'xformers', "sage3"])
parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
parser.add_argument('--num_heads', type=int, default=32, help='Number of heads')
parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
args = parser.parse_args()

head = args.num_heads
batch = args.batch_size
headdim = args.head_dim

assert args.method in ['fa2', 'torch', 'xformers', "sage3"], "Unsupported method"

# only one of the following is True
torch.backends.cuda.enable_flash_sdp(args.method == 'fa2')   # use FA2
torch.backends.cuda.enable_math_sdp(args.method == 'torch')  # use Torch
torch.backends.cuda.enable_mem_efficient_sdp(args.method == 'xformers')  # use xformers

if args.method == "sage3":
    from sageattn3 import sageattn3_blackwell
    sdpa = sageattn3_blackwell
    # torch.nn.functional.scaled_dot_product_attention = sdpa

print(f"Baseline: {args.method}")
print(f"batch: {batch}, head: {head}, headdim: {headdim}")

is_causal = False
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    for i in range(5): sdpa(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')

is_causal = True
print(f"is_causal: {is_causal}")
for seq_len in {1024, 2048, 4096, 8192, 16384, 32768}:
    flops = 4 * head * batch * headdim * seq_len * seq_len // (2 if is_causal else 1)
    q = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    k = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    v = torch.randn(batch, head, seq_len, headdim, dtype=torch.float16, device="cuda")
    for i in range(5): sdpa(q, k, v, is_causal=is_causal)
    torch.cuda.synchronize()
    _, time = benchmark_forward(sdpa, q, k, v, is_causal=is_causal, repeats=100, verbose=False, desc='Triton')
    print(f'{seq_len} flops:{flops/time.mean*1e-12}')