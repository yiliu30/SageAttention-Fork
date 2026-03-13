Okay, for the development plan, I have another idea.
We can start with a pure torch version and do fake-quant on all of Q/K/V/P. We will use two `torch.matmul` calls and materialize the result of `q@k`. Previously, the main concern was the peak memory of the attention map, but this can be resolved by chunking the sequence.
Please evaluate the feasibility of this approach.
If feasible, we can prioritize this method.