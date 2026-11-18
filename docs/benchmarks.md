# Benchmarks: Sharplight vs llama.cpp vs vLLM

Baseline measured on 2026-11-13. Measurement only: nothing under `src/` or `tests/` was modified. All scripts and raw results
(JSON/JSONL/text, GPU snapshots, server logs) live in [`bench/`](../bench) and [`bench/results/`](../bench/results).

## 1. Setup and conditions

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120, 300 W power limit, driver 610.57.04). Boost clock up to 3090 MHz, about 2340 MHz observed during runs |
| CPU / RAM | Ryzen 9 9950X, 91 GB (typically 55-70 GB used by the desktop session, so little free RAM) |
| OS | Fedora 44, Linux 7.1.8 |
| Sharplight | working tree of 2026-11-13 (no git metadata available), `dotnet build -c Release`, .NET SDK 11.0.100, Vulkan provider with Slang kernels; benchmarked binaries: `samples/Sharplight.Cli` (`--bench`), `samples/Sharplight.Server` (now `sharplight serve`) |
| llama.cpp | b11177 (commit 1ab7e5ad2), builds `runtime.llama-b11177-cuda13` and `runtime.llama-b11177-vulkan` (Vulkan device 0 = the NVIDIA GPU; the RADV iGPU is ignored by default), `-ngl 99 -fa 1/on` |
| vLLM | 0.33.0 in `~/.venvs/vllm` (uv, Python 3.12.13, torch 2.14.0+cu130, flash-attn backend, CUDA graphs on). `VLLM_USE_FLASHINFER_SAMPLER=0` because no CUDA toolkit matching the wheels is installed (FlashInfer JIT needs nvcc). Prefix caching off |

**Shared GPU.** The GPU is shared with the user's desktop (kwin, VS Code, Chrome, Rider, a Python process) and three idle
`sharplight serve` processes that were already running (about 10-13 GB of VRAM in use before every run, 0-2 % utilisation when idle).
`nvidia-smi` was sampled before every measurement (`bench/results/*.gpu.txt`, and a line per run in `single_sharplight_*.txt`); no other
heavy GPU user was seen, but samples taken right after the previous process exited show residual utilisation, and another agent
was running tests in the same period, so a few outliers (marked by wide min..max ranges, mainly the small-model prefill rows) may come
from interference. Every cell therefore reports the **median over repetitions with (min..max)**:

* llama-bench: 5 repetitions (`-r 5`), median of `samples_ts`.
* Sharplight CLI `--bench N`: 3 process runs; each run is a warm prefill of N tokens (best of 4 inside the run) and 64 decode steps (median step); table shows the median over the 3 runs.
* vLLM: offline `LLM` API, median of 5 repetitions per point (`bench/vllm_single.py`).
* Serving runs: one run per cell (3-8 requests per worker, 8-192 requests); treat differences below about 10 % as noise.

**Models and quantisations** (same GGUF for Sharplight and llama.cpp; vLLM needs HF checkpoints and cannot load these GGUFs, so its
quantisation differs):

| key | Sharplight / llama.cpp | vLLM |
|---|---|---|
| qwen27b-q8 | Qwen3.8-27B Q8_0 GGUF (29 GB; Sharplight also as `.slm`) | `Qwen/Qwen3.8-27B-FP8` (download, 31 GB) |
| qwen3-0.6b-q8 | Qwen3-0.6B Q8_0 | `~/models/hf/Qwen3-0.6B` bf16 |
| llama3.2-1b-q8 | Llama-3.2-1B-Instruct Q8_0 | `~/models/hf/Llama-3.2-1B-Instruct` bf16 |
| gemma3-1b-q8 | gemma-3-1b-it Q8_0 | `~/models/hf/gemma-3-1b-it` bf16 |
| mistral7b-q4km | Mistral-7B-Instruct-v0.3 Q4_K_M | `mistralai/Mistral-7B-Instruct-v0.3` bf16 (download, 14.5 GB) |
| olmoe-q4km | OLMoE-1B-7B-0924-Instruct Q4_K_M | `~/models/hf/OLMoE-1B-7B-0924-Instruct` bf16 (max 4096 positions, so its "deep" point is depth 3800) |
| nemotron30b-q4km | NVIDIA-Nemotron-3-Nano-30B-A3B Q4_K_M | FP8 checkpoint downloaded, **could not run** (FP8 KV cache forces FlashInfer, whose JIT fails without a matching nvcc) |

The vLLM columns are bf16 (or FP8 for the 27B) and therefore move 2-3.5x more weight bytes than the Q4_K_M models: on Mistral/OLMoE
they are **not comparable for decode**, only for the batching behaviour. For Qwen3.8-27B, FP8 (about 29 GB) is the closest match to Q8_0 (29 GB).
The downloaded HF checkpoints are in `~/models/hf-vllm/`.

## 2. Single-stream results

Units: tokens/s, higher is better. `SL/x` = Sharplight divided by engine x (below 1.00x means Sharplight is slower).
Rows: `pp512` / `pp2048` = prompt processing of 512 / 2048 tokens from an empty context; `tg128 @d0` = decode from an (almost) empty context
(Sharplight: depth 16 after a 16-token prefill; vLLM: 16 tokens; llama-bench: depth 0); `tg128 @d4096` = decode at a context depth of 4096
(OLMoE/vLLM: 3800).

| model | metric | Sharplight med (min..max) | llama.cpp CUDA med (min..max) | llama.cpp Vulkan med (min..max) | vLLM med | SL/CUDA | SL/Vulkan | SL/vLLM |
|---|---|---|---|---|---|---|---|---|
| gemma3-1b-q8 | pp512 | 45642 (45634..45679) | 45353 (31416..45456) | 46408 (45920..46509) | 35259 | 1.01x | 0.98x | 1.29x |
| gemma3-1b-q8 | pp2048 | 41899 (41844..41902) | 46479 (42244..46680) | 46661 (45842..47104) | 65038 | 0.90x | 0.90x | 0.64x |
| gemma3-1b-q8 | tg128 @d0 | 554 (547..557) | 538 (535..539) | 500 (494..501) | 205 | 1.03x | 1.11x | 2.70x |
| gemma3-1b-q8 | tg128 @d4096 | 505 (488..523) | 524 (522..526) | 488 (486..492) | 297 | 0.96x | 1.03x | 1.70x |
| llama3.2-1b-q8 | pp512 | 53317 (50152..53531) | 58918 (43826..59213) | 50868 (50363..51035) | 71161 | 0.90x | 1.05x | 0.75x |
| llama3.2-1b-q8 | pp2048 | 48470 (46673..49116) | 58092 (55435..59754) | 49845 (49024..49986) | 100434 | 0.83x | 0.97x | 0.48x |
| llama3.2-1b-q8 | tg128 @d0 | 752 (698..753) | 739 (738..740) | 703 (689..707) | 459 | 1.02x | 1.07x | 1.64x |
| llama3.2-1b-q8 | tg128 @d4096 | 483 (473..486) | 681 (679..683) | 636 (631..639) | 444 | 0.71x | 0.76x | 1.09x |
| mistral7b-q4km | pp512 | 6387 (6371..6392) | 11229 (10612..11937) | 9840 (9296..10496) | 13908 | 0.57x | 0.65x | 0.46x |
| mistral7b-q4km | pp2048 | 6074 (6071..6088) | 10756 (10521..10789) | 9415 (9380..9464) | 15549 | 0.56x | 0.65x | 0.39x |
| mistral7b-q4km | tg128 @d0 | 135 (135..136) | 246 (245..247) | 229 (226..233) | 95.4 | 0.55x | 0.59x | 1.42x |
| mistral7b-q4km | tg128 @d4096 | 107 (106..108) | 215 (138..221) | 203 (200..207) | 91.3 | 0.50x | 0.53x | 1.17x |
| nemotron30b-q4km | pp512 | 5586 (5551..5591) | 8346 (7436..8833) | 7242 (7172..7487) | n/a | 0.67x | 0.77x | - |
| nemotron30b-q4km | pp2048 | 5445 (5438..5446) | 7983 (7706..8274) | 7130 (7118..7142) | n/a | 0.68x | 0.76x | - |
| nemotron30b-q4km | tg128 @d0 | 221 (218..231) | 328 (311..331) | 280 (279..284) | n/a | 0.67x | 0.79x | - |
| nemotron30b-q4km | tg128 @d4096 | 207 (205..209) | 312 (295..331) | 275 (274..276) | n/a | 0.66x | 0.75x | - |
| olmoe-q4km | pp512 | 15014 (14977..15059) | 27624 (18768..29607) | 27081 (25431..27740) | 38093 | 0.54x | 0.55x | 0.39x |
| olmoe-q4km | pp2048 | 14407 (14246..14513) | 24564 (23599..25280) | 26648 (25865..27770) | 65885 | 0.59x | 0.54x | 0.22x |
| olmoe-q4km | tg128 @d0 | 503 (497..529) | 844 (762..845) | 605 (568..633) | 350 | 0.60x | 0.83x | 1.44x |
| olmoe-q4km | tg128 @d4096 | 393 (356..409) | 616 (585..622) | 503 (358..536) | 254 | 0.64x | 0.78x | 1.55x |
| qwen27b-q8 | pp512 | 2663 (2538..2680) | 3058 (2914..3063) | 2394 (2368..2436) | 6348 | 0.87x | 1.11x | 0.42x |
| qwen27b-q8 | pp2048 | 2508 (2490..2518) | 3030 (2696..3033) | 2332 (1998..2338) | 7425 | 0.83x | 1.08x | 0.34x |
| qwen27b-q8 | tg128 @d0 | 44.8 (44.6..44.9) | 48.5 (37.5..48.9) | 44.1 (40.8..44.3) | 48.5 | 0.92x | 1.02x | 0.92x |
| qwen27b-q8 | tg128 @d4096 | 40.6 (40.1..41.0) | 47.9 (47.4..47.9) | 43.2 (39.8..43.4) | 47.6 | 0.85x | 0.94x | 0.85x |
| qwen3-0.6b-q8 | pp512 | 62021 (34682..62032) | 26977 (21071..38436) | 61296 (60828..61505) | 31164 | 2.30x | 1.01x | 1.99x |
| qwen3-0.6b-q8 | pp2048 | 43732 (20334..44888) | 55097 (48865..55235) | 56316 (55165..56608) | 119055 | 0.79x | 0.78x | 0.37x |
| qwen3-0.6b-q8 | tg128 @d0 | 825 (824..827) | 817 (808..818) | 725 (717..734) | 487 | 1.01x | 1.14x | 1.70x |
| qwen3-0.6b-q8 | tg128 @d4096 | 483 (477..484) | 629 (608..635) | 578 (573..583) | 452 | 0.77x | 0.84x | 1.07x |

Notes on the table:
* Qwen3-0.6B / Gemma / Llama-1B prefill rows are 8-60 ms long and clock-ramp sensitive; the min..max spread there (for example Sharplight 0.6B pp512 34682..62032) is noise, not signal. llama.cpp CUDA 0.6B pp512 (26977, min 21071) is a clock-ramp artefact as well.
* vLLM small-model numbers are measured through the offline Python API, which adds roughly 2 ms of host overhead per token (decode) and about 10 ms per call (prefill); they are a fair "what you get from the Python engine" number, not a kernel-level number.
* Nemotron on vLLM: not measured (see section 1).

## 3. Batched serving throughput

`bench/loadgen.py` (asyncio + httpx, `/v1/completions` streaming): closed loop with N workers, each sending 3-8 requests back to back, so N
requests are always in flight. Prompts are unique random-word sequences (about 250 tokens, no prefix-cache hits), `max_tokens=128`,
temperature 0. llama-server and vLLM get `ignore_eos` (always exactly 128 tokens); Sharplight Serve has no `ignore_eos`, so OLMoE streams stop
early (average 108-117 tokens, which lowers its aggregate tok/s slightly). Aggregate tok/s = output tokens / wall time of the whole run;
per-stream tok/s = 1 / mean inter-token gap; TTFT and ITL are measured client-side from SSE chunks (one chunk per token). One server per
engine is started for the largest concurrency (Sharplight `--slots 64 --ctx 512`, llama-server `-np 64 -c 32768`, vLLM
`--max-num-seqs 64 --max-model-len 512`) and the four concurrency levels run against it. A check with matching slot counts on the 27B
(`--slots 4` at c=4: 63 tok/s, `--slots 16` at c=16: 227 tok/s) gave the same numbers as the 64-slot server, so the shared server is not penalising Sharplight.
llama-server for the 27B and Nemotron runs with `--cache-ram 0` (the default 8 GB prompt cache made the 27B server grow to 17 GB RSS and get killed by the kernel OOM killer on this RAM-starved desktop).


#### llama3.2-1b-q8

| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sharplight | 427 | 449.8 | 12 | 27 | 2.2 | 5.0 | 128 | 0 |
| 1 | llama-cuda13 | 493 | 547.6 | 23 | 42 | 1.8 | 3.3 | 128 | 0 |
| 1 | llama-vulkan | 310 | 333.6 | 31 | 39 | 3.0 | 5.8 | 128 | 0 |
| 1 | vllm | 458 | 470.0 | 8 | 11 | 2.1 | 2.4 | 128 | 0 |
| 4 | sharplight | 783 | 202.2 | 22 | 42 | 4.9 | 8.5 | 128 | 0 |
| 4 | llama-cuda13 | 637 | 175.7 | 81 | 103 | 5.7 | 10.4 | 128 | 0 |
| 4 | llama-vulkan | 572 | 153.4 | 68 | 82 | 6.5 | 10.2 | 128 | 0 |
| 4 | vllm | 1533 | 397.6 | 14 | 19 | 2.5 | 3.2 | 128 | 0 |
| 16 | sharplight | 1716 | 111.9 | 29 | 207 | 8.9 | 13.7 | 128 | 0 |
| 16 | llama-cuda13 | 998 | 71.0 | 269 | 392 | 14.1 | 26.3 | 128 | 0 |
| 16 | llama-vulkan | 422 | 40.5 | 1457 | 3591 | 24.7 | 33.6 | 128 | 0 |
| 16 | vllm | 5419 | 375.5 | 42 | 50 | 2.7 | 3.1 | 128 | 0 |
| 64 | sharplight | 3469 | 58.4 | 64 | 518 | 17.1 | 23.2 | 128 | 0 |
| 64 | llama-cuda13 | 1340 | 22.2 | 355 | 474 | 45.1 | 61.0 | 128 | 0 |
| 64 | llama-vulkan | 1454 | 25.7 | 326 | 1727 | 38.9 | 57.3 | 128 | 0 |
| 64 | vllm | 15728 | 313.2 | 108 | 161 | 3.2 | 3.9 | 128 | 0 |

#### mistral7b-q4km

| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sharplight | 123 | 128.3 | 51 | 55 | 7.8 | 7.9 | 128 | 0 |
| 1 | llama-cuda13 | 198 | 211.8 | 48 | 51 | 4.7 | 7.3 | 128 | 0 |
| 1 | llama-vulkan | 204 | 220.1 | 50 | 52 | 4.5 | 4.7 | 128 | 0 |
| 1 | vllm | 94 | 95.4 | 25 | 26 | 10.5 | 11.5 | 128 | 0 |
| 4 | sharplight | 196 | 51.5 | 139 | 181 | 19.4 | 19.4 | 128 | 0 |
| 4 | llama-cuda13 | 311 | 87.6 | 184 | 285 | 11.4 | 19.9 | 128 | 0 |
| 4 | llama-vulkan | 321 | 87.4 | 143 | 147 | 11.4 | 11.8 | 128 | 0 |
| 4 | vllm | 365 | 94.9 | 74 | 76 | 10.5 | 11.5 | 128 | 0 |
| 16 | sharplight | 505 | 33.8 | 183 | 853 | 29.6 | 64.3 | 128 | 0 |
| 16 | llama-cuda13 | 626 | 46.1 | 530 | 766 | 21.7 | 27.0 | 128 | 0 |
| 16 | llama-vulkan | 411 | 30.2 | 858 | 1046 | 33.1 | 46.2 | 128 | 0 |
| 16 | vllm | 1205 | 87.0 | 251 | 254 | 11.5 | 12.0 | 128 | 0 |
| 64 | sharplight | 1201 | 21.4 | 321 | 2717 | 46.7 | 99.5 | 128 | 0 |
| 64 | llama-cuda13 | 1928 | 36.8 | 747 | 1282 | 27.2 | 19.6 | 128 | 0 |
| 64 | llama-vulkan | 1233 | 23.6 | 1029 | 2267 | 42.4 | 32.2 | 128 | 0 |
| 64 | vllm | 2980 | 62.5 | 910 | 984 | 16.0 | 14.9 | 128 | 0 |

#### nemotron30b-q4km

| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sharplight | 194 | 213.0 | 64 | 65 | 4.7 | 4.9 | 128 | 0 |
| 1 | llama-cuda13 | 286 | 321.6 | 52 | 57 | 3.1 | 3.6 | 128 | 0 |
| 1 | llama-vulkan | 237 | 268.4 | 68 | 69 | 3.7 | 4.0 | 128 | 0 |
| 4 | sharplight | 192 | 210.3 | 2065 | 2080 | 4.8 | 4.9 | 128 | 0 |
| 4 | llama-cuda13 | 413 | 114.5 | 149 | 170 | 8.7 | 8.9 | 128 | 0 |
| 4 | llama-vulkan | 315 | 86.2 | 175 | 200 | 11.6 | 12.8 | 128 | 0 |
| 16 | sharplight | 189 | 207.6 | 10209 | 10219 | 4.8 | 5.0 | 128 | 0 |
| 16 | llama-cuda13 | 546 | 38.9 | 538 | 674 | 25.7 | 27.3 | 128 | 0 |
| 16 | llama-vulkan | 350 | 24.9 | 678 | 2294 | 40.1 | 38.8 | 128 | 0 |
| 64 | sharplight | 189 | 207.4 | 42386 | 42629 | 4.8 | 5.0 | 127 | 0 |
| 64 | llama-cuda13 | 922 | 17.4 | 1259 | 3221 | 57.4 | 60.7 | 128 | 0 |
| 64 | llama-vulkan | 784 | 14.8 | 1512 | 4279 | 67.6 | 83.5 | 128 | 0 |

#### olmoe-q4km

| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sharplight | 423 | 468.0 | 24 | 28 | 2.1 | 3.6 | 108 | 0 |
| 1 | llama-cuda13 | 595 | 694.9 | 29 | 41 | 1.4 | 2.9 | 128 | 0 |
| 1 | llama-vulkan | 521 | 599.4 | 32 | 41 | 1.7 | 3.3 | 128 | 0 |
| 1 | vllm | 445 | 463.9 | 14 | 15 | 2.2 | 2.3 | 128 | 0 |
| 4 | sharplight | 489 | 133.1 | 38 | 69 | 7.5 | 9.8 | 116 | 0 |
| 4 | llama-cuda13 | 871 | 261.4 | 93 | 131 | 3.8 | 4.7 | 128 | 0 |
| 4 | llama-vulkan | 731 | 206.7 | 84 | 93 | 4.8 | 5.3 | 128 | 0 |
| 4 | vllm | 848 | 221.5 | 24 | 77 | 4.5 | 4.9 | 128 | 0 |
| 16 | sharplight | 930 | 66.4 | 63 | 314 | 15.1 | 25.9 | 115 | 0 |
| 16 | llama-cuda13 | 850 | 81.9 | 427 | 2346 | 12.2 | 18.4 | 128 | 0 |
| 16 | llama-vulkan | 699 | 58.6 | 740 | 1430 | 17.1 | 17.3 | 128 | 0 |
| 16 | vllm | 1890 | 132.9 | 61 | 362 | 7.5 | 8.6 | 128 | 0 |
| 64 | sharplight | 1909 | 34.2 | 161 | 949 | 29.2 | 47.0 | 117 | 0 |
| 64 | llama-cuda13 | 1776 | 32.9 | 667 | 1092 | 30.4 | 37.5 | 128 | 0 |
| 64 | llama-vulkan | 1896 | 34.6 | 615 | 1049 | 28.9 | 21.4 | 128 | 0 |
| 64 | vllm | 5216 | 90.2 | 156 | 216 | 11.1 | 12.1 | 128 | 0 |

#### qwen27b-q8

| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sharplight | 40 | 41.6 | 118 | 126 | 24.0 | 24.3 | 128 | 0 |
| 1 | llama-cuda13 | 41 | 42.5 | 138 | 144 | 23.5 | 41.9 | 128 | 0 |
| 1 | llama-vulkan | 36 | 38.3 | 210 | 267 | 26.1 | 36.2 | 128 | 0 |
| 1 | vllm | 46 | 47.2 | 56 | 178 | 21.2 | 25.6 | 128 | 0 |
| 4 | sharplight | 60 | 15.5 | 329 | 415 | 64.6 | 64.8 | 128 | 0 |
| 4 | llama-cuda13 | 68 | 19.2 | 542 | 3731 | 52.2 | 57.5 | 128 | 0 |
| 4 | llama-vulkan | 50 | 13.4 | 752 | 1158 | 74.7 | 114.7 | 128 | 0 |
| 4 | vllm | 164 | 43.2 | 175 | 274 | 23.2 | 28.0 | 128 | 0 |
| 16 | sharplight | 207 | 13.7 | 422 | 1496 | 72.9 | 166.0 | 128 | 0 |
| 16 | llama-cuda13 | 106 | 7.7 | 3015 | 4007 | 129.4 | 215.6 | 128 | 0 |
| 16 | llama-vulkan | 80 | 5.5 | 2817 | 4891 | 181.0 | 226.2 | 128 | 0 |
| 16 | vllm | 556 | 40.5 | 566 | 573 | 24.7 | 25.6 | 128 | 0 |
| 64 | sharplight | 479 | 8.5 | 709 | 6434 | 118.0 | 236.0 | 128 | 0 |
| 64 | llama-cuda13 | 263 | 5.0 | 4169 | 12888 | 201.0 | 378.0 | 128 | 0 |
| 64 | llama-vulkan | 213 | 4.1 | 6218 | 16391 | 245.3 | 511.7 | 128 | 0 |
| 64 | vllm | 1166 | 24.6 | 1625 | 2185 | 40.6 | 40.4 | 128 | 0 |


| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 16 | sharplight | 227 | 15.1 | 422 | 1444 | 66.0 | 92.4 | 128 | 0 |


| conc | engine | agg tok/s | per-stream tok/s | TTFT p50 ms | TTFT p95 ms | ITL mean ms | ITL p95 ms | avg out tok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 4 | sharplight | 63 | 16.2 | 323 | 349 | 61.8 | 62.7 | 128 | 0 |

### Batched decode step, kernel level (`bench/BatchBench`)

Per-step latency of Sharplight's multi-sequence decoder (`LlmModel.CreateBatchDecoder`, 64 slots, about 250 tokens of context per slot,
logits wanted for every row). This isolates the GPU/engine step from HTTP and sampling:

| batch | Qwen3.8-27B Q8_0 step ms (tok/s) | Mistral-7B Q4_K_M | Llama-1B Q8_0 | OLMoE Q4_K_M |
|---|---|---|---|---|
| 1 | 24.8 (40) | 7.5 (133) | 1.37 (727) | 1.96 (510) |
| 2 | 37.1 (54) | 11.8 (169) | 2.34 (855) | 3.31 (604) |
| 4 | 60.7 (66) | 18.8 (213) | 3.71 (1078) | 6.31 (634) |
| 8 | 113.6 (70) | 33.8 (237) | 7.15 (1118) | 9.11 (878) |
| 16 | 58.8 (272) | 24.8 (646) | 5.83 (2742) | 12.0 (1331) |
| 32 | 68.3 (469) | 28.1 (1138) | 8.90 (3594) | 14.8 (2167) |
| 64 | 92.2 (694) | 32.0 (2000) | 12.1 (5277) | 19.9 (3222) |

Reference points from the serving runs: vLLM's inter-token latency at 64 streams is 40.6 ms on the 27B (FP8), 3.2 ms on Llama-1B (bf16), 11.1 ms
on OLMoE. Without logits the Llama-1B step at batch 64 takes 5.1 ms instead of 12.1 ms (the lm-head, 128k-entry logits readback and
CPU-side sampling cost 7 ms per step); through the HTTP server the measured ITL at c=64 is 17.1 ms, i.e. another 5 ms of host-side engine/HTTP work.

## 4. Load time (time to first generated token, `bench/load_time.sh`)

Wall-clock seconds for `sharplight <model> -p hi -n 1` (resp. `llama-completion -n 1`), including process start, weight upload, kernel
compilation or cache lookup, 1-token prefill and 1 decode. Sharplight has three cache layers: the Slang kernel disk cache
(`$XDG_DATA_HOME/sharplight/kernels`, i.e. `~/.local/share/sharplight/kernels`, filled automatically), the driver pipeline cache, and the `.slm`
model package created with `sharplight model.gguf --save out.slm` (built model + compiled kernels + pipeline cache; load it by passing the `.slm` path).
"Cold caches" = empty `XDG_DATA_HOME` and `__GL_SHADER_DISK_CACHE_PATH`.
"Cold page cache" = model file evicted with `posix_fadvise(DONTNEED)` before each run (`bench/evict.py`; a real cold start from NVMe).

| model | engine / case | seconds |
|---|---|---|
| Qwen3.8-27B Q8_0 | Sharplight GGUF, cold kernel + driver caches (file in page cache) | 23.5 |
| | Sharplight GGUF, warm caches, file in page cache | 4.8, 4.8 |
| | Sharplight GGUF, warm caches, cold page cache | 9.7, 8.6 (8.4, 8.5 in an earlier run) |
| | Sharplight `.slm`, file in page cache | 8.8 (first), 4.8 |
| | Sharplight `.slm`, cold page cache | 8.0, 7.9 |
| | llama.cpp CUDA, file in page cache | 8.9 (first), 4.4 |
| | llama.cpp Vulkan, file in page cache | 8.1, 7.7 |
| | llama.cpp CUDA / Vulkan, cold page cache | 36.1, 22.2 / 25.7, 39.3 |
| | vLLM FP8 (`LLM()` init, torch.compile and CUDA graph caches warm) / `vllm serve` ready | 131 / 126 |
| Llama-3.2-1B Q8_0 | Sharplight GGUF cold caches | 8.9 |
| | Sharplight GGUF warm / `.slm` | 1.0-1.1 / 0.94-0.97 |
| | llama.cpp CUDA / Vulkan | 0.77-1.05 / 1.02-1.07 |
| | vLLM bf16, first ever start (cold torch.compile cache) / later `vllm serve` ready | 229 / 62 |
| Mistral-7B Q4_K_M | Sharplight GGUF cold caches | 12.8 |
| | Sharplight GGUF warm / `.slm` | 1.3-1.6 / 1.2 |
| | Sharplight GGUF / llama.cpp CUDA / llama.cpp Vulkan, cold page cache | 2.2-2.9 / 1.5-1.7 / 1.9 |
| | llama.cpp CUDA / Vulkan, file in page cache | 0.94-1.03 / 1.17 (3.9 first) |
| | vLLM bf16 `LLM()` init / `vllm serve` ready | 54 / 43 |
| Qwen3-0.6B / Gemma-3-1B bf16 (vLLM only) | first ever start, cold torch.compile cache | 241 / 143 |

Take-aways: the Slang kernel/pipeline cache is what matters (cold start costs 8-23 s of shader compilation; warm start is on par with
llama.cpp); the `.slm` package is not faster than a GGUF with a warm kernel cache on this machine, its benefit is portability of the compiled state.
Sharplight reads the 27B from a cold page cache in 8-10 s versus 22-39 s for llama.cpp (mmap); vLLM needs 1-4 minutes (torch.compile + CUDA graph capture, cached afterwards).


## 5. Ranked gaps (where Sharplight is slower)

Ratios are Sharplight / best comparable engine (higher is better for Sharplight). "Comparable" means same quantisation class: llama.cpp
for the GGUF models, vLLM FP8 for the 27B, vLLM bf16 only for batching behaviour.

| # | Gap | Size | Evidence |
|---|---|---|---|
| 1 | **Serving a Mamba/SSM-hybrid (Nemotron-3-Nano) does not batch at all** | c=64: 189 vs 922 tok/s llama.cpp CUDA (**0.21x**); c=16 0.35x; c=4 0.46x; TTFT at c=64 is 42 s | Server log: `Multi-sequence decoding unavailable (op SelectiveScan uses state but has no multi-sequence form); serving one sequence at a time.` Throughput stays at the c=1 value (about 190 tok/s) for every concurrency. Feature gap, not a kernel gap. |
| 2 | **Batched decode on the 27B hybrid (gated delta net + attention)** | vs vLLM FP8: c=4 0.37x (60 vs 164), c=16 0.37x (207 vs 556), c=64 0.41x (479 vs 1166). vs llama.cpp CUDA: 0.88x at c=4, but 1.9x / 1.8x at c=16 / 64 | Step latency B=1 24.8 ms, B=4 60.7 ms, B=8 113.6 ms, B=16 58.8 ms, B=64 92.2 ms (vLLM: about 24 ms up to 16 streams, 40 ms at 64). Using four streams costs 2.5x the single-stream step, so aggregate throughput is flat between 1 and 8 streams. |
| 3 | **Batched decode on small dense / MoE models** | Llama-1B vs vLLM bf16: c=4 0.51x, c=16 0.32x, c=64 0.22x (3469 vs 15728); OLMoE: c=4 0.58x, c=16 0.49x, c=64 0.37x vs vLLM and 0.56x at c=4 vs llama.cpp CUDA (489 vs 871). Mistral Q4_K_M: c=4 0.63x, c=64 0.62x vs llama.cpp CUDA. Sharplight still beats llama.cpp at c=16/64 for Llama-1B (1.7x / 2.6x) and the 27B | Kernel-level step: Llama-1B B=4 is 2.7x of B=1, B=8 5.2x; B=64 12.1 ms of which about 7 ms is logits (lm-head + readback), plus about 5 ms of engine/HTTP overhead. vLLM's step at 64 streams is 3.2 ms. |
| 4 | **Q4_K / Q6_K quantised models, single stream** | Mistral-7B Q4_K_M: decode 135 vs 246 tok/s CUDA (**0.55x**), pp512 6387 vs 11229 (**0.57x**); OLMoE: pp 0.54x, decode 0.60x (0.83x vs llama.cpp Vulkan); Nemotron-30B: 0.67x prefill and decode. Q8_0 models are at parity | Profile: Q4_K matvec kernels reach 0.54-0.90 TB/s, Q6_K 0.84 TB/s, whole token 0.59 TB/s effective (llama.cpp CUDA 1.07 TB/s), against 1.59 TB/s for the Q8_0 matvec of the 27B. |
| 5 | **Prefill throughput on large dense models vs FP8** | 27B: pp512 2663 vs 6348 tok/s vLLM FP8 (**0.42x**), pp2048 0.34x; vs llama.cpp CUDA 0.87x / 0.83x, vs llama.cpp Vulkan 1.11x / 1.08x. Llama-1B pp2048: 0.83x vs llama.cpp CUDA, 0.48x vs vLLM; Qwen3-0.6B pp2048 0.79x | `gemm_cm2` reaches 101 TFLOPS (Mistral Q4_K) to 178 TFLOPS (27B Q8_0) at T=512. |
| 6 | **Decode at long context on small models** | tg128 @d4096: Llama-1B 483 vs 681 tok/s CUDA (**0.71x**), Qwen3-0.6B 0.77x, 27B 0.85x, Mistral 0.50x (compounded by #4), OLMoE 0.64x; Gemma-3-1B 0.96x | Profile: decode attention reads KV at only about 0.18 TB/s at depth 4096 (47 us per layer for 8.4 MB), 0.75 ms of a 2.0 ms step. |
| 7 | **Server per-token overhead at c=1 for small models** | Llama-1B c=1: 427 vs 493 tok/s llama.cpp CUDA (0.87x); OLMoE 423 vs 595 (0.71x). CLI shows 752 / 503 tok/s, so the server loses 40 % | ITL mean 2.2 ms vs 1.37 ms step: host-side logits readback, sampling over a 128k vocabulary, SSE per token. |

Where Sharplight is at or ahead of the field: Q8_0 single-stream decode (Qwen3.8-27B 0.92x CUDA / 1.02x Vulkan, small Q8_0 models 1.01-1.07x
CUDA / 1.07-1.14x Vulkan, and 1.4-2.7x over vLLM's bf16 Python path on the small models, partly a quantisation effect), Gemma-3-1B (near parity everywhere, 0.90x at pp2048), large-concurrency serving versus llama-server (27B c=16/64: 1.95x / 1.8x
CUDA, Llama-1B c=16/64: 1.7x / 2.6x), TTFT at high concurrency versus llama-server (Llama-1B c=64: 64 vs 355 ms p50), and cold-file load of the 27B (8-10 s vs 22-39 s).

## 5a. Serving-stack follow-up (server overhead, chunked prefill, speculation, structured outputs)

Measured after the table above, on the same machine while other jobs kept the GPU at 100 % utilisation, so wall-clock tok/s
swings by ±30 % between runs. The robust numbers are the server's own `/metrics` (GPU-synchronous step time versus the
inter-token latency it reports) and step counts.

**Per-token server overhead (gap #7).** Llama-3.2-1B Q8_0, c=1, 256 tokens with `ignore_eos`: decoder step mean 1.81 ms against
an inter-token latency of 2.02 ms before (0.21 ms, 10 % host overhead), 1.74 ms against 1.84 ms after (0.10 ms, 5.6 %).
Changes: logits rows are handed to the sampler without copies (`IBatchDecoder.Step` returns slices of the readback buffer),
large readbacks are split across threads, sampling and stop checks run in parallel once four or more sequences are in the step,
arg-max is vectorised, and SSE chunks are written with `Utf8JsonWriter` straight into the response pipe with one flush per
burst of ready events. At c=64 the profile share of the logits memmove dropped from 23 % to 13.5 % of engine time and per-sequence
sampling/emit from 17.7 % to 7.9 %. The remaining gap between the CLI's 752 tok/s and the server's 427 tok/s mean in the table
is mostly measurement (CLI best-of versus server mean under contention, plus TTFT inside the server's mean); what is left on the
host is the thread-pool wake per streamed token. `ignore_eos` and `min_tokens` are now accepted, so the table's OLMoE caveat no
longer applies to new runs.

**Chunked prefill with decode interleaving** (`bench/loadgen.py --long-prompt-words 2500 --long-every 3`, Llama-1B, 8 slots,
c=8, 100-word prompts plus every third request about 3k tokens, 128 output tokens):

| step budget / chunk | agg tok/s | TTFT p50 ms | short-request TTFT p95 ms | ITL p99 ms | worst inter-token gap ms |
|---|---|---|---|---|---|
| whole prompts (4096 / 4096) | 287 | 128 | 248 | 28.8 | 4347 |
| 512 / 512 (default) | 554 | 61 | 203 | 30.9 | 36 |
| 256 / 256 | 430 | 221 | 506 | 31.7 | 37 |

With chunking the worst stall a running stream sees is one step (36 ms) instead of seconds. Part of the unchunked worst case
is the first compilation of the 4096-row graph during the run (the warm-up only used short prompts), which is itself an argument
for a bounded step shape.

**Prompt-lookup speculation** (`sharplight serve --speculate-ngram N`; an edit task whose answer copies the prompt, Llama-1B,
c=1, temperature 0, 6 requests of 140 tokens; output identical to non-speculative decoding):

| drafts | engine steps | GPU step time total | drafted | accepted | wall tok/s (noisy) |
|---|---|---|---|---|---|
| off | 846 | 2.20 s | 0 | 0 | 390-550 |
| 4 | 372 | 1.22 s (0.55x) | 672 | 474 (70 %) | 545-644 |
| 8 | 318 | 0.97 s (0.44x) | 888 | 528 (59 %) | 731-864 |

**Structured outputs.** Qwen3-0.6B (151k vocabulary): vocabulary trie built once per tokenizer in 0.2-0.6 s (on first use);
the first mask of a JSON-schema grammar 0.3-3 ms; a JSON string body 11-16 ms the first time per grammar, then 0.15-0.5 ms in any
other context or request (shared per-stack-top cache) and 0.01-0.2 ms for a repeated state. A schema answer and a forced tool
call on the real model parse and validate (`RealModelStructuredTests`).

## 6. Hypotheses for the top gaps

Diagnostics used: `SHARPLIGHT_VK_PROFILE=1` (per-kernel GPU timestamps in `--bench`, and in `bench/BatchBench --profile B` for batched steps; raw output in
`bench/results/profiles/`), `SHARPLIGHT_ATTN_SPLIT/ATTN_FA` ablations, `SHARPLIGHT_NO_*` switches listed in `src/Sharplight.Shaders/DiagnosticSwitches.cs`,
the server log, and the dispatch/barrier counts printed by `--bench` (for example 420 dispatches / 291 barriers per decode step for Mistral-7B).

**Gap 1, SSM serving.** `SelectiveScan` (Mamba-2) has no multi-sequence form, so `LlmModel.CreateBatchDecoder` falls back to `SessionBatchDecoder`, which runs the
slots of a step one after another. Adding a multi-sequence selective scan (and its conv state) would remove the gap; `delta_rule` already has one, which is why the Qwen hybrid batches.

**Gap 2/3, small-batch decode.** Evidence from `bench/results/profiles/batch_*.txt`:
* For 2-8 rows the lowering picks the row-batched `matvec` kernels (`MatvecMaxRows = 8` in `GraphLowerer.MatMul.cs`). Their cost grows almost linearly with rows: on the 27B, `[5120→17408]x2` takes 119 us at T=1 and 252 us at T=4 (the weights are not amortised across rows, the kernels become dequantisation/FMA bound); `[17408→5120]` 68 us vs 250 us. The 8-row step (113.6 ms) is slower than the 16-row step (58.8 ms) that already uses the tensor-core path, so the crossover is far too high; lowering `MatvecMaxRows` to 2-4, or a dedicated skinny-M tensor-core / `mma` path with weights read once, is the first experiment.
* The tensor-core path for 9-64 rows (`gemm_cm2` split-k, 128-row tiles) is bandwidth-inefficient for skinny M: `[5120→17408]` at T=16 reads 94.7 MB in 143 us = 0.66 TB/s versus 1.59 TB/s for the T=1 matvec; `to_f16` activation conversion plus `splitk_reduce` add 9940 + 9920 tiny launches per 20 steps (74.6 ms of 1081 ms GPU time, about 7 %), and gate/up are not fused as in the matvec path (`matmul2 ... fused[Silu,Mul]` exists only for matvec). A step at B=16 should cost about the same as B=1 (weight-bandwidth bound: about 25 ms for the 27B, 1.5-2 ms for Llama-1B) but costs 2.4x and 4x of that.
* Logits: `IBatchDecoder.Step` returns a `float[vocab]` per sequence. For Llama-1B (vocab 128256) at B=64 that is 33 MB per step over PCIe plus 64 CPU argmax/softmax passes; the step takes 12.1 ms with logits and 5.1 ms without. For the 27B vocab (248320) it is 63 MB per step. Sampling on the GPU (argmax/top-k/top-p kernel, return only token ids) would remove most of this.
* Remaining host overhead (about 5 ms per step at c=64 for Llama-1B, 2.2 ms vs 1.37 ms at c=1): per-token SSE flush and request bookkeeping in `InferenceEngine`/`OpenAiApi`; the engine step itself is single-threaded between GPU submissions (GPU idle while the CPU samples and schedules). vLLM overlaps scheduling with the previous step (async scheduling) and uses CUDA graphs; Sharplight records command buffers per shape (`record T=...` in the load timing), which is fine for B=1 but each new batch size is a separate recording.
* The matvec kernels at T=1 are at the bandwidth roof for Q8_0, so single-stream and batch-1 are fine; the loss is entirely in the 2-64 range.

**Gap 4, Q4_K/Q6_K.** The Q8_0 path has an "aligned quants" repack (`SHARPLIGHT_NO_REPACK`, `q8_0r` in the profile); K-quants are read in their 144/210-byte super-block file layout and dequantised with scalar unpacking (`load8_q4_k` in `common.slang`). Profile of Mistral decode: `[4096→14336]x2` q4_k 73.7 us for 66 MB = 0.90 TB/s, `[14336→4096]` q4_k 53.6 us for 33 MB = 0.62 TB/s, `[4096→4096]` q4_k 17.4 us for 9.4 MB = 0.54 TB/s, `[14336→4096]` q6_k 57.7 us for 48 MB = 0.84 TB/s, versus 1.59 TB/s for Q8_0. The device reports `int_dot` and `matrix_int8` (`sharplight --devices`); llama.cpp uses them (q8_1-quantised activations, `dp4a`/int8 MMA) while Sharplight dequantises to float. For prefill (`gemm_cm2`, 594 us for a 60 GFLOP gate/up at T=512 = 101 TFLOPS) the in-kernel K-quant dequantisation limits the tensor-core feed; Q8_0 reaches 178 TFLOPS. Hypotheses to test: a repacked/aligned K-quant layout, integer-dot matvec with quantised activations, and dequantise-to-f16 staging in shared memory for the GEMM.

**Gap 5, prefill.** Prefill of the 27B at T=512 is spent 79 % in `gemm_cm2` (178 TFLOPS of f16 tensor-core math on 513 us tiles) and 9.5 % in `delta_rule`; llama.cpp CUDA (int8 tensor cores with q8_1 activations) is 15 % faster, vLLM FP8 (fp8 tensor cores) 2.4x faster. The device already reports `matrix_int8` and 16x16x32 `i8*i8+i32` / `f8E4m3*f8E4m3` cooperative-matrix shapes (`sharplight --devices`), so the hypothesis is that an 8-bit MMA GEMM (activations quantised to int8/fp8, Q8_0 weights used as stored) instead of the f16 coopmat would close most of the gap. Llama-1B pp2048: `gemm_cm2` is 66 % of GPU time, `attention fa` 16 %; `SHARPLIGHT_ATTN_FA=0` drops 4096-token prefill from 43.6k to 8.7k tok/s, so the flash path is essential and fine.

**Gap 6, long-context decode.** `SHARPLIGHT_ATTN_SPLIT=0` drops Llama-1B decode at depth 4096 from 489 to 155 tok/s, so split-key attention is working, but it still only reaches about 0.18 TB/s (KV is 8 KV heads x 64 dims; 47 us per layer at 4096 vs 26.6 us at 2048). Likely too few workgroups/too short per-workgroup key ranges for GQA with 8 KV heads, plus the extra `attention combine` launch and its barrier; try more splits at depth > 2048 and processing all query heads of a KV head in one workgroup (the kernels already share K/V reads only if grouped this way, which could be verified with `SHARPLIGHT_DUMP_KERNELS`).

## 7. How to reproduce

```bash
# build (not rebuilt for the bench: the binaries in samples/*/bin/Release were used as they were)
dotnet build Sharplight.slnx -c Release
dotnet build bench/BatchBench -c Release              # batched-decode micro benchmark (not in the .slnx)

# llama.cpp single stream (both builds), all models:           bench/results/single_llama-*.jsonl
bench/run_llama_single_all.sh                          # llama-bench -ngl 99 -fa 1 -p 512,2048 -n 128 -d 0,4096 -r 5 -o jsonl
# Sharplight single stream (3 process runs per point):          bench/results/single_sharplight_*.txt
bench/run_sharplight_single_all.sh                     # sharplight MODEL --bench N --ctx N+256 --batch 512, N = 16,512,2048,4096
# vLLM single stream (offline LLM API):                          bench/results/single_vllm_*.json
VLLM_USE_FLASHINFER_SAMPLER=0 ~/.venvs/vllm/bin/python -I bench/vllm_single.py ~/models/hf/Llama-3.2-1B-Instruct llama3.2-1b-bf16 --gpu-util 0.25
bench/run_vllm_single_all.sh                           # (+ --max-len 4000 --depth 3800 for OLMoE, see below)

# serving (one server per engine/model, concurrency 1 4 16 64):   bench/results/serve_<engine>_<model>_c<N>.json
bench/run_serve_all.sh llama3.2-1b-q8                  # sharplight, llama-cuda13, llama-vulkan, vllm (VLLM_MODEL / VLLM_UTIL env)
#   Sharplight:  sharplight serve MODEL --slots 64 --ctx 512 --listen http://127.0.0.1:18080 --temp 0
#   llama.cpp:   llama-server -m MODEL -ngl 99 -fa on -np 64 -c 32768 --port 18080 --no-webui [--cache-ram 0]
#   vLLM:        vllm serve HF_DIR --served-model-name m --max-model-len 512 --max-num-seqs 64 --no-enable-prefix-caching --gpu-memory-utilization 0.3
#   load:        python bench/loadgen.py --url http://127.0.0.1:18080 -c N --per-worker 3 [--ignore-eos]   # -c 1,4,16,64
# load time:                                                    bench/results/load_time*.txt
bench/load_time.sh <models.sh entries>; bench/load_time_coldfile.sh
# batched decode step and per-kernel profiles:                   bench/results/profiles/
bench/BatchBench/bin/Release/net11.0/BatchBench MODEL --ctx 1024 --profile 4
SHARPLIGHT_VK_PROFILE=1 samples/Sharplight.Cli/bin/Release/net11.0/sharplight MODEL --bench 512 --ctx 768
# tables
python3 bench/summarize_single.py; python3 bench/summarize_serve.py
```

Sharplight serve/CLI binaries used: `samples/Sharplight.Cli/bin/Release/net11.0/sharplight`, `samples/Sharplight.Server/bin/Release/net11.0/sharplight-server` (now `sharplight serve`).
`vllm_single.py` options used beyond the defaults: `--gpu-util 0.5` (27B), `--gpu-util 0.35 --depth 3800 --max-len 4000` (OLMoE).

### Caveats and things that went wrong

* vLLM could not be benchmarked on Nemotron-3-Nano (FP8 checkpoint needs FP8 KV cache, hence the FlashInfer backend, whose JIT needs an nvcc matching the CUDA 13.0 wheels; the pip nvcc 13.5 is incompatible with the 13.0 headers). GGUF loading in vLLM was not usable: vLLM needs an HF config for GGUF and none of the 27B / Mistral / Nemotron checkpoints are local, so HF checkpoints were downloaded instead (`~/models/hf-vllm/`, about 75 GB; delete when no longer needed).
* The first llama-server 27B serving attempt was killed by the kernel OOM killer (prompt cache in host RAM); it was rerun with `--cache-ram 0` and the earlier partial files deleted.
* The first `load_time_coldfile.sh` run hung because `llama-completion` waited on stdin; rerun with `</dev/null`. The 27B Sharplight cold-file numbers of that first run (8.4, 8.5 s) agree with the rerun.
* Sharplight Serve has no `ignore_eos`, so OLMoE responses stop at 108-117 tokens on average; its per-token metrics (ITL, per-stream tok/s) are unaffected.
* Sharplight prefill figures are the best of 4 in-process repetitions per run (the CLI reports best), medians are taken over 3 process runs; llama-bench reports all 5 samples (median shown).
* No git history is available in this tree, so "version" for Sharplight is the working tree as of the measurement date. A repo-wide cleanup is about to start; numbers here are the pre-cleanup baseline.

## 8. After the decode / prefill performance work (2026-11-13)

Same commands as section 7: `sharplight MODEL --bench N --ctx N+256 --batch 512`, N = 16, 512, 2048, 4096, 3 process runs per
point, median shown ("before" = the baseline of section 2, llama.cpp CUDA column copied from there). The GPU was shared with other
agents' test runs and benchmarks during these measurements; small-model prefill rows stay noisy.

What changed (each switch selects the previous kernel, which is also the fallback on devices without the feature):

* **Integer-dot matvec** (`mmvq.slang`, needs `DeviceFeatures.IntDot`): activations quantized to q8_1 once per row and shared by all
  matmuls reading them, int8 dot products against Q8_0 / Q4_K / Q5_K / Q6_K weights for 1-16 rows (Q8_0 from 2 rows, K-quants up
  to 8), also for expert matmuls (MatMulId, up to 128 items) and the fused gate/up pair. `SHARPLIGHT_MMVQ_MAX_ROWS`,
  `SHARPLIGHT_MOE_MMVQ_MAX`.
* **Packed-half K-quant decoding** in the tensor-core GEMMs (`gemm_cm2`, `moe_gemm_cm`): one vector load of the codes and half2 math
  instead of a per-value f32 conversion (Mistral `4096→14336` GEMM at T=512: 100 → 155 TFLOPS).
* **32/64-token GEMM tiles** for 17-64 rows (`SHARPLIGHT_GEMM2_SKINNY=0` restores the 128-token tiles).
* **Split-K flash decoding**: all query heads of a KV head in one workgroup, split count chosen per row at run time
  (`SHARPLIGHT_ATTN_DECODE=0`), also for multi-sequence batches up to 64 rows.
* **GPU sampling**: greedy / temperature / top-k / top-p / min-p with a seeded RNG on the device, only token ids read back
  (`LlmLoadOptions.DeviceSampling`, `SHARPLIGHT_DEVICE_SAMPLING=0`; penalties, logit bias and top_logprobs still sample on the CPU).
* Workgroup-per-row LayerNorm for decode (Phi-2 Q4_K_M decode 322 → 458 tok/s; llama-bench CUDA 362) and a thread-per-row
  reduction for the MoE slot sum.

Precision: the int-dot matvec errs by 0.2-0.4 % relative rms per matmul (uniform inputs, Q4_K/Q5_K/Q6_K/Q8_0), a few percent in
the logits of small models and up to ~20 % on Mamba-130m tokens with activation outliers; greedy text then follows llama.cpp
only until near-tied tokens. The Families greedy-string tests and the multi-sequence test therefore run with `int_dot` off.
The packed-half decoding rounds the K-quant scale and min to f16 (multi-sequence logits within 1e-2 instead of 5e-3).
All new kernels address weight rows, expert slices and KV-cache elements on integers (`rowStart`/`byteAt`, NVIDIA's 4 GiB
pointer-offset truncation).

### Single stream, before → after

| model | metric | before | after | after/before | llama.cpp CUDA | after/CUDA |
|---|---|---|---|---|---|---|
| gemma3-1b-q8 | pp512 | 45642 | 45270 | 0.99x | 45353 | 1.00x |
| gemma3-1b-q8 | pp2048 | 41899 | 41956 | 1.00x | 46479 | 0.90x |
| gemma3-1b-q8 | tg128 @d0 | 554 | 560 | 1.01x | 538 | 1.04x |
| gemma3-1b-q8 | tg128 @d4096 | 505 | 533 | 1.06x | 524 | 1.02x |
| llama3.2-1b-q8 | pp512 | 53317 | 55057 | 1.03x | 58918 | 0.93x |
| llama3.2-1b-q8 | pp2048 | 48470 | 49585 | 1.02x | 58092 | 0.85x |
| llama3.2-1b-q8 | tg128 @d0 | 752 | 749 | 1.00x | 739 | 1.01x |
| llama3.2-1b-q8 | tg128 @d4096 | 483 | 663 | 1.37x | 681 | 0.97x |
| mistral7b-q4km | pp512 | 6387 | 9496 | 1.49x | 11229 | 0.85x |
| mistral7b-q4km | pp2048 | 6074 | 8677 | 1.43x | 10756 | 0.81x |
| mistral7b-q4km | tg128 @d0 | 135 | 248 | 1.83x | 246 | 1.01x |
| mistral7b-q4km | tg128 @d4096 | 107 | 218 | 2.04x | 215 | 1.01x |
| nemotron30b-q4km | pp512 | 5586 | 5867 | 1.05x | 8346 | 0.70x |
| nemotron30b-q4km | pp2048 | 5445 | 5626 | 1.03x | 7983 | 0.70x |
| nemotron30b-q4km | tg128 @d0 | 221 | 239 | 1.08x | 328 | 0.73x |
| nemotron30b-q4km | tg128 @d4096 | 207 | 226 | 1.09x | 312 | 0.72x |
| olmoe-q4km | pp512 | 15014 | 19897 | 1.33x | 27624 | 0.72x |
| olmoe-q4km | pp2048 | 14407 | 19069 | 1.32x | 24564 | 0.78x |
| olmoe-q4km | tg128 @d0 | 503 | 705 | 1.40x | 844 | 0.84x |
| olmoe-q4km | tg128 @d4096 | 393 | 585 | 1.49x | 616 | 0.95x |
| qwen27b-q8 | pp512 | 2663 | 2656 | 1.00x | 3058 | 0.87x |
| qwen27b-q8 | pp2048 | 2508 | 2441 | 0.97x | 3030 | 0.81x |
| qwen27b-q8 | tg128 @d0 | 44.8 | 43.9 | 0.98x | 48.5 | 0.90x |
| qwen27b-q8 | tg128 @d4096 | 40.6 | 41.7 | 1.03x | 47.9 | 0.87x |
| qwen3-0.6b-q8 | pp512 | 62021 | 62192 | 1.00x | 26977 | 2.31x |
| qwen3-0.6b-q8 | pp2048 | 43732 | 45136 | 1.03x | 55097 | 0.82x |
| qwen3-0.6b-q8 | tg128 @d0 | 825 | 833 | 1.01x | 817 | 1.02x |
| qwen3-0.6b-q8 | tg128 @d4096 | 483 | 651 | 1.35x | 629 | 1.03x |

Open gaps: Nemotron-30B (0.70-0.73x; its expert GEMM reaches about 47 TFLOPS against 155 for the dense K-quant GEMM), OLMoE
(0.72-0.84x except deep decode), prefill at 2048 tokens (0.81-0.90x), the 27B decode (0.87-0.90x). An int8 tensor-core prefill
(MMQ-style) was not done.

### Batched decode step, before → after (`bench/BatchBench`)

`BatchBench MODEL --ctx 2048` (64 slots, 256 tokens of context per slot; with `--ctx 1024` as in section 3 the slots now run out of
context, because four modes are measured). Step ms, minimum over 30 steps, logits read back for every row ("before" = section 3);
"greedy" is the same step with device greedy sampling (token ids only).

| batch | 27B before | 27B after | 27B greedy | Mistral before | Mistral after | Mistral greedy | Llama-1B before | Llama-1B after | Llama-1B greedy | OLMoE before | OLMoE after | OLMoE greedy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 24.8 | 22.7 | 22.8 | 7.5 | 4.26 | 4.20 | 1.37 | 1.47 | 1.35 | 1.96 | 1.44 | 1.41 |
| 2 | 37.1 | 21.2 | 20.6 | 11.8 | 4.57 | 4.51 | 2.34 | 1.62 | 1.39 | 3.31 | 1.93 | 1.83 |
| 4 | 60.7 | 23.4 | 22.7 | 18.8 | 6.52 | 6.43 | 3.71 | 1.99 | 1.64 | 6.31 | 2.65 | 2.54 |
| 8 | 113.6 | 30.4 | 29.6 | 33.8 | 9.38 | 9.21 | 7.15 | 3.58 | 2.82 | 9.11 | 5.11 | 4.84 |
| 16 | 58.8 | 44.2 | 41.9 | 24.8 | 11.1 | 10.8 | 5.83 | 6.01 | 4.31 | 12.0 | 10.2 | 9.76 |
| 32 | 68.3 | 53.7 | 49.1 | 28.1 | 13.0 | 12.4 | 8.90 | 7.18 | 4.02 | 14.8 | 21.3 | 27.3 |
| 64 | 92.2 | 79.0 | 66.6 | 32.0 | 17.6 | 16.6 | 12.1 | 14.4 | 5.22 | 19.9 | 19.0 | 16.5 |

Aggregate throughput with device greedy sampling at batch 16: 27B 382 tok/s (44 at batch 1), Mistral 1486 (238), Llama-1B 3712
(741), OLMoE 1639 (709). The Llama-1B logits column at 16-64 rows is dominated by the 128k-entry logits readback and varies by 2x
between runs; the OLMoE batch-32 row was disturbed by another GPU user (its no-logits step measured 13.4 ms in the same run).

## 9. Multi-head latent attention on tensor cores (2026-11-13)

DeepSeek-V2-Lite (16 heads, cache rows 576 = 512 latent + 64 rope) and MiniCPM3-4B (40 heads, 288 = 256 + 32) attend in the
absorbed form: one cache row per token serves as key (all 576 values) and value (the first 512) for every head. The generic
kernels read that row once per head; `fa_mqa` (`attention.slang`) runs the 16 heads of one query row as the 16 rows of a
cooperative-matrix tile, so each cache row is loaded once per workgroup and S = QKᵀ and O = PV run on tensor cores
(f16 in, f32 accumulate). The same kernel does prefill and split-K decode (grid z = key ranges, merged by `decode_combine`).
The expanded alternative (decompress K/V per head, then the regular flash kernel) was rejected after measuring that attention was
80 % of prefill time at 2048 tokens (795 of 990 ms), i.e. the absorbed kernel itself was the problem. `SHARPLIGHT_ATTN_MQA=0`
restores the previous kernels.

`sharplight MODEL --bench N --ctx N+256 --batch 512`, best prefill / median decode, tok/s:

| model | N | prefill before | prefill after | decode before | decode after |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite Q4_K_M | 512 | 5531 | 10831 | 280 | 327 |
| | 2048 | 2000 | 9675 | 175 | 311 |
| | 4096 | 1067 | 8462 | 152 | 297 |
| MiniCPM3-4B | 512 | 3790 | 12008 | 135 | 170 |
| | 2048 | 977 | 8375 | 134 | 168 |
| | 4096 | 493 | 6364 | 124 | 168 |

Kernel level (`BenchTests.LatentAttention`, DeepSeek shape): decode at 4096 keys 69.6 → 22.5 µs including the combine; a 512-row
prefill chunk at 3584 keys of history 712 µs. llama.cpp CUDA b11177 (`-fa 1`), measured in the same session on the shared GPU:
DeepSeek pp512 3906, pp2048 2652, tg128 108; MiniCPM3 pp512 2150, pp2048 1002, tg128 42. These are far below the figures quoted
earlier (12.9k / 9.9k prefill) and were not investigated further. MiniCPM3 decode is bound by its per-head expert-style matmuls,
not attention (~12 % of the step).

Quantised KV caches (`--kv-type q8_0|q4_0`): memory, speed and accuracy are in [kv-cache.md](kv-cache.md#24-quantised-kv-storage-implemented).

## 10. Paged KV cache: concurrency per GiB and c = 64 throughput (2026-11-13)

`sharplight serve MODEL --ctx 4096 --slots S [--kv-tokens N]`; `bench/loadgen.py -c 64 --per-worker 6 --ignore-eos` (251-token
prompts, 128 output tokens, 384 requests). Fixed: every slot reserves 4096 tokens. Paged: one pool of N tokens in 16-token
blocks, shared by 64 slots. VRAM = server process allocation; raw results in `bench/results/paged/`.

| model | KV layout | VRAM | sequences at once | agg tok/s | TTFT p50 ms | ITL mean ms |
|---|---|---|---|---|---|---|
| Llama-3.2-1B Q8_0 | fixed, 4 slots (16K) | 2100 MiB | 4 | 2248 | 3355 | 1.7 |
| | paged, 16K | 2100 MiB | 51 (234 preemptions) | 7115 | 160 | 5.9 |
| | fixed, 16 slots (64K) | 3636 MiB | 16 | 3957 | 1555 | 4.0 |
| | paged, 64K | 3636 MiB | 64 | 10187 | 40 | 5.7 |
| | fixed, 64 slots (256K) | 9780 MiB | 64 | 10498 | 35 | 5.6 |
| Mistral-7B Q4_K_M | fixed, 16 slots (64K) | 12696 MiB | 16 | 893 | 6342 | 17.3 |
| | paged, 24K | 7814 MiB | 64 | 2011 | 210 | 29.0 |
| | paged, 64K | 12678 MiB | 64 | 1976 | 212 | 29.5 |
| | fixed, 64 slots (256K) | 37254 MiB | 64 | 2065 | 209 | 28.2 |

In equal memory the paged cache runs 4x the sequences (here capped by the 64 slots) at 2.2-2.6x the throughput; it matches the
fixed 64-slot cache within 1-4 % with a quarter (Llama) to a fifth (Mistral, 24K pool) of the memory. With a shared ~800-token
system prompt (`--shared-prefix-words 800`) 64 contexts of ~980 tokens run in a 16K pool (Llama-1B, 2.1 GiB, 11438 tok/s, the
same as the 9.8 GiB fixed cache). Details and the shared-prefix table: [kv-cache.md](kv-cache.md#25-paged-blocks-admission-by-blocks-preemption-prefix-sharing-implemented).

## 11. MoE, block formats, int8 tensor-core prefill and fusion (2026-11-13)

`sharplight MODEL --bench 512 --ctx 768` (best of 3 process runs; decode = best of 64 steps after the 512-token prefill) against
`llama-bench -p 512 -n 128 -fa 1 -r 3` (b11177 CUDA) run right after it on the same, shared GPU. "before" is the branch state
when this work started (these models on master: gpt-oss-20b 6.4k / 213, ERNIE 6.6k / 202, Ministral-3 3B 14.6k / 245).

| model | metric | before | after | llama.cpp CUDA | after/CUDA |
|---|---|---|---|---|---|
| olmoe-q4km | pp512 | 22859 | 26791 | 26541 | 1.01x |
| olmoe-q4km | tg | 759 | 761 | 816 | 0.93x |
| ernie4.5-21b-a3b-q4km | pp512 | 8965 | 10405 | 10383 | 1.00x |
| ernie4.5-21b-a3b-q4km | tg | 318 | 320 | 365 | 0.88x |
| nemotron3-nano-30b-q4km | pp512 | 6353 | 9327 | 8464 | 1.10x |
| nemotron3-nano-30b-q4km | tg | 247 | 309 | 338 | 0.91x |
| gpt-oss-20b-mxfp4 | pp512 | 6687 | 11831 | 17432 | 0.68x |
| gpt-oss-20b-mxfp4 | tg | 220 | 310 | 345 | 0.90x |
| ministral3-3b-q4km | pp512 | 19889 | 19817 | 20666 | 0.96x |
| ministral3-3b-q4km | tg | 378 | 376 | 396 | 0.95x |

What changed:

* **Expert GEMMs on the dense tensor-core GEMM** (`gemm_cm2` with `MOE=1`): the expert-grouped tiles of `moe_route` drive
  the dense kernel (gathered activation rows, scattered output rows, 32/64/128-row tiles). It replaces `moe_gemm_cm`
  (`SHARPLIGHT_MOE_GEMM2=0` restores it): OLMoE +17 %, ERNIE +15 %, Nemotron +13 %, gpt-oss +42 % prefill.
* **Packed-half decoding of MXFP4, Q4_0, Q5_0 and plain Q8_0** in the f16 tensor-core GEMM (previously a per-value f32
  decode): E2M1 bits placed into half bits (exact, subnormals included) and one multiply by the block scale; the 4-bit
  formats through the exact 1024 + q trick. Word loads funnel-shifted to the 17/18/22/34-byte blocks (`load16_any`).
  gpt-oss expert GEMM 514 → 392 µs, Nemotron's Q5_0 / Q8_0 expert GEMMs 1.0 → 0.6 ms.
* **Integer-dot matvec for the 32-value block formats** (Q8_0 as stored, Q4_0, Q5_0, IQ4_NL, MXFP4; K % 32 == 0, so also
  gpt-oss's K = 2880): one block per chunk, Q4_0/Q5_0 offsets through the activation block sums, MXFP4 as the doubled int8
  values decoded with bytewise SIMD arithmetic (a dynamically indexed codebook cost 35 %). `SHARPLIGHT_MMVQ_BLOCK=0`
  disables it. gpt-oss decode 225 → 310, Nemotron (Q5_0 experts) 245 → 310 tok/s. Plain Q8_0 single rows stay on the f32
  matvec (int-dot is faster for gpt-oss's 2880-wide Q8_0 matrices but slower for Llama-1B and Qwen3-0.6B).
* **Elementwise groups merged along data flow** (`SHARPLIGHT_NO_GROUP_MERGE=1` disables): a group takes an earlier group of
  the same shape it reads when no unit in between reads that group or writes state. gpt-oss's clamped SwiGLU with the
  expert bias adds (5 kernels) becomes one: 749 → 653 dispatches per step.
* **int8 tensor-core GEMM (MMQ)** for prefill (`mmq.slang`, `DeviceFeatures.MatrixInt8`, q8_1 activations, 16x16x32 int8
  cooperative matrices, i2f via the 1.5 · 2^23 bias, Q6_K's 16-value scales through a second MMA on the upper half):
  Q8_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS. Measured at T=512 on the Mistral shapes (min of 50, TFLOPS, f16 path → MMQ):
  Q8_0 ≈ parity, Q4_K 140-170 → 90-160, Q5_K 122-154 → 77-140, Q6_K 106-152 → 60-112, IQ4_XS 56-83 → 82-152. The kernel
  needs 247-255 registers per thread (B-fragment loads and the per-block scaling dominate; ablations: pure MMA ≈ 876
  TOPS-equivalent, with scaling 519, without global loads 303). It is therefore on by default only for IQ4_NL / IQ4_XS
  (`SHARPLIGHT_MMQ=0|all|q4_k,...`; Mistral pp512: f16 9530, MMQ all 7630). The 27B Q8_0 prefill stays on the f16 GEMM
  (~177 TFLOPS); its remaining gap to llama.cpp (188 vs ~167 ms per 512 tokens) is mostly the delta rule (18 ms).
* `SHARPLIGHT_VK_PIPELINE_STATS=1` prints register count, shared memory and other driver statistics of every pipeline
  (`VK_KHR_pipeline_executable_properties`).

Precision: `IntDotAccuracyTests` (Families) teacher-forces ~90 positions of a fixed text through Llama-1B Q4_K_M,
Qwen3-0.6B Q8_0, OLMoE Q4_K_M, Mistral-7B Q4_K_M and gpt-oss-20b with and without `int_dot` (chunks of 1-3 tokens): the
same top-1 token at 94-100 % of positions, every flip at a reference margin of at most 1.2 % of the logit spread, maximum
logit difference 0.8-5.4 % of the spread (asserted: ≥ 95 % top-1 with near-ties within 2 % counted, ≤ 10 %).

Remaining gaps: gpt-oss prefill (0.68x: expert GEMM 392 µs × 72 at ~87 TFLOPS vs ~170 for the dense GEMM, the f16
activation conversion and partial expert tiles), MoE decode (0.88-0.93x: expert biases still gathered by a separate
`gather_axis` launch, and ~2 µs per small launch), and the int8 MMQ kernel's register pressure.

## 12. CPU provider: instruction-set tiers, integer dots, GEMM and tiled attention (2026-11-13)

Machine: Ryzen 9 9950X (Zen 5, 16 cores / 32 threads, AVX-512 F/BW/VL/DQ + VNNI + AVX-VNNI), 91 GB DDR5, Fedora 44,
.NET 11. The machine was shared with other agents and the desktop session during every run (load average 17-37 on 32
logical CPUs), so all numbers are noisy: medians move by ±20 %, the minimum / best values are the more stable ones. Every
process (Sharplight before and after, llama.cpp) was pinned to the 16 physical cores with `taskset -c 0-15` and used 16
threads. "before" is the master build at commit 13be2df (binaries copied before the work started).

```
# LLM, 3 process runs each (the CLI prints the best prefill and best / median of 64 decode steps after the prefill)
taskset -c 0-15 [SHARPLIGHT_CPU_THREADS=16] sharplight MODEL.gguf --cpu --bench 512
taskset -c 0-15 llama-bench -m MODEL.gguf -p 512 -n 64 -r 3 -t 16          # b11177 CPU build
taskset -c 0-15 llama-bench -m MODEL.gguf -p 0 -n 64 -d 512 -r 3 -t 16
# ONNX, 3 process runs of 30 timed runs each
taskset -c 0-15 [SHARPLIGHT_CPU_THREADS=16] sharplight-onnxrun MODEL.onnx --provider cpu --runs 30 [--shape pixel_values=-1x3x224x224]
# per tier: the same with SHARPLIGHT_CPU_ISA=scalar|v128|avx2|avxvnni|avx512|avx512vnni
# kernels: SHARPLIGHT_BENCH=1 dotnet tests/Sharplight.Cpu.Tests/bin/Release/net11.0/Sharplight.Cpu.Tests.dll -method "*Throughput"
```

### LLM, before → after vs llama.cpp CPU (tok/s, 16 threads)

| model | metric | before | after | llama.cpp CPU | after / llama.cpp |
|---|---|---|---|---|---|
| Qwen3-0.6B Q8_0 | prefill 512 (best of 3 runs) | 692 | 1912 | 1424 ± 101 | 1.34x |
| Qwen3-0.6B Q8_0 | decode after 512 (median, runs 1-3) | 42 / 42 / 48 | 57 / 56 / 52 | 55 ± 5 (tg64 @ d512) | ~1.0x |
| Qwen3-0.6B Q8_0 | decode (best step) | 53 | 65 | 69 ± 3 (tg64 @ d0) | |
| Llama-3.2-1B Q4_K_M | prefill 512 | 553 | 1099 | 1156 ± 36 | 0.95x |
| Llama-3.2-1B Q4_K_M | decode after 512 (median, runs 1-3) | 33 / 28 / 29 | 57 / 54 / 42 | 44 ± 3 (tg64 @ d512) | ~1.2x |
| Llama-3.2-1B Q4_K_M | decode (best step) | 38 | 59 | 54 ± 3 (tg64 @ d0) | |

The Sharplight decode is measured after the 512-token prefill (context 512), so it compares with llama.cpp's
`tg64 @ d512`.

### ONNX vision models, before → after (ms per image, batch 1, 16 threads; min and median of 30 runs, 3 process runs)

| model | before min | before median | after min | after median | min speed-up |
|---|---|---|---|---|---|
| resnet50-v1-12 | 18.7-19.4 | 23-63 | 13.4-14.8 | 15.3-17.9 | 1.4x |
| yolov8n (640x640) | 29.8-33.6 | 40-79 | 30.9-31.7 | 34-37 | 1.0x |
| convnext-tiny (224x224) | 29.1-31.3 | 34-44 | 22.2-23.2 | 23-24 | 1.3x |

The unpinned master build used all 32 logical CPUs through `Parallel.For` and was much slower on this loaded machine
(resnet50 min 82 ms, decode 33-40 tok/s); the spinning pool with a cap of one thread per core removes most of that.

### Per tier (`SHARPLIGHT_CPU_ISA`, this branch, 16 threads)

Best of 2 process runs (prefill tok/s, decode best-step tok/s) and resnet50 min of 30 runs. The machine load was higher
during this sweep (load average 26-37) than for the tables above, so compare the rows with each other, not with them.

| tier | Qwen3-0.6B Q8_0 prefill 512 | Qwen3 decode | Llama-3.2-1B Q4_K_M prefill 512 | Llama decode | resnet50 min ms |
|---|---|---|---|---|---|
| scalar | 67 | 30 | 29 | 16 | 192 |
| v128 | 61 | 28 | 29 | 14 | 159 |
| avx2 (+FMA, `vpmaddubsw` int dots) | 1094 | 62 | 594 | 50 | 17.1 |
| avxvnni (+ `vpdpbusd ymm`) | 1002 | 54 | 573 | 47 | 17.0 |
| avx512 (8x32 GEMM tile, 512-bit `vpmaddubsw`) | 1569 | 60 | 905 | 52 | 13.6 |
| avx512vnni (+ `vpdpbusd zmm`, default here) | 1682 | 61 | 916 | 52 | 16.8 |

The scalar and v128 tiers share the scalar 4x8 GEMM tile (there is no 128-bit GEMM kernel), so their prefill is ~25x
slower; their decode keeps half the speed because it is bound by memory. Decode differences between the vector tiers are
within the noise; VNNI shows up in the kernel benchmark (single thread q4_k matvec 10.8 GB/s with zmm VNNI).

### Kernels (single process, 16 threads unless noted, `BenchTests`)

| kernel | before | after |
|---|---|---|
| matvec 17408x5120, 1 thread, q4_k / q5_k / q6_k | 5.0 / 5.3 / 4.8 GB/s (float dequant + f32 dot) | 10.8 / 10.0 / 15.3 GB/s (int8 dots, VNNI) |
| matvec 17408x5120, 1 thread, q8_0 / q4_0 | 15.4 / 9.0 GB/s | 22 / 14.1 GB/s |
| 17408x5120 T=16, q4_k / q8_0 | 0.54 TFLOP/s (GEMM) | 1.27 / 1.10 TFLOP/s (int8 dots) |
| 17408x5120 T=128 / T=512 | 1.6-1.7 TFLOP/s | 2.1 / 2.8 TFLOP/s |
| attention decode, 512 cached keys (Qwen3 shape) | ~135 µs (general path) | 44 µs |
| attention decode, 4096 cached keys | | 160 µs |
| attention prefill T=512 | 22.4 ms | 0.82 ms |

### What changed

* **Hot spots first** (`SHARPLIGHT_CPU_PROFILE=1` prints per-op times split into decode / prefill at exit). Before:
  prefill = MatMul 63 % + attention 26 %, decode = MatMul 83 % + attention 15 %; resnet50 (once the pool was in) = conv
  70 %, Relu + Add 28 % (scalar Relu, Add walking 7..56-element rows).
* **Tiers** (`CpuIsa`, `CpuIsaLevel`): scalar, Vector128, AVX2+FMA, AVX-VNNI, AVX-512 (F/BW/VL/DQ), AVX-512 VNNI,
  detected once; kernels test JIT-constant `Use*` properties, so every untaken path is removed at JIT time. The device
  reports the tier (`isa` capability) and `int_dot` (the provider-neutral feature Vulkan also uses) from AVX2 up.
* **Integer dots for small batches** (`IntDot`, `MatMulOp` below 32 rows): activations quantized to int8 per 32 values,
  weights decoded per row to unsigned codes with a multiplier and an offset per 16 values (Q8_0, Q4_0/1, Q5_0/1,
  Q4_K/Q5_K/Q6_K, IQ4_NL/XS, MXFP4), int32 sums by `vpdpbusd` (zmm/ymm) or `vpmaddubsw` + `vpmaddwd` (with a sign trick
  for codes ≥ 128), each weight row decoded once for all activation rows. `int_dot` can be disabled per device or per
  compile like on Vulkan (the exact-greedy family tests do).
* **GEMM for ≥ 32 rows (≥ 8 without int dots)**: the quantized weight is dequantized panel by panel into the packed B
  operand of the 8x32 (AVX-512) / 6x16 (AVX2) microkernel, K chunks of 256; M blocks keep all threads busy.
* **Tiled flash attention**: GQA heads × tokens mapped to vector lanes (16 / 8), key blocks of 32 with an online softmax,
  keys split across tasks and merged for decode, a dot-product path for 1-2 tokens with software prefetch of the K/V rows.
* **Thread pool**: persistent spinning workers (200 µs spin, then blocking) instead of `Parallel.For` per op; per-job
  counters so no straggler takes work of the next job; nested or concurrent calls fall back to `Parallel.For`.
* **Elementwise**: 512-bit paths for binary ops, Relu/Neg/Square, exp/sigmoid/SiLU; binary ops merge trailing
  dimensions into long rows and use finer chunks for transposed operands.

### Diagnostic switches

| variable | effect |
|---|---|
| `SHARPLIGHT_CPU_ISA` | cap the tier: `scalar`, `v128`, `avx2`, `avxvnni`, `avx512`, `avx512vnni` (an unsupported cap selects the next lower supported tier) |
| `SHARPLIGHT_CPU_THREADS` | worker count (default: logical processor count; the `threads` device setting overrides it) |
| `SHARPLIGHT_CPU_POOL=0` | use `Parallel.For` instead of the spinning pool |
| `SHARPLIGHT_CPU_SPIN_US` | pool spin time after each job (default 200) |
| `SHARPLIGHT_CPU_GEMM_MIN_ROWS` | rows from which MatMul uses the GEMM (default 8, 32 with integer dots) |
| `SHARPLIGHT_CPU_NO_FUSION=1` | lower every node on its own: no bias / residual / activation epilogues on Conv and MatMul (§21) |
| `SHARPLIGHT_CPU_NO_PREPACK=1` | pack the B panels of static f32 MatMul weights on every run instead of once (§21) |
| `SHARPLIGHT_CPU_PROFILE=1` | per-op timing table on stderr at exit |

Correctness: `tests/Sharplight.Cpu.Tests` (1103 tests) passes under every `SHARPLIGHT_CPU_ISA` value; the integer-dot tiers
are compared with the scalar kernel (and the scalar kernel with a double-precision product of the decoded operands), the
decoders with ggml's dequantization, MatMul with int dots with a 1 % relative-error bound.

Remaining gaps: decode is DRAM-bound at ~30-40 GB/s per matvec (llama.cpp's level); Relu is not fused into the conv
epilogue (7 % of resnet50); the decomposed LayerNorm of convnext still runs as 5 elementwise passes; exp/sigmoid use
.NET's `Vector512.Exp` (0.3 ns/value) rather than a cheaper polynomial (yolov8n's SiLU is ~20 % of its time); no ARM64
(NEON / SDOT) integer-dot kernel and no 128-bit GEMM tile yet (the Vector128 tier is the portable fallback).

## 13. FP8 checkpoints, the fp8 tensor-core GEMM and the int8 GEMM rework (2026-11-13)

Conditions: the GPU was shared with another application and with other benchmark agents for the whole session, so end-to-end
`--bench` numbers fluctuate by up to 40 % between back-to-back runs. The tables therefore report kernel minima (min of 50,
`BenchTests.MatMulPrefill`, T = 512) and `SHARPLIGHT_VK_PROFILE=1` prefill sums of per-kernel minimum × launch count
(contention-free GPU time; host overhead excluded), with the best end-to-end runs as a cross-check.

### Matmul kernels, T = 512 (TFLOPS, min of 50)

| shape (K→N) | f8e4m3_b128: f16 GEMM | f8e4m3_b128: fp8 GEMM | f8e4m3_b32: fp8 GEMM | Q8_0: f16 GEMM | Q8_0: int8 MMQ | Q4_K: f16 GEMM | Q4_K: MMQ before → after |
|---|---|---|---|---|---|---|---|
| 5120→17408 | 158-162 | **258-259** | 217-221 | 170 | 211 | 187 | 158 → 184 |
| 17408→5120 | 162 | **252** | 192-193 | - | - | - | - |
| 4096→14336 | 148 | **232-234** | 216-218 | 158 | 187 | 149 | 124 → 150 |

### Qwen3.8-27B prefill, 512 tokens

| model / path | GPU time (Σ min × count) | ≙ tok/s | best `--bench` pp512 | pp2048 (best) |
|---|---|---|---|---|
| Qwen3.8-27B-FP8, f16 GEMM (`SHARPLIGHT_FP8_GEMM=0`) | 200.6 ms (matmuls 170.0) | 2552 | 2226 | 1745 |
| Qwen3.8-27B-FP8, **fp8 GEMM** (default) | **144.1 ms (matmuls 111.9)** | **3554** | **3025** | **2626** |
| Qwen3.8-27B Q8_0, f16 GEMM (`SHARPLIGHT_MMQ=0`) | 184.4 ms (matmuls 154.0) | 2777 | 2023 | 1280 |
| Qwen3.8-27B Q8_0, int8 MMQ (default) | 185.2 ms (matmuls 154.6) | 2765 | 1973 | 2200 |
| llama.cpp CUDA Q8_0 (section 2) | | | 3058 | 3030 |
| vLLM FP8 (section 2) | | | 6348 | 7425 |

Nemotron-3-Nano-30B-A3B-FP8 (dense projections on the fp8 GEMM, experts on the f16 MoE GEMM): 54.6 → 51.5 ms GPU time
(9380 → 9935 tok/s), best `--bench` pp512 5779 → 7282. Decode is unchanged (matvecs; 27B FP8 ≈ 38-42 tok/s like Q8_0).

The fp8 GEMM makes the FP8 checkpoint the fastest 27B prefill in Sharplight: about 1.4x the f16 path and, by GPU time, above
llama.cpp CUDA's Q8_0 (3554 vs 3058 tok/s), but still about 0.56x of vLLM's FP8 (CUTLASS block-scaled FP8 GEMMs at a
higher fraction of the 4th-gen tensor-core peak; Vulkan exposes only the 16x16x32 subgroup shape with an f32 accumulator).
Of the remaining 144 ms, 112 ms are matmuls (≈ 250 TFLOPS), 31 ms the delta rule, conv1d and norms.

### What changed

* **FP8 weights** (merged from the FP8 track): `f8e4m3`, `f8e5m2`, block-scaled `f8e4m3_b32` / `f8e4m3_b128`; vLLM /
  DeepSeek block `weight_scale_inv`, ModelOpt / compressed-tensors per-tensor, per-channel and per-block `weight_scale`
  converted to block scales on import (model-support 2.4). Accuracy (Families `Fp8CheckpointTests`, teacher-forced,
  `SHARPLIGHT_TEST_LARGE=1`, prefill chunk on the fp8 GEMM via `SHARPLIGHT_MMQ_MIN_ROWS=64`): Qwen3.8-27B-FP8 vs Q8_0 same
  top-1 72/76, all 76 within 2 % of the logit spread (f16 path: 71/76); Nemotron FP8 vs Q4_K_M 70/77, 77/77.
* **fp8 tensor-core GEMM** (`gemm_fp8.slang`, device feature `matrix_fp8` = VK_EXT_shader_float8 cooperative matrices
  with the 16x16x32 f8E4m3 x f8E4m3 + f32 shape; `fp8` = shaderFloat8). Activations are quantized to E4M3 with one f32
  scale (amax / 448) per 128 values (`quantize_fp8`, round to nearest even, one launch shared by all matmuls reading the
  same rows). Each weight block (128 or 32 values) is multiplied into a fresh f32 fragment and added to the accumulator once
  with `activation scale × weight scale`, so per-block scaling costs one FMA per accumulator element per block (vs 3-4 for
  the int8 kernel). Same 64/128-token tiles, split-K and accumulator-layout probe as the int8 GEMM (the generic per-element
  path when the probe does not find the paired layout). Used for prefill batches of 65+ rows of `f8e4m3_b128` /
  `f8e4m3_b32` weights with K % 128 == 0; `SHARPLIGHT_FP8_GEMM=0` or `--disable-features matrix_fp8` use the f16 GEMM.
  Tests: `Fp8GemmTests` (exact E4M3-representable activations match the f32 oracle to 2e-3; random activations within the
  E4M3 rounding bound; large grid; feature disabled), also on RADV and llvmpipe (no `matrix_fp8`: f16/scalar fallback).
* **int8 GEMM (MMQ) rework** (`mmq.slang`): 128-value K steps (64 for Q6_K) staged per barrier instead of 32, a rolled
  block loop (`[MaxIters]`, smaller code, fewer registers), paired loads of two Q4_K / Q5_K 32-value units, shared-memory
  scale arrays instead of an index table. Q4_K +20-25 % at kernel level, now at parity with the f16 GEMM (not faster: the
  3-4 FP ops of per-32-block scaling per accumulator element cost about as much as the MMAs; ablation without scaling
  and global loads ≈ 373-451 TFLOPS-equivalent). Q8_0 is 11-24 % faster than the f16 GEMM on 4096+ x 4096+ matrices and
  is now on by default there (`MmqLargeOnlyFormats`; smaller Q8_0 matrices like Llama-1B's stay on the f16 GEMM, where MMQ
  was slower). Q4_K / Q5_K / Q6_K stay opt-in (`SHARPLIGHT_MMQ=q4_k,...`); Mistral-7B Q4_K_M pp512 f16 ≈ 8000 vs MMQ ≈ 7860
  (contended), so the 11229 tok/s llama.cpp CUDA target is not reached by MMQ. `SHARPLIGHT_MMQ_BK` overrides the K step.

### What Vulkan exposes for FP8 / FP4 (NVIDIA 610.57.04, RTX PRO 6000 Blackwell)

* `VK_KHR_cooperative_matrix` (subgroup scope): f16, bf16, i8/u8 and **f8E4m3 / f8E5m2** inputs, 16x16x32 for the 8-bit
  types, f16 or f32 accumulators. This is what `matrix_int8` / `matrix_fp8` use.
* `VK_NV_cooperative_matrix2` (workgroup scope, `matrix_workgroup`): f16 / bf16 / int8 / fp8 up to 32x32x32 with 256
  invocations.
* `VK_NV_cooperative_vector`: f16, int8 and fp8 E4M3 / E5M2 inputs (f16 results).
* **No FP4 / E2M1** in any of them and no float4 extension: the Blackwell block-scaled FP4 (NVFP4 / MXFP4) tensor-core
  instructions are not reachable from Vulkan. `--devices` prints the shape lists (`coopmat2_shapes`, `coopvec_types`).
  NVFP4 and MXFP4 weights therefore keep the existing paths: E2M1 values decoded to packed halves for the f16 tensor-core
  GEMM in prefill, the f32 matvec (NVFP4) or the int-dot matvec (MXFP4) in decode. An int8 or fp8 GEMM for NVFP4 would need
  its 16-value scales applied per half of every 32-deep MMA, the same per-block scaling overhead that keeps the K-quant MMQ
  at parity, so it was not added.

## 14. MoE prefill: sorted f16 expert activations, aligned MXFP4 experts, in-place bias rows (2026-11-13)

What changed (each behind a switch that restores the section-11 path, which is also the fallback without `matrix_f16`):

* **Expert-sorted f16 activations** (`moe_gather_x16`, `SHARPLIGHT_MOE_X16=0`): after `moe_route`, one small launch gathers the
  routed rows of x into expert-sorted f16 rows (shared by gate and up, which read the same x with the same routing; down gathers
  its per-slot input once). The expert-grouped `gemm_cm2` (`X_SORTED`) reads dense f16 rows instead of gathering and converting
  f32 rows in each of its N/128 column tiles. Bit-identical logits (same f16 rounding).
* **Aligned MXFP4 experts** (`WeightRepack.Mxfp4Split`, type `mxfp4r`, `SHARPLIGHT_REPACK_MXFP4=0`): rank-3 MXFP4 tensors with
  N % 16 == 0 are uploaded in 16-row groups [codes of 16 rows][E8M0 scales of 16 rows], so every block's codes are one aligned
  16-byte load (gemm_cm2, mmvq, generic load8). Small gain (gpt-oss expert GEMM ~3 %).
* **Row gathers read in place** (`SHARPLIGHT_NO_GATHER_FOLD=1`): a GatherAxis on axis 0 of a [rows, N] table consumed by one
  plain fused elementwise group (gpt-oss's per-expert biases `Add(MatMulId, GatherAxis(bias, ids))`) is read as
  `table[ids[..], n]` inside that group's kernel: 72 `gather_axis` launches per gpt-oss step fewer (prefill and decode).

GPU timestamps (`SHARPLIGHT_VK_PROFILE=1`, min per launch over 3 process runs, pp512, old → new, same build with the switches):

| model | expert GEMMs (sum of per-launch minima) | per expert GEMM |
|---|---|---|
| gpt-oss-20b-mxfp4 | 29.2 → 21.9 ms | 406 → 305 µs (85 → 113 TFLOPS) |
| olmoe-q4km | 20.2 → 15.7 ms | |
| ernie4.5-21b-a3b-q4km | 26.4 → 18.8 ms | |
| nemotron3-nano-30b-q4km | 35.1 → 28.0 ms | |
| qwen3.6-35b-a3b-q4km | 7.1 → 6.0 ms | |
| gpt-oss-20b-mxfp4 pp2048 (4 × 512) | 28.1 → 19.4 ms per 512 chunk | |

Wall clock (`sharplight MODEL --bench 512 --ctx 768`, best of 3 alternating process runs). These runs shared the GPU with another application
and another agent's model (90 % utilization at the 300 W cap before starting), so absolute numbers are low and noisy; the
section-11 column was measured on a quieter GPU. The first gpt-oss row is from a quieter period.

| model | metric | old path | new | section 11 | llama.cpp CUDA |
|---|---|---|---|---|---|
| gpt-oss-20b-mxfp4 | pp512 (quieter) | 9154 | 11079 | 11831 | 17432 |
| gpt-oss-20b-mxfp4 | pp512 | 5562 | 8961 | 11831 | 17432 |
| gpt-oss-20b-mxfp4 | pp2048 | 5275 | 6435 | | |
| gpt-oss-20b-mxfp4 | tg | 283 | 288 | 310 | 345 |
| olmoe-q4km | pp512 | 12815 | 23190 | 26791 | 26541 |
| ernie4.5-21b-a3b-q4km | pp512 | 7813 | 8645 | 10405 | 10383 |
| nemotron3-nano-30b-q4km | pp512 | 6789 | 5454 (noise; GEMMs −20 %) | 9327 | 8464 |
| qwen3.6-35b-a3b-q4km | pp512 | 5783 | 7323 | | |

Routing statistics (debug-printed counts, gpt-oss, random tokens): experts are very unevenly loaded (per layer ~8.5 of 32 experts
empty, a few with 250-510 of the 2048 items), so 64-row tiles are 67 % full (32-row: 82 %, 128-row: 48 %). Skipping the MMA of
empty warps did not change the time: the expert GEMM is bound by the per-tile weight decode and loads, not by MMA, so tighter
packing alone would gain little (estimated ~9 % of the GEMM for 64-row tiles with 32-row remainder tiles, minus a barrier between
two launches). Ablations that change weight or activation values also change the routing and hence the tile count; they are not
valid timing experiments for MoE.

Remaining: the f16 expert GEMM is weight-decode bound for small per-expert row counts (Nemotron: 128 experts, ~24 rows each,
47-55 TFLOPS). The int8 path (q8_1 activations against the stored codes, no decode to f16) is the next step: the sorted gather is
the natural place to quantize the routed rows to q8_1 once, and an MMQ kernel with the `MOE` tile scheme of gemm_cm2 would read
dense sorted q8_1 rows and scatter outputs through the routing scratch.

## 15. Qwen3.8-27B decode across engines and backends (2026-11-13)

Output tokens/s after a 16-token prompt on an otherwise idle machine (load average 2 before the run; the CPU rows load it
themselves). Sharplight: `sharplight MODEL [--cpu] --bench 16 --ctx 512`, median of 64 decode steps, median over 3 process runs.
llama.cpp b11177: `llama-bench -m MODEL -p 0 -n 128 -r 3 -ngl 99 -fa 1` (CUDA and Vulkan builds), CPU build with `-n 64 -t 16`
(mean ± sd). CPU runs of both engines are pinned to the 16 physical cores (`taskset -c 0-15`). Script: the loop in
`bench/run_27b_engines.sh`.

| Engine / backend | Q4_K_M (16.8 GB) | Q8_0 (29.0 GB) |
|---|---:|---:|
| **Sharplight Vulkan** | **65.6** | 40.7 |
| llama.cpp CUDA | 60.4 ± 2.8 | 41.9 ± 1.2 |
| llama.cpp Vulkan | 59.4 ± 2.0 | 41.5 ± 1.0 |
| **Sharplight CPU** | **3.03** | **1.80** |
| llama.cpp CPU | 2.89 ± 0.03 | 1.69 ± 0.00 |

Per-run Sharplight medians: Vulkan Q4_K_M 66.3 / 58.6 / 65.6 (best step 69.6), Q8_0 40.7 / 40.6 / 41.7 (best 46.4); CPU Q4_K_M
3.03 / 3.03 / 2.97, Q8_0 1.84 / 1.80 / 1.72. GPU decode is bandwidth bound: Q4_K_M 14.2 ms per step for 15.6 GiB of weights and
state (about 1.2 TB/s), Q8_0 21.4 ms for 26.8 GiB. CPU decode is bound by DDR5 bandwidth (Q4_K_M about 319 ms per step,
about 52 GB/s) for both engines.

## 16. Re-baseline against vLLM on identical weights, and where the gap is (2026-11-13/14)

Measurement only (nothing under `src/` or `tests/` changed). Goal: a like-for-like comparison with vLLM after the work of
sections 8-15, a kernel-level attribution of Sharplight's batched step, and a ranked list of levers. Raw results:
[`bench/results/s16/`](../bench/results/s16) (`profiles/` for BatchBench and per-kernel profiles, `disturbed/` for runs that
were repeated because another application was using the GPU).

### 16.1 Setup

* Same machine and engine builds as section 1 (RTX PRO 6000 Blackwell Max-Q, 300 W cap; vLLM 0.33.0 with FLASH_ATTN, CUDA
  graphs, async scheduling, `VLLM_USE_FLASHINFER_SAMPLER=0`, prefix caching off; llama.cpp b11177). Sharplight: `main` at
  ed1dd8f, Release build.
* **Identical weights** for Sharplight and vLLM: `Qwen/Qwen3.8-27B-FP8` (block-scaled E4M3; Sharplight loads the vLLM
  checkpoint natively), `Llama-3.2-1B-Instruct` bf16 and `OLMoE-1B-7B-0924-Instruct` bf16 (HF safetensors). llama.cpp cannot
  load the FP8 checkpoint and ran the 27B as Q8_0 GGUF; Sharplight on that GGUF is listed for reference.
* **Conditions.** The GPU sat at its 300 W limit and 85-89 °C during the long serving runs (SM clock 1.75-2.3 GHz), so absolute
  numbers are 10-20 % below the cooler runs of section 2 for *both* engines (vLLM 27B decode 48.5 → 39.5 tok/s). Each comparison
  below was measured in the same period. Another application (~1.2 GB) used the GPU from the vLLM small-model single-stream runs to
  the end of the first Sharplight 27B serving pass; everything it overlapped was repeated afterwards (old files in
  `disturbed/`) except the OLMoE serving rows and the vLLM MTP single-stream rows, which are marked. Another agent's processes
  (up to 16 GB, mostly idle) appear in a few snapshots (`*.gpu.txt`).
* **Robust numbers:** single stream = median over 3 process runs (Sharplight, plus GPU ms per step from timestamps) or 5
  repetitions (vLLM, llama-bench); serving = median over 2-3 rounds with the server's `/metrics`; kernel attribution =
  per-kernel GPU timestamps (`SHARPLIGHT_VK_PROFILE`) and the minimum step GPU time over 30 steps (BatchBench).

### 16.2 Commands

```bash
export VLLM_VENV=~/.venvs/vllm LLAMA_DIR=<folder with runtime.llama-b11177-*>
dotnet build Sharplight.slnx -c Release && dotnet build bench/BatchBench -c Release
# single stream: Sharplight (--bench N, N = 16 512 2048 4096), vLLM (offline API), llama-bench
bench/s16_single.sh sharplight qwen27b-fp8 <dir of Qwen3.8-27B-FP8>
bench/s16_single.sh vllm qwen27b-fp8 <dir of Qwen3.8-27B-FP8> --gpu-util 0.5
bench/s16_single.sh vllm qwen27b-fp8-mtp3 <dir> --gpu-util 0.5 --spec-method mtp --spec-tokens 3 --points tg
bench/s16_single.sh llama-cuda13 qwen27b-q8 <Qwen3.8-27B-Q8_0.gguf>
NS="16 512 2048 3800" bench/s16_single.sh sharplight olmoe-bf16 <OLMoE dir>      # 4096 positions only
# serving (one server per engine, c = 1 4 16 64, ~250-token prompts, 128 tokens with ignore_eos, /metrics deltas)
ROUNDS=2 bench/s16_serve.sh sharplight qwen27b-fp8 <dir> "1 4 16 64"
ROUNDS=2 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
NATURAL=1 ROUNDS=1 bench/s16_serve.sh vllm qwen27b-fp8-mtp3 <dir> "1 4 16 64" --gpu-memory-utilization 0.5 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
bench/s16_serve.sh vllm llama1b-bf16-noasync <dir> "1 16 64" --gpu-memory-utilization 0.3 --no-async-scheduling   # or --enforce-eager
# kernel attribution
bench/BatchBench/bin/Release/net11.0/BatchBench <dir> --ctx 2048 --modes greedy,no-logits,logits --profile 16,64 --profile-mode greedy
SHARPLIGHT_VK_NO_BARRIERS=1 bench/BatchBench/bin/Release/net11.0/BatchBench <dir> --ctx 2048 --modes greedy --sizes 1,16,64
SHARPLIGHT_VK_PROFILE=1 samples/Sharplight.Cli/bin/Release/net11.0/sharplight <dir> --bench 2048 --ctx 2304 --batch 2048
python3 bench/profile_categories.py bench/results/s16/profiles/batch_qwen27b-fp8.txt
# tables, then strip home paths / process command lines from the raw files before committing them
python3 bench/s16_summarize.py; python3 -I bench/sanitize_results.py bench/results/s16
```

Script changes: `loadgen.py --metrics` stores the change of the server's `/metrics` over the timed part (Sharplight steps, step
time and rows; vLLM iterations and spec-decode counters), `--natural` sends short natural-language instructions (needed for
speculative decoding); `vllm_single.py` gained `--spec-method/--spec-tokens` (reports acceptance), `--text-prompt` and
`--points`; BatchBench loads HF directories, takes `--modes` and several `--profile` sizes, and prints the dispatch / barrier
count and the first-step (recording) time per batch size; `gpu_check.sh` also records memory clock, temperature and P-state.

### 16.3 Single stream

Tokens/s; Sharplight decode = median step at depth N (16 for "d0"); vLLM tg = (t(129) − t(1)) / 128 through the offline API.

| model | metric | Sharplight | vLLM | SL / vLLM | Sharplight Q8_0 | llama.cpp CUDA Q8_0 |
|---|---|---|---|---|---|---|
| Qwen3.8-27B FP8 | pp512 | 2822 | 5171 | 0.55x | 2517 | 2847 |
| | pp2048 | 3004 | 5944 | 0.51x | 2578 | 2569 |
| | tg @d0 | 39.2 (GPU 25.3 ms/step) | 39.5 | 0.99x | 39.9 | 46.0 |
| | tg @d4096 | 39.0 | 37.7 | 1.03x | 40.9 | 43.8 |
| Llama-3.2-1B bf16 | pp512 | 54104 | 59255 | 0.91x | | |
| | pp2048 | 46352 | 74524 | 0.62x | | |
| | pp4096 | 40499 | 72883 | 0.56x | | |
| | tg @d0 | 525 | 389 | 1.35x | | |
| | tg @d4096 | 481 | 379 | 1.27x | | |
| OLMoE-1B-7B bf16 | pp512 | 29583 | 33738 | 0.88x | | |
| | pp2048 | 26443 | 59119 | 0.45x | | |
| | pp3800 | 22348 | 60020 | 0.37x | | |
| | tg @d0 | 495 | 402 | 1.23x | | |
| | tg @d3800 | 429 | 225 | 1.91x | | |

Sharplight prefill above 512 tokens runs in 512-row chunks (`--batch 512`); one 2048-row chunk of the 27B gives 3127 tok/s, the
same. Single-stream decode is at the bandwidth roof on both engines (the 27B's fused FP8 gate/up matvec moves 178 MB in 110 µs =
1.61 TB/s, the bf16 lm-head 1.70 TB/s); vLLM's small-model decode numbers include its Python per-step overhead (the serving
c = 1 rows below are the fairer comparison). llama.cpp's Q8_0 decode (46.0) was 15 % faster than both in this period (section 15
measured 41.9 vs 40.7 on a cooler GPU).

### 16.4 Serving

`loadgen.py`, closed loop, ~250-token random-word prompts, 128 output tokens (ignore_eos), temperature 0. Sharplight
`--slots 64 --ctx 512` (contiguous KV), vLLM `--max-num-seqs 64 --max-model-len 512`. "step ms" = mean of Sharplight's
`sharplight_step_duration_seconds` (wall time of one decoder step: host preparation, GPU, read-back); its sum covered 99-100 % of
the wall time at every point, so the engine thread is never idle between steps. Rows/step include prefill rows (vLLM:
`iteration_tokens_total`, 170-185 at c = 64). Median of 2-3 rounds (min..max).

| model | conc | Sharplight agg tok/s | vLLM agg tok/s | SL / vLLM | SL TTFT p50 / p95 ms | vLLM TTFT p50 / p95 | SL ITL mean / p99 ms | vLLM ITL mean / p99 | SL step ms (rows) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3.8-27B FP8 | 1 | 36 (35..36) | 41 (38..43) | 0.88x | 97 / 109 | 82 / 140 | 27.3 / 31.1 | 24.2 / 31.9 | 27.8 (2.9) |
| | 4 | 55 (55..55) | 144 (142..146) | **0.38x** | 265 / 290 | 202 / 237 | 71.2 / 81.9 | 26.5 / 41.2 | 71.9 (11.7) |
| | 16 | 295 (292..298) | 486 (474..499) | 0.61x | 344 / 1135 | 656 / 819 | 50.5 / 174 | 27.8 / 47.4 | 52.9 (46.0) |
| | 64 | 572 (571..573) | 1056 (1045..1067) | 0.54x | 614 / 5293 | 1993 / 2619 | 98.5 / 215 | 44.7 / 74.9 | 101.7 (171.5) |
| Llama-3.2-1B bf16 | 1 | 445 (417..447) | 400 (354..404) | 1.11x | 8 / 10 | 9 / 10 | 2.20 / 2.9 | 2.45 / 2.9 | 2.23 (2.9) |
| | 4 | 903 (860..903) | 1359 (1309..1377) | 0.66x | 19 / 22 | 15 / 18 | 4.33 / 6.0 | 2.85 / 4.3 | 4.37 (11.7) |
| | 16 | 4170 (3807..4179) | 4743 (4234..4743) | 0.88x | 22 / 76 | 46 / 49 | 3.59 / 11.6 | 3.08 / 4.5 | 3.73 (46.0) |
| | 64 | 8964 (8674..9019) | 13595 (12559..13722) | 0.66x | 40 / 360 | 130 / 179 | 6.19 / 15.3 | 3.78 / 16.9 | 6.44 (171.5) |
| OLMoE bf16 (GPU shared) | 1 | 386 (356..386) | 346 (345..356) | 1.12x | 16 / 18 | 15 / 18 | 2.49 / 3.6 | 2.79 / 4.1 | 2.57 (2.9) |
| | 4 | 548 (449..575) | 663 (648..696) | 0.83x | 29 / 41 | 30 / 40 | 7.15 / 15.3 | 5.80 / 8.7 | 6.87 (10.6) |
| | 16 | 1336 (1324..1399) | 1629 (1535..1636) | 0.82x | 46 / 147 | 72 / 85 | 11.5 / 24.3 | 9.33 / 12.9 | 11.5 (46.0) |
| | 64 | 3311 (3203..3381) | 4011 (3986..4154) | 0.83x | 82 / 727 | 203 / 263 | 17.3 / 31.6 | 14.5 / 29.3 | 17.3 (171.5) |

Compared with section 3 the 27B gap at c = 4 is unchanged (0.37x then, 0.38x now: the step at 4 rows costs 2.6x the 1-row
step), while c = 16 / 64 improved from 0.37x / 0.41x to 0.61x / 0.54x and the small models from 0.22-0.51x to 0.66-0.88x.
Sharplight has the lower TTFT p50 at 16-64 streams (new prompts join the running batch at once) but a much longer TTFT p95 and
ITL p99 tail (a 512-row prefill chunk shares the step with the decodes).

**What vLLM's host machinery is worth** (same harness, features switched off; Llama-1B baseline from the table above):

| vLLM variant | Llama-1B c=1 | c=16 | c=64 | 27B FP8 c=1 | c=16 | c=64 |
|---|---|---|---|---|---|---|
| default (CUDA graphs + async scheduling) | 400 | 4743 | 13595 | 41 | 486 | 1056 |
| `--no-async-scheduling` | 311 (−22 %) | 3056 (−36 %) | 8041 (−41 %) | | | |
| `--enforce-eager` (no CUDA graphs) | 105 (−74 %) | 1517 (−68 %) | 5365 (−61 %) | 26 (−37 %) | 401 (−17 %) | 874 (−17 %) |

Sharplight already has the equivalent of CUDA graphs (Vulkan command buffers recorded once per step shape and replayed), and
its host work per step is small (16.6), so neither of these is where its gap comes from.

### 16.5 Speculative decoding: vLLM MTP on Qwen3.8-27B-FP8

The checkpoint ships one MTP layer (`mtp.*`, FP8); vLLM 0.33 runs it with `speculative_config={"method": "mtp",
"num_speculative_tokens": k}` (the layer is reused for k > 1). Natural-language prompt (an essay instruction), temperature 0.
The single-stream rows ran while another application used the GPU (baseline 38.4 instead of 39.5), so read the ratios.

| k | single stream tok/s @d47 (speedup) | acceptance | mean accepted length | @d4096 (speedup) | acceptance |
|---|---|---|---|---|---|
| off | 38.4 | | | 38.2 | |
| 1 | 60.5 (1.58x) | 0.84 | 1.84 | 57.2 (1.50x) | 0.89 |
| 2 | 70.6 (1.84x) | 0.70 | 2.39 | 71.6 (1.87x) | 0.79 |
| 3 | 74.7 (1.95x) | 0.59 | 2.76 | 81.3 (2.13x) | 0.70 |

Serving (natural prompts of ~25 tokens, 128 output tokens, one round; ITL is per SSE chunk, which carries several tokens with
speculation, so only the aggregate is comparable):

| conc | off agg tok/s | MTP k=1 (acceptance) | speedup | MTP k=3 (acceptance) | speedup |
|---|---|---|---|---|---|
| 1 | 42 | 61 (0.78) | 1.45x | 76 (0.56) | 1.81x |
| 4 | 147 | 234 (0.80) | 1.59x | 278 (0.57) | 1.89x |
| 16 | 554 | 758 (0.80) | 1.37x | 843 (0.58) | 1.52x |
| 64 | 1370 | 1261 (0.81) | 0.92x | 965 (0.58) | 0.70x |

MTP is worth 1.5-1.9x up to 16 streams and loses at 64, where the step is no longer weight-bandwidth bound and the extra
verification and draft rows cost more than the accepted tokens save. In Sharplight the serving engine disables speculation for this model today: `MultiSequenceDecoder.CanTruncate` is false
for any model with recurrent state (the gated-delta-net and conv states of the 48 linear-attention layers cannot be rolled back
to the accepted prefix), so MTP needs per-draft-position state snapshots (or recomputation of the rejected tail) first.

### 16.6 Kernel attribution of the batched step

`BatchBench MODEL --ctx 2048` (64 slots, 256 tokens of context each, device greedy sampling = the serving path). Step: wall
min / GPU min (ms) over 30 steps; "first" = wall time of the first step at that batch size (includes recording the command
buffers for the new shape).

| batch | 27B FP8 | 27B Q8_0 (int-dot matvec ≤ 16 rows) | Llama-1B bf16 | OLMoE bf16 | 27B FP8 first step |
|---|---|---|---|---|---|
| 1 | 23.9 / 23.7 | 24.0 / 23.6 | 2.07 / 1.88 | 2.10 / 2.04 | 62.6 |
| 2 | 39.3 / 38.9 | 20.5 / 20.2 | 2.96 / 2.72 | 3.32 / 3.21 | 81.1 |
| 4 | 62.5 / 61.7 | 24.1 / 23.9 | 3.74 / 3.61 | 5.00 / 4.89 | 81.9 |
| 8 | 118.1 / 117.8 | 29.2 / 28.8 | 6.38 / 6.16 | 6.36 / 6.04 | 135.6 |
| 16 | 45.7 / 45.3 | 42.4 / 41.6 | 3.04 / 2.82 | 6.35 / 6.19 | 58.4 |
| 32 | 47.0 / 46.6 | 52.2 / 51.2 | 3.20 / 3.10 | 8.07 / 7.80 | 60.7 |
| 64 | 66.2 / 65.8 | 65.8 / 65.6 | 3.89 / 3.69 | 10.41 / 10.15 | 82.8 |
| dispatches / barriers per step | 901 / 596 (1-8 rows), 2022 / 1653 (16-64) | 901 / 596 (1), 1158 / 853 (2-16), 2022 / 1653 (32-64) | 213 / 148, 470 / 389 | 357 / 260, 566 / 453 | |

Profile shares (sum of per-launch GPU time over 20 steps, `bench/profile_categories.py`; per-launch timestamps serialize the
launches, so small kernels weigh more here than in the real step):

| category | 27B FP8 b16 | 27B FP8 b64 | Llama-1B b16 | Llama-1B b64 | OLMoE b16 | OLMoE b64 |
|---|---|---|---|---|---|---|
| dense GEMM / matvec (incl. lm-head) | 80.4 % | 66.4 % | 62.7 % | 54.4 % | 13.2 % | 8.8 % |
| GEMM glue (f16 activation copies, split-K reduce) | 8.6 % | 7.3 % | 18.9 % | 19.4 % | 8.0 % | 6.7 % |
| MoE (expert GEMMs, routing, gathers) | | | | | 61.7 % | 50.3 % |
| delta rule + conv1d | 5.0 % | 19.0 % | | | | |
| attention (split-K decode + combine) | 2.0 % | 4.0 % | 10.1 % | 17.9 % | 10.6 % | 28.2 % |
| norms / elementwise / rope / cache writes | 3.9 % | 3.3 % | 7.7 % | 7.7 % | 6.4 % | 5.9 % |
| sampling | 0.1 % | 0.1 % | 0.5 % | 0.5 % | 0.1 % | 0.1 % |

Findings:

* **Small-batch cliff for 16-bit and FP8 weights.** 2-8 rows of FP8 (and bf16) weights take the row-batched f32 matvec, whose
  cost grows linearly with rows: the 27B step at 4 rows is 2.6x and at 8 rows 4.9x the 1-row step, and 8 rows are 2.6x slower
  than 16 rows. The Q8_0 GGUF of the same model on the integer-dot matvec (weights read once for up to 16 rows) does 4 rows in
  24.1 ms and 8 in 29.2 ms. This is the whole c = 4 gap (serving step 71.9 ms vs vLLM ITL 26.5 ms).
* **Skinny tensor-core GEMM below the bandwidth roof.** 17-64 rows (and 9-16 for 16-bit / FP8) use `gemm_cm2` split-K with
  32/64-row tiles: the FP8 `[5120→17408]` reads its 89 MB at 0.93 TB/s at 16 rows (96 µs) and 0.70 TB/s at 64 rows (127 µs),
  against 1.61 TB/s for the 1-row matvec. Matmuls take 35.2 ms (b16) and 46.3 ms (b64) of GPU time per step, against ~18 ms for
  reading the 29 GB of weights once at 1.6 TB/s.
* **Barriers and small launches.** `SHARPLIGHT_VK_NO_BARRIERS=1` (wrong results, timing only; launches may then overlap) cuts
  the 27B step 45.3 → 33.9 ms at b16 (−25 %) and 65.8 → 52.7 ms at b64 (−20 %), nothing at b1; Llama-1B 1.88 → 1.51 (b1,
  −20 %), 2.82 → 1.67 (b16, −41 %), 3.69 → 2.23 (b64, −40 %). The batched graphs have 2022 / 1653 (27B) and 470 / 389
  (Llama-1B) dispatches / barriers per step; every matmul on the 9-64-row path adds an f16 activation copy (`to_f16`) and a
  split-K reduction, ~500 tiny launches of each per 27B step (7-9 % of the 27B profile, 19 % of Llama-1B's).
* **Delta rule** at 64 rows costs 223 µs per layer (10.7 ms per step, 19 %), but it moves the 64 × 3.1 MB f32 state in and out
  (403 MB per layer) at 1.81 TB/s, i.e. at the bandwidth roof. vLLM keeps the same f32 state (`mamba_ssm_dtype`), so this is not
  a gap; only a narrower state type would shrink it.
* **Host overhead is small.** Wall − GPU per step is 0.2-0.4 ms (27B: ≤ 1 %; Llama-1B / OLMoE: 5-9 % at b16-64). Logits read-back
  (no device sampling) adds 6-8 ms at b64 on Llama-1B (12.3 vs 3.9 ms wall), which device sampling already removes.
* **Command-buffer recording is negligible.** The first step at a new shape costs 10-40 ms once (27B; 5-13 ms Llama-1B), and the
  serving runs recorded 105 / 128 / 152 distinct shapes (27B / Llama / OLMoE; prefill chunks vary the row count, the 64-entry
  cache evicts) in 0.97 / 0.11 / 0.52 s in total: 0.3-0.8 % of the timed wall time.
* **Serving vs kernel step.** The serving step at c = 64 (27B 101.7 ms, Llama-1B 6.44 ms) is 1.5-1.7x the pure-decode BatchBench
  step because ~107 prefill rows ride along per step on average; vLLM carries the same mix (180-185 rows per iteration) in
  44.7 / 3.78 ms. The 27B c = 1 serving step (27.8 ms) is ~3 ms above the CLI / BatchBench 1-row step (24-25 ms); not
  investigated further.

### 16.7 Prefill attribution

`SHARPLIGHT_VK_PROFILE=1 sharplight MODEL --bench N --batch N` (one chunk of N rows), sum of per-launch GPU time:

| | 27B FP8 T=512 | 27B FP8 T=2048 | Llama-1B T=2048 | OLMoE T=2048 |
|---|---|---|---|---|
| GPU time (≙ tok/s) | 175.2 ms (2923) | 636.3 ms (3219) | 39.1 ms (52.4k) | 63.4 ms (32.3k) |
| vLLM wall for the same prompt length | 99 ms | 345 ms | 27.5 ms | 34.6 ms |
| dense GEMM | 74.7 % | 70.5 % | 72.9 % | 13.8 % |
| GEMM glue | 2.8 % | 1.9 % | 4.3 % | 4.2 % |
| MoE expert GEMMs + routing | | | | 49.8 % |
| delta rule | 12.7 % | 12.7 % | | |
| conv1d | 4.6 % | 6.6 % | | |
| attention (flash) | 1.7 % | 3.3 % | 14.8 % | 11.2 % |
| norms / elementwise | 3.5 % | 5.1 % | 8.1 % | 21.0 % (MoE combine `fused[Mul]` + `ReduceSum`: 13 %) |

* The FP8 GEMM (`gemm_fp8`, 16x16x32 subgroup MMA) runs at 264-266 TFLOPS (`[5120→17408]`: 346 µs at T=512, 1375 µs at
  T=2048); the bf16 `gemm_cm2` at ~157 TFLOPS (Llama `[2048→8192]` at T=2048). vLLM's whole 27B pp2048 takes 345 ms, less than
  Sharplight's GEMMs alone (448 ms).
* Workgroup-scope cooperative matrices as implemented today do not help: `SHARPLIGHT_GEMM_NV2=1` (f16 `gemm_nv2`) on the Q8_0
  27B is 1.6-1.8x slower than the int8 MMQ (`[5120→17408]` 805 vs 451 µs, prefill 294 vs 189 ms); for FP8 weights it is not
  selected at all (the f16 fallback stays on `gemm_cm2`, 545 µs against 346 µs for `gemm_fp8`).
* The delta rule takes 370 µs per layer at T=512 and 1533 µs at T=2048 (sequential over tokens, 12.7 % of prefill); conv1d grows
  superlinearly (139 → 788 µs per layer for 4x the tokens, 6.6 % at T=2048), a plain kernel problem.

### 16.8 Ranked levers to close the vLLM gap

Payoffs are estimates from the measurements above (27B = Qwen3.8-27B FP8, the model with the largest gap).

| # | Lever | Evidence | Estimated payoff |
|---|---|---|---|
| 1 | **Weight-read-once small-batch matvec for 16-bit and FP8 weights (2-16 rows)**, the analogue of the int-dot `mmvq` path that Q8_0 and the K-quants already have (bf16/f16: f16 dots or cooperative vectors; FP8: per-block decode to f16 / int8) | FP8 27B step 23.9 / 39.3 / 62.5 / 118 ms at 1 / 2 / 4 / 8 rows vs 24.0 / 20.5 / 24.1 / 29.2 ms for Q8_0 on `mmvq`; Llama-1B bf16 2.07 → 6.38 ms at 8 rows; serving c = 4 is 0.38x (27B) and 0.66x (Llama-1B) | 27B c = 4: step 72 → ~28 ms, ~2.5x throughput (≈ vLLM's 144 tok/s); Llama-1B c = 4 ~1.5x; every step with 2-8 decode rows (ramp-up, tails) |
| 2 | **Skinny-M tensor-core GEMM at the bandwidth roof (9-64 rows)**: weights streamed once per step (K-parallel workgroups instead of split-K + reduce, FP8 / int8 operands without the f16 activation copy, pipelined loads) | FP8 GEMM at 0.70-0.95 TB/s vs 1.61 TB/s for the matvec; matmuls 35 / 46 ms per b16 / b64 step vs ~18 ms roofline | 27B b16 step 45 → ~30 ms, b64 66 → ~45 ms (c = 16 / 64 from 0.61x / 0.54x to ~0.85x / ~0.75x of vLLM); similar share on Llama-1B |
| 3 | **Fewer barriers and launches in batched steps**: activation conversion / quantization fused into the producer, split-K reduction into the consumer or a single-pass GEMM, independent small launches merged between barriers | No-barrier ablation −20..−25 % (27B b16 / b64), −40 % (Llama-1B b16 / b64), −20 % at b1 on Llama-1B; glue kernels 7-9 % (27B) / 19 % (Llama-1B) of the profile | 10-25 % of the batched step (the ablation is an upper bound); largest for small models; overlaps partly with #2 |
| 4 | **MTP speculative decoding** (needs delta-rule / conv1d state rollback; `CanTruncate` is false for recurrent models today) | vLLM: 1.58-1.95x single stream (k = 1-3, acceptance 0.59-0.84), 1.45-1.89x serving up to c = 16, 0.70-0.92x at c = 64 | ~1.8x at c = 1-4, ~1.5x at c = 16 on the 27B; off above ~32 streams |
| 5 | **Faster FP8 / int8 / bf16 tensor-core GEMM for prefill** (multi-stage pipelined loads, larger tiles; a workgroup-scope 32x32x32 fp8 kernel only if it beats the subgroup kernel, which the f16 `gemm_nv2` does not) | `gemm_fp8` 264 TFLOPS, GEMMs 70-75 % of 27B prefill, vLLM pp2048 1.96x; bf16 GEMM ~157 TFLOPS (Llama / OLMoE prefill 0.45-0.62x) | GEMM 1.6x → 27B prefill ~1.35x; with #6 / #7 the 27B prefill reaches ~0.8x of vLLM |
| 6 | **Chunked tensor-core delta rule** for prefill | 12.7 % of 27B prefill (370 µs per layer at T=512); the decode delta rule is already at the bandwidth roof | ~10 % of 27B prefill; nothing for decode |
| 7 | **conv1d prefill kernel and MoE combine fusion** | conv1d 6.6 % of 27B pp2048 (superlinear, 139 → 788 µs); OLMoE prefill `fused[Mul]` + `ReduceSum` 13 % | 5-7 % (27B prefill), ~10 % (MoE prefill) |
| 8 | **Batched paged attention efficiency** (decode attention bandwidth) | 2-4 % of the 27B step, 10-18 % of Llama-1B, 28 % of OLMoE (MHA) at b64 with only 256-token contexts; Llama b64 reads 33.5 MB in 45 µs per layer (0.74 TB/s) | ~5-15 % for small MHA / GQA models now; grows with context length |
| 9 | **fp8 KV cache** | KV traffic is small at the measured contexts (27B: 64 KB per token over its 16 attention layers); at 4k contexts × 64 streams it would be ~17 GB per step, more than the weights. q8_0 / q4_0 KV exist already | nothing at short contexts; up to ~1.5-2x for attention-bound long-context serving (mostly covered by q8_0 KV) |
| 10 | **Async scheduling / overlapping host and GPU** | Wall − GPU 0.2-0.4 ms per step (≤ 1 % 27B, 5-9 % small models); step metric covers 99-100 % of wall. vLLM loses 22-41 % without it because its Python host work is much larger | ≤ 1 % (27B), 5-9 % (1B models) |
| 11 | **Pre-recorded command buffers per batch-size bucket** | Recording is 0.3-0.8 % of serving wall time (105-152 distinct shapes, 0.1-1.0 s per run); command buffers are already cached per shape (the CUDA-graph analogue, worth 17-74 % to vLLM) | < 1 %: refuted (a larger recording cache or bucketed prefill-chunk row counts would remove even that) |

Not levers: single-stream decode (Sharplight 39.2 vs vLLM 39.5 tok/s on the 27B, 1.23-1.35x on the 1B models; both at the
bandwidth roof), the decode-side delta rule (bandwidth roof, same state size as vLLM), sampling (0.1-0.5 %).


## 17. Speculative decoding: MTP drafts with recurrent-state rollback (2026-11-14)

Qwen3.8-27B is a hybrid (48 Gated DeltaNet + 16 gated-attention layers) with one MTP layer. Speculation needs the
recurrent state of rejected drafts rolled back; Sharplight does that by record-and-replay inside the recurrent kernels
(`Graph.RecurrentRollback`: a verification run leaves the state at its start and records its per-row op inputs, about 35 MB
per slot for 3 drafts; the next run replays the kept rows before its own), so the target needs no state snapshots and no
recomputation pass. The MTP head (`mtp_draft` graph) runs as its own small decoder on the target's hidden states and drafts
greedily; verification keeps a draft while it equals the target's own sample (exact for greedy and sampled decoding). Greedy
output is token-identical to decoding without speculation (CLI check on the Q8_0 GGUF and the FP8 checkpoint, 200 tokens each;
tiny-model engine tests on CPU and Vulkan).

Setup as section 16 (RTX PRO 6000 Blackwell Max-Q, shared GPU: the user's own apps were active during parts of the runs, see
`bench/results/s17/*.gpu.txt`; numbers are single rounds and carry ~5-10 % noise). Model: `Qwen3.8-27B-Q8_0.gguf`
(lmstudio-community, includes the NextN layer) and the FP8 checkpoint of section 16. The NVFP4 "MTP" GGUF is not loadable
(separate `.scale` / `.input_scale` tensors that no importer maps). Raw output: `bench/results/s17/`.

```bash
dotnet build bench/SpecBench -c Release
bench/SpecBench/bin/Release/net11.0/SpecBench <model> --drafts 0,1,2,3        # single stream, as vllm_single.py --text-prompt
LOADGEN_PY=python3 NATURAL=1 ROUNDS=1 OUT_DIR=bench/results/s17 bench/s16_serve.sh sharplight qwen27b-q8-mtp3-b <gguf> "1 4 16 64" --speculate-mtp 3
```

### 17.1 Single stream

`SpecBench`: natural-language prompt (the essay instruction of `vllm_single.py`, padded with random tokens to the depth), greedy,
end-of-generation ignored, tg = 128 / (t(129 tokens) − t(1 token)), median of 5 (max in parentheses); one slot through the
`InferenceEngine`. Mean accepted length per step = 1 + k · acceptance. vLLM rows: section 16.5 (FP8, same prompt).

| k | Q8_0 @d47 tok/s | acceptance | Q8_0 @d4096 | acceptance | FP8 @d47 | acceptance | vLLM FP8 @d47 (acceptance) |
|---|---|---|---|---|---|---|---|
| off | 36.1 (37.9) | | 37.0 (38.9) | | 30.7 (36.3) | | 38.4 |
| 1 | 71.0 (71.8) | 0.80 | 68.7 (77.3) | 0.83 | 26.7 (36.1) | 0.84 | 60.5 (0.84) |
| 2 | 81.1 (90.6) | 0.71 | 73.3 (74.9) | 0.65 | 38.2 (44.9) | 0.76 | 70.6 (0.70) |
| 3 | 86.6 (89.7) | 0.58 | 78.0 | 0.60 | 37.1 (38.5) | 0.59 | 74.7 (0.59) |

* On the Q8_0 GGUF MTP is worth 1.9x (k = 1) to 2.4x (k = 3) single stream, ahead of vLLM's MTP on FP8 (1.58-1.95x, 74.7 tok/s
  at k = 3). Acceptance matches vLLM's (0.84 / 0.70 / 0.59 for k = 1-3), which also checks the draft head itself.
* On FP8 weights MTP barely pays: verification steps of 2-4 rows hit the small-batch cliff of the FP8 matvec (section 16.6,
  lever #1: 23.9 / 39.3 / 62.5 ms at 1 / 2 / 4 rows), which eats the accepted tokens. The FP8 runs were also the noisiest
  (another process on the GPU). Lever #1 is the prerequisite for MTP on FP8 / bf16 checkpoints.

### 17.2 Serving

`bench/s16_serve.sh` (64 slots, 512-token contexts, natural prompts of ~25 tokens, 128 output tokens, ignore_eos, one round),
`--speculate-mtp 3`, engine defaults `SpeculationMaxBatch` 16 and `SpeculationMaxDraftRows` 16 (drafts per sequence =
min(k, 16 / decoding sequences): k = 3 up to 5 sequences, 1 at 16, none above 16). Aggregate tok/s; acceptance from
`/metrics`.

| conc | off | MTP k = 3 (acceptance) | speedup | vLLM FP8 off | vLLM FP8 MTP k = 3 (speedup) |
|---|---|---|---|---|---|
| 1 | 35.3 | 80.5 (0.53) | 2.28x | 42 | 76 (1.81x) |
| 4 | 147.6 | 183.8 (0.56) | 1.25x | 147 | 278 (1.89x) |
| 16 | 327.7 | 479.3 (0.82) | 1.46x | 554 | 843 (1.52x) |
| 64 | 824.3 | 905.1 (no drafts at c = 64; ramp and tail only) | 1.10x | 1370 | 965 (0.70x) |

* MTP gains 1.25-2.3x up to 16 streams and is at parity at 64, where the engine stops speculating (801-905 vs 824-831 tok/s over
  the two passes; vLLM loses 30 % there). The first baseline pass ran while another application loaded the GPU (c = 1: 25.7 tok/s).
  The first pass with the old limit (`SpeculationMaxBatch` 8, `serve_*-mtp3_*` and `-mtp1_*`) had no drafts at c = 16 (1.09x);
  allowing 16 sequences with one draft each gave 1.43x (`-mtp3-mb16`), hence the new defaults.
* c = 4 gains least (1.25x): 4 × 4 verification rows make the target step 42 ms against 27 ms for 4 rows (16 rows of the int-dot
  matvec, logits and hidden states for every row), plus three draft-head launches per step. A smaller k at c = 4 or device-side
  verification (lever below) would recover part of it.
* The remaining serving gap to vLLM is the batched-step gap of section 16 (levers #1-#3), which speculation multiplies but
  does not close.

### 17.3 What it costs and what is left

* Memory: the record state is `[slots, 2, k + 1, W]` f32 per recurrent op (Qwen3.8-27B: about 35 MB per slot at k = 3, 2.2 GB
  for 64 slots); the draft head adds its own copy of the embedding and LM head (2.5 GB for the Q8_0 27B; weights are not
  shared between compiled models yet) and one attention layer's KV cache.
* Verification reads back full logits rows (1 MB per row for the 248k vocabulary) and the hidden states of every row; a
  device-side multi-row greedy / compare kernel would remove both from the speculative step.
* Drafting is greedy only; drafting from the head's distribution with min(1, p/q) acceptance would raise acceptance for sampled
  requests (vLLM does not do this for MTP either).

(The first two items are addressed in 17.4.)

### 17.4 Follow-up: shared weights, device-side verification, rollback-form prefill, NextN heads (2026-11-14)

Changes:

* **Shared weights** (`CompileOptions.SharedWeights`, `IComputeDevice.SharesWeights`, CPU and Vulkan): compiled models of one
  source share device buffers per weight (keyed by device, source, type, size and the provider's repack layout; reference
  counted). The draft head now reuses the decoder's embedding and LM head: on the Q8_0 27B that is 2.70 GB (2 × 1.35 GB Q8_0)
  no longer duplicated; the head's own compilation is 0.43 GiB weights + state (its NextN layer and KV cache).
* **Device-side verification**: with device sampling (the default), every verification row is sampled on the GPU (greedy, or
  from the seeded stream at the row's position, so seeded sampling stays identical to decoding without speculation), and the
  step reads back one token and log-probability per row instead of 1 MB logits rows. Hidden states are downloaded only for the
  rows the draft head consumes, and not at all on steps without a capture (they were read in full every step). The draft head
  takes its argmax on the device too.
* **Footprint**: `EstimateFootprint(..., draftTokens, draftHead, maxLogitsRows)` counts the record states and the head (its own
  weights only); the hosted loader reserves them before admission.
* **Rollback-form prefill** (item of section 18.1): with speculation enabled, long prompt runs take the chunked delta rule and
  token-parallel conv1d after the token kernel replays committed rows.
* **NextN heads** for DeepSeek-V3 (`latent_decoder`) and GLM-4.5 (`decoder`), HF and GGUF; tiny-model engine tests show
  greedy identity on CPU and Vulkan (no real checkpoint was available to measure acceptance).
* The Families-suite segfault was a GC-moved extension-name array passed to `vkCreateDevice` (fixed by pinning it through
  the call); 6 / 6 full-suite runs pass after the fix, 1 / 3 before.

Correctness: greedy CLI output on `Qwen3.8-27B-Q8_0.gguf` with `--speculate-mtp 3` is identical to no speculation (200 tokens
on an 8-token prompt, 140 tokens on a 636-token prompt that goes through the chunked rollback prefill; acceptance 53 % / 49 %).
Serving runs before/after show identical step counts and accepted drafts (e.g. c = 16: 292 steps, 3648 / 4462 accepted).

**Measurements.** The GPU was shared during all of these runs (another agent's 27B process at ~100 % utilisation for most
of them, plus another application), so absolute numbers are about half of 17.1 / 17.2 and only interleaved A/B comparisons mean
anything. Raw output: `bench/results/s17b/`.

Serving c = 4, `--speculate-mtp 3`, three interleaved rounds (aggregate tok/s, mean inter-token latency):

| round | before (main) | after |
|---|---|---|
| 1 | 86.2 (43.2 ms) | 91.0 (40.9 ms) |
| 2 | 82.7 (45.1 ms) | 86.7 (42.9 ms) |
| 3 | 83.9 (44.4 ms) | 87.0 (43.0 ms) |

Device verification gains about 5 % at c = 4 under contention (the step is dominated by the shared GPU; the saved transfers
are a fixed ~16 MB per step). A single c = 1 / 4 / 16 / 64 pass gave 35.9 → 36.3, 137 → 89 (the "before" run hit an idle window;
not reproduced by the interleaved rounds above), 231 → 257 and 447 → 481 tok/s.

Single stream (`SpecBench`, k = 0 / 3, depths 47 and 4096): only the first "before" run had an idle GPU (k = 3: 92.9 tok/s at
d47, 78.8 at d4096; TTFT at d4096 1601 ms without speculation and 1760 ms with it, so the rollback form cost about 10 % of
prefill before the change). The "after" runs were all contended and are not comparable. A clean single-stream and TTFT
comparison is still to be done on an idle GPU.

### 17.5 Qwen3.8-Flash-Next: the hyper-connected MTP head (2026-11-16)

Qwen3.8-Flash-Next (`qwen4exp`) ships one MTP layer that works on the hyper-connected residual: it reads the trunk's four
residual streams before the final mixer, projects `[enorm(e); hnorm(h)_s]` per stream, runs one hyper-connected QSA attention +
MoE layer and mixes its streams with its own `hc_head` before the shared LM head (llama.cpp `graph_mtp`). Sharplight builds it
as `MtpDraftHead.BuildHyperConnected` (decode output `hidden_streams` as the draft hidden source) from HF checkpoints, GGUFs
with the NextN block, and the separate `mtp-*.gguf` draft file (CLI `--draft <file>`). On the tiny model the drafts equal a
reference assembled from transformers' own modules (`tools/mk_qwen4exp_mtp_reference.py`) on CPU and Vulkan, HF, GGUF and
sidecar alike, and greedy output with the head is token-identical.

Real model: unsloth UD-IQ1_S (66 GB, 3 shards) + `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, one RTX PRO 4500 (32 GB) with CPU offload,
CLI, raw essay prompt (the SpecBench text), greedy, 200 tokens. The 32 GB GPU cannot hold the layer with the n-gram table, so the
planner placed every trunk layer on the CPU (1 GiB on the GPU, 61 GiB on the CPU): the numbers are CPU decode, not a GPU result.

| k | tok/s | acceptance (accepted / drafted) | output vs k = 0 |
|---|---|---|---|
| off | 0.99 | | |
| 1 | 1.04 | 0.78 (87 / 111) | same for the first half, then one near-tie |
| 2 | 0.96 | 0.71 (117 / 164) | as k = 1 |
| 3 | 0.60 | 0.58 (126 / 218) | as k = 1 |

* Acceptance 0.78 / 0.71 / 0.58 is what the Qwen3.8-27B head reaches (section 17.1: 0.80 / 0.71 / 0.58) and vLLM's on the 27B
  (0.84 / 0.70 / 0.59), which checks the head on the real weights.
* The three speculative runs are identical to each other; they leave the plain run at one near-tie (" Pre" 22.225 vs " Origins"
  22.167 in the prefill logits at that position, which pick " Pre" as plain decoding does): the CPU verification rows (2-4 at a
  time) went through the chunked dequantize-then-dot path while single-row decode used the fused dequantize-dot kernel, which sums
  in another order. Fixed: up to 4 rows now run the fused kernel row by row (`MatMulOp`, `MatMulIdOp`), so verification rows
  are bit-identical to single-row decode.
* No speedup on this placement: the MoE layers run on the CPU, where each verification row reads its own experts, so a step
  costs nearly (k + 1) times a single-row step. Fixed in the planner since: a lookup table (a weight only `Gather` reads, the
  26.8 GiB n-gram table here) is its own placement group and goes to the CPU alone, so the trunk layers fill the GPU(s).

#### 17.5.1 After the placement fix: trunk on the GPU

Same model, prompt and CLI run; the files on local disk. Caution: single cold CLI runs, one per k (see 17.6), so the speedups
below are rough; the `SpecBench` rerun follows. The 26.8 GiB n-gram table (IQ4_NL) is its own placement group and goes to
the CPU; 40 of the 48 layers fill the 32 GB GPU (27.7 GiB), 8 layers run on the CPU.

| k | one RTX PRO 4500 + CPU tok/s | acceptance | output vs k = 0 |
|---|---|---|---|
| off | 17.8 | | |
| 1 | 19.7 (1.11x) | 0.87 (92 / 106) | one near-tie |
| 2 | 25.0 (1.41x) | 0.79 (121 / 154) | as k = 1 |
| 3 | 20.7 (1.16x) | 0.64 (130 / 204) | as k = 1 |

* Plain decode went from 0.99 to 17.8 tok/s with the placement fix (llama.cpp `-ngl 33` on the older 96 GB machine: 13.9 Vulkan, 18.8 CUDA).
* k = 2 is the best setting here (1.41x); the 8 CPU layers make every verification row cost more, so k = 3 loses again.
* The speculative runs are identical to each other and meet the plain run at the same near-tie as above (gap 0.06), now
  flipped the other way: on the GPU, `RowInvariantRows` makes the matmuls and attention of 2-4-row steps bit-identical to
  single-row steps (the row-invariance tests now include the IQ formats and expert matmuls, all bit-identical), but not every
  op this model runs (the delta-rule replay, the indexer, the n-gram convolution, the CPU layers). Open item: find the op that
  breaks invariance here.

#### 17.5.2 SpecBench, two GPUs

`SpecBench --gpus 2 --cpu-offload --draft mtp-Qwen3.8-Flash-Next-Q8_0.gguf` on two RTX PRO 4500 (the whole trunk on the GPUs, the
n-gram table on the CPU), plain and speculative through the same engine, depth 47, median of 3 (load average ~18 on the machine,
spreads tight).

| k | tok/s (min-max) | speedup | acceptance (accepted / drafted) |
|---|---|---|---|
| off | 58.2 (55.1-59.1) | | |
| 1 | 85.3 (83.9-85.6) | 1.47x | 0.88 (179 / 204) |
| 2 | 72.1 (71.0-79.0) | 1.24x | 0.81 (236 / 293) |
| 3 | 78.2 (77.3-95.7) | 1.34x | 0.70 (258 / 370) |

One GPU + CPU (8 layers on the CPU) at load average 33 gave plain 24.4, k = 1 27.6 tok/s with acceptance 0.88 / 0.79 / 0.70; its
timings were dominated by CPU contention and are not used.

Greedy text through the engine: k = 1, 2, 3 are identical to each other and leave plain decoding at one list-item near-tie after
~520 characters. Two row-invariance gaps were found and fixed on the way: the Vulkan expert matmul left the row-invariant matvec for
the grouped GEMM above 16 items (a 4-row step with 10 experts per row is 40 items), and the CPU matmuls summed 2-4 rows in another
order than one row. `SpeculativeRowInvarianceTests` (4-row verification step against 4 single-row steps, logits bitwise) now pass on
Vulkan for tiny Llama, Qwen3.5, Qwen3.8-MoE, Qwen3-Next-MoE and Qwen3.8-Flash-Next; the remaining gaps are listed as an open item in
model-support.md.

### 17.6 MiMo-7B-RL: MTP on bf16 weights (2026-11-16)

XiaomiMiMo/MiMo-7B-RL (HF `mimo`, bf16 safetensors, one MTP layer `model.mtp_layers.0` imported as the NextN draft head), one RTX
PRO 4500. `SpecBench` (as section 17.1: the same `InferenceEngine` path for plain and speculative decoding, essay prompt at depth 47,
greedy, median of 5 or 3 runs; `--dump` checks the text): output is byte-identical to plain decoding for k = 1, 2 and 3.

| k | tok/s (min-max) | speedup | acceptance (accepted / drafted) |
|---|---|---|---|
| off | 56.0 (55.8-56.4) | | |
| 1 | 77.4 (73.3-83.4) | 1.38x | 0.70 (524 / 754, two runs) |
| 2 | 73.0 (71.4-74.8) | 1.30x | 0.42 (175 / 413) |
| 3 | 63.7 (59.8-66.1) | 1.14x | 0.30 (179 / 601) |

MiMo's MTP layer is trained for the next-but-one token only; chained further it guesses much worse, so k = 1 is the setting.
A first CLI measurement reported 1.98x: its plain run was a single cold run (37.6 tok/s; the first run after the download, while
the GPU was shared) against later speculative runs. Rerun with a warm shader cache the CLI's plain decode gives 56.9 tok/s, the
same as the engine (`SpecBench` 56.0; Llama-3.2-1B Q8_0: CLI 494, engine 451 tok/s), so the CLI's `Generator` path is not slower.
Speedups in this section come from `SpecBench` only (warm-up, repeated runs, plain and speculative through the same engine).

## 18. Chunked gated delta rule and token-parallel conv1d for prefill (2026-11-14)

Levers 6 and 7 of section 16.8. Kernels: `src/Sharplight.Shaders/Kernels/chunked.slang` (new), `recurrent.slang`
(`conv1d_tokens`, `conv1d_state`); lowering: `GraphLowerer.DeltaChunk.cs`, `GraphLowerer.Recurrent.cs`. Script:
[`bench/s18_prefill.sh`](../bench/s18_prefill.sh), raw output in [`bench/results/s18/`](../bench/results/s18).

### 18.1 Design

**Delta rule.** The `GatedDeltaRule` op (every model that lowers to it: Qwen3.5 / 3.6 / 3.8, qwen3next) gets the chunked
WY / UT form of FLA's `chunk_gated_delta_rule` on f16 cooperative matrices with f32 accumulation. Per value head and 64-token
chunk, with G the chunk-local inclusive cumulative log decay:

```
L  = strictly lower (beta_t exp(G_t - G_s) k_t.k_s)      X = (I + L)^-1      Tb = X diag(beta)
P  = lower incl. diagonal (scale exp(G_t - G_s) q_t.k_s)
U  = Tb (V - diag(exp(G)) K S^T)
O  = scale diag(exp(G)) Q S^T + P U
S^T <- exp(G_last) S^T + K^T diag(exp(G_last - G)) U
```

(checked against the token recurrence in float64 to 1e-16). Three launches:

* `delta_chunk_qk`: L2-normalised q / k rows (the op's absorbed `L2Norm`) as f16 planes.
* `delta_chunk_prep`: one workgroup per (head, chunk), all chunks in parallel. It computes K Kᵀ and Q Kᵀ on tensor cores and
  X by forward substitution in f32. Each of 64 threads solves one column in registers, so the solve needs no workgroup
  barriers; the first version took one barrier per row and the kernel ran at 291 µs instead of 172 µs on the 27B at T = 2048.
* `delta_chunk_state`: one workgroup per (head, 32 state columns, sequence run). It walks the run's chunks in order. The
  state stays in f32 accumulators for the whole run, is read and written once in the existing `[HV, DV, DK]` layout, and the
  outputs are written per chunk.

Behaviour, limits and switches:

* **Multi-sequence batches.** Runs of ≥ 64 rows (`SHARPLIGHT_DELTA_CHUNK_MIN`) take the chunked form. Shorter runs take the
  token-by-token kernel in the same step, which now skips the long runs (`SEQ_MAX_ROWS`). A run starting at position 0
  starts from zero state.
* **Single sequences and decode.** Single sequences below 64 rows, and decode, stay on the token-by-token kernel.
* **Scratch.** Scratch is indexed by global row, so runs never collide. For the 27B at T = 2048 it is 2 × 2 × 2112 × 16 ×
  128 f16 for q / k (35 MB), 2 × 2 × 2112 × 48 × 64 f16 for Tb / P (52 MB), plus G: about 87 MB with two parts and half
  that with one, shared by all layers.
* **Device requirements.** The device needs `matrix_f16`, `float16`, subgroup ops with 32-wide subgroups,
  DK, DV % 16 = 0, DK ≤ 256 and the tiles within shared memory (47 KB at 32 columns and two parts). Otherwise the
  token-by-token kernel runs everywhere (`SHARPLIGHT_DELTA_CHUNK=0` forces that). The CPU provider is unchanged.
* **Precision.**
  * `PARTS = 2` (default unless `CompileOptions.AllowReducedPrecision`) splits every operand into f16 hi + lo, with lo
    scaled by 2048 so it does not flush as a subnormal. It runs three MMAs per product, about 22 mantissa bits.
  * `PARTS = 1` runs one f16 product with f32 accumulation, the precision class of FLA's bf16 / f16 kernels in vLLM.
  * `SHARPLIGHT_DELTA_CHUNK_PARTS=1|2` overrides the choice.
* **Driver finding (NVIDIA 610).** The compiler folds `float(half(x))` round trips, including those through
  cooperative-matrix conversions, so `x - float(half(x))` became 0 and the "split" had no lo part (rel. error 4e-4, the
  same as f16). The split therefore rounds hi with integer arithmetic on the f32 bits, and the state is staged through f32
  shared memory and split in scalar code.

**conv1d.** `conv1d` walked all tokens of a run per channel in one thread. 10240 channels gave 80 workgroups, each issuing
T dependent-latency loads, so the cost grew with T times the load latency. Prefill batches with a run of ≥ 16 rows now run
two launches:

* `conv1d_tokens`: 8 consecutive outputs of one channel per thread, from x and the old state (grid: channel block × token
  block × run).
* `conv1d_state`: after a barrier, writes each run's last KS-1 inputs as the new state.

Short batches and decode keep the old kernel (`SHARPLIGHT_CONV1D_PARALLEL=0` forces it everywhere). The results are bitwise
identical to the old kernel (model logits relRms 0.0).

**State layout:** unchanged (delta-rule state `[slot, HV, DV, DK]` f32, conv state `[slot, KS-1, C]` f32).

**Rollback form (section 17).** Graphs built with `MultiSequence.Transform(..., rollbackRows)` have a commits table and
record banks. Their recurrent ops run the token-by-token `delta_rule` / per-channel `conv1d` first: every run replays its
committed rows (from the read bank) into the slot's state, and short runs (tentative verification runs, decode) then
process, record and store as before. Runs longer than the threshold (64 rows for the delta rule, 16 for conv1d, at least
k + 1 so a tentative run never qualifies) stop after the replay and store the replayed state; the chunked delta rule and
`conv1d_tokens` / `conv1d_state` then serve them from that state, after a barrier. So a decoder with speculation enabled
gets this section's prefill gains too (17.4). `RecurrentRollbackTests` covers a partially kept speculative round followed
by a 64-row run against the reference.

### 18.2 Accuracy

**Op level** (`ChunkedDeltaRuleTests`, Vulkan vs the CPU oracle over all outputs and the final state, max relative rms
per step).

The batches are `[150, 1, 64, 3, 63, 200, 1]` rows. They mix the chunked and token-by-token kernels and pass the state back
and forth, and include partial chunks. The multi-sequence variant mixes long and short runs in one step and has a slot
restart. Head shapes: (HK, HV, DK, DV) = (1, 3, 128, 128), (2, 4, 64, 32), (4, 4, 32, 64), (2, 2, 16, 48), (2, 6, 128, 128),
with q / k normalised in the graph or beforehand and decays of 0.01-0.5.

| kernel | relRms |
|---|---|
| token-by-token `delta_rule` (reference) | 3.1e-7 |
| chunked, two parts (default) | 3.6e-7 - 1.8e-6 |
| chunked, one f16 part | 5.1e-4 |
| `conv1d_tokens` + `conv1d_state` | ≤ 1e-7 (identical to `conv1d`) |

**Model level.** The prompt is 636 tokens (the first 2600 characters of `docs/code-style.md` plus an instruction): 512-row
prefill chunks plus a 124-row chunk, then 48 greedy tokens. The table compares the last-prompt-token logits and the
greedy text of the chunked forms against the token-by-token kernels on the same build.

| model | logits relRms two parts / one part | top-1 | greedy text (48 tokens) | noise floor: GPU token-by-token vs CPU |
|---|---|---|---|---|
| Qwen3.5-0.8B bf16 | 8.3e-4 / 8.0e-4 | same | identical / identical | 5.6e-4 |
| Qwen3.8-27B FP8 | 3.3e-2 / 3.6e-2 | same | identical / identical | |
| Qwen3.8-27B Q8_0 | 1.6e-2 / 1.4e-2 | same | identical / identical | 1.3e-2 |
| Qwen3.8-27B Q4_K_M | 1.9e-3 / 2.0e-3 | same | near tie at token ~45 (two parts) / identical | |
| Qwen3.6-35B-A3B Q4_K_M | 3.5e-2 / 3.1e-2 | same | identical / identical | |
| Qwen3-Coder-Next Q4_K_M (qwen3next) | 2.5e-2 / 3.0e-2 | same | near tie at token 2 / near tie at token 2 | |

The two forms are indistinguishable at model level, even though they are 3 orders of magnitude apart at op level. Any
perturbation, however small, re-draws the f16 / fp8 / int8 rounding of the activations in the following GEMMs. The
model-level difference therefore sits at the engine's own noise floor: the same GPU kernels against the CPU provider differ
by as much, and the 0.8B's GPU-vs-CPU distance is 5.6e-4 (token-by-token), 6.9e-4 (two parts) and 6.5e-4 (one part).

Both divergences are near ties of the token-by-token run itself:

* **Qwen3-Coder-Next.** After "… three sentences: what is", the top two logits are "enforced" 15.26 and "the" 15.24.
* **27B Q4_K_M.** At "Code style …" the four leading candidates lie within 0.8 and the variants move them by ≤ 0.013.

The family tests (`Sharplight.Families.Tests`) pass; see 18.5.

### 18.3 Prefill throughput, Qwen3.8-27B

`bench/s18_prefill.sh prefill`: one prefill chunk of N rows (`--bench N --ctx N+256 --batch N`), best of the CLI's
repetitions, tok/s. "before" is the same build with `SHARPLIGHT_DELTA_CHUNK=0 SHARPLIGHT_CONV1D_PARALLEL=0`, which is
exactly the kernel set of `main` before this change. The GPU was free apart from short-lived test processes of other
agents (logged per row). llama.cpp b11177 CUDA: `llama-bench -p 512,2048,8192 -n 0 -r 3 -ngl 99 -fa 1` (default batch 2048 /
ubatch 512).

| weights | N | before | after, two parts (default) | after, one part | gain (default) | llama.cpp CUDA |
|---|---|---|---|---|---|---|
| FP8 | 512 | 2989 | 3733 | 4127 | +25 % | |
| | 2048 | 3401 | 3797 | 3735 | +12 % | |
| | 8192 | 3106 | 3393 | 3387 | +9 % | |
| Q8_0 | 512 | 2672 | 2863 | 2934 | +7 % | 2922 |
| | 2048 | 2734 | 2949 | 3000 | +8 % | 2919 |
| | 8192 | 2635 | 2853 | 2927 | +8 % | 2799 |
| Q4_K_M | 512 | 2490 | 2623 | 2690 | +5 % | 2706 |
| | 2048 | 2418 | 2592 | 2573 | +7 % | 2708 |
| | 8192 | 2282 | 2467 | 2366 | +8 % | 2512 |

Decode is unchanged (single-token steps keep the token-by-token kernels): 41-43 tok/s best in every row.

With these changes:

* **Q8_0.** Sharplight is level with llama.cpp CUDA at 512 and ahead at 2048 / 8192.
* **Q4_K_M.** Sharplight is 2-4 % behind; the K-quant GEMM is the remainder.

### 18.4 Kernel GPU time, 27B FP8

`SHARPLIGHT_VK_PROFILE=1`, min per launch, µs per layer (48 delta-net layers); raw output in `profile_fp8.txt`.

| N | delta rule before | chunked two parts (qk + prep + state) | chunked one part | conv1d before | conv1d after (tokens + state) | step GPU time before → after (two parts / one part) |
|---|---|---|---|---|---|---|
| 512 | 372 | 194 (6 + 59 + 129) | 109 (4 + 41 + 64) | 143 | 14 (12 + 2) | 176.4 → 140.3 / 135.5 ms |
| 2048 | 1537 | 717 (25 + 170 + 522) | 377 (14 + 121 + 242) | 791 | 81 (79 + 2) | 596.2 → 530.2 / 514.6 ms |
| 8192 | 6228 | 2929 (177 + 621 + 2131) | 1572 (123 + 436 + 1013) | 3128 | 408 (406 + 2) | 2549.9 → 2350.8 / 2271.0 ms |

Effect on the kernels:

* **Delta rule, two parts.** 2.1x faster than the token-by-token kernel at every length.
* **Delta rule, one part.** 3.4-4.1x faster.
* **conv1d.** 9.8-10.2x faster. It is now bandwidth bound: 8192 × 10240 f32 read and written in 406 µs ≈ 1.65 TB/s. It
  scales linearly; the old kernel was not superlinear but latency bound (one dependent load chain per channel and 80
  workgroups).
* **Share of prefill at T = 2048.** The delta-net share fell from 18.7 % to 7.2 % (two parts).

The sequential `delta_chunk_state` pass dominates. Each workgroup walks 32 (T = 2048) chunks in order with
seven barrier-separated phases. With 192 workgroups (48 heads × 4 column blocks) on 188 multiprocessors it is latency bound,
not tensor-core bound.

Things that did not help:

* Software-pipelining the next chunk's loads.
* 16-column blocks: more workgroups, but each phase is just as long.

The two-part form runs 3 MMAs per product. With four subgroups (128 threads) its register count drops from 236 to about 128,
so two workgroups fit per multiprocessor: 609 → 520 µs.

### 18.5 Tests

* `Sharplight.Vulkan.Tests`: all pass, including the new `ChunkedDeltaRuleTests` (6 single-sequence shapes, 2
  multi-sequence scripts, and the token-by-token fallback with `matrix_f16` disabled). Its graphs also carry a 136-channel
  conv1d, so the token-parallel conv and its state hand-over are covered on the same batches.
* After rebasing onto `main` with the MTP / rollback merge: `Sharplight.Vulkan.Tests` 951 (17 skipped),
  `Sharplight.Cpu.Tests` 1133 (27 skipped), `Sharplight.Families.Tests` including `RecurrentRollbackTests`: pass.

### 18.6 Reuse for other linear-recurrent ops (design notes, not implemented)

* **Mamba-2 SSD** (`SelectiveScan` with a scalar decay per head) is the same chunk algebra with β = 1 and no delta
  correction: there is no L and no triangular inverse, so `Tb` = I and U = V. A chunked SSD would be a prep kernel without
  the solve (only P = C Bᵀ ⊙ decay mask) plus the existing sequential pass with step 4 dropped. The kernels take the decay
  only through G, so this needs no new decay handling. It was not implemented here because the SSD models in the tests are
  small (mamba2-130m, Falcon-H1-0.5B, granite-4.0-h-350m). The token-by-token selective scan is not the bottleneck of their
  prefill.
* **RWKV-7** uses a per-channel (vector) decay and a generalised delta rule, S ← S diag(w) + S a bᵀ + v kᵀ. The per-channel
  decay breaks the scalar `exp(G_t − G_s)` factorisation. Following FLA's `chunk_rwkv7` / `chunk_dplr`, it would need
  decay-scaled K / Q (k̃ = k ⊙ exp(G), q̃ = q ⊙ exp(−G) within the chunk, numerically safe only with sub-chunks of 16) and a
  low-rank (a, b) term in L. The prep and state kernels could reuse their tiling and the hi / lo split, but not the scalar
  G path.
* **ONNX `LinearAttention`** (the ONNX agent's CPU-only IR op). When it is a gated / ungated delta rule or plain linear
  attention with scalar per-head decay, it can lower to `GatedDeltaRule` on Vulkan:
  * Plain decayed linear attention, S ← e^g S + v kᵀ, has no delta correction: U = V and Tb = I, the SSD case above.
  * A missing decay is g = 0.
  * A missing L2 norm is `NORM_Q` / `NORM_K` = 0.

  The chunked kernels then serve it unchanged, with the token-by-token kernel as fallback. ONNX graphs default to f32
  without `AllowReducedPrecision`, so they get the two-part (near-f32) form. The only addition would be a macro for the
  β-free / identity-T variant to skip the solve.

### 18.7 Open issues

* **Default precision.**
  * The two-part split is the default because `AllowReducedPrecision` is off for LLMs, and the state is the analogue of
    f32 weights.
  * At model level the one-part form is indistinguishable (18.2) and saves a further 340 µs per layer at T = 2048 on the
    27B (≈ 3 % of prefill, +8 % at T = 512 on FP8).
  * Making it the default for LLM graphs is a policy decision.
* **Sequential pass latency.** FLA's split would shorten the sequential chain to two products per chunk (`W Sᵀ` and the state
  update) by moving the output products into a chunk-parallel kernel. The cost is storing the per-chunk states: 32 × 48 ×
  64 KB = 100 MB at T = 2048, 400 MB at 8192. The estimated gain is about 1-2 % of 27B prefill.
* **Chunked prefill with batch 512.** The 27B's default `--batch 512` cuts long prompts into 512-row passes. Each pass starts
  the chunked form from the carried state, which is correct, but the per-pass fixed costs (3 launches) are paid per 512
  rows.

## 19. Small-batch decode: skinny tensor-core GEMM, fp8 / int8 GEMM from 17 / 33 rows, barrier batching (2026-11-14)

Levers 1-3 of section 16.8, generic at kernel-selection level (no model-specific code). Kernels:
`src/Sharplight.Shaders/Kernels/matmul_skinny.slang` (new, `gemm_sk`); lowering: `GraphLowerer.SkinnyGemm.cs` (new),
`GraphLowerer.MatMul.cs`, `GraphLowerer.IntDot.cs`, `GraphLowerer.Mmq.cs`; recording: `HazardTracker.Order`,
`VulkanSegment.Record`. Raw results: [`bench/results/s19/`](../bench/results/s19) (`batch/` BatchBench and per-kernel
profiles, `spec/` SpecBench, serving at the top level, `disturbed/` a first pass that ran while another application used the GPU).

### 19.1 What changed

* **Skinny tensor-core GEMM (2-16 rows).** One workgroup owns 16-32 weight rows. Its 4-8 subgroups split K, each staging its
  own A / B slices in shared memory (subgroup barriers only, next block prefetched into registers), and they sum their
  partial 16x16 f32 tiles through shared memory at the end. Every weight is read once, with no split-K scratch and no
  reduction launch. Weights are decoded to f16 by the gemm_cm2 decoders, so every format gemm_cm2 reads works (f16 direct,
  bf16, FP8 with block / channel scales, NVFP4, Q8_0, K-quants, ...). The fused gate/up pair runs as one dual launch with
  the SiLU·mul epilogue. Device requirements: `matrix_f16`, `float16`, 32-wide subgroups, K % 32 = 0. Otherwise, and with
  `SHARPLIGHT_SKINNY_GEMM=0`, the previous kernels run.
  * Formats without an int-dot matvec (f16, bf16, FP8, NVFP4, ...): 2-16 rows.
  * Q8_0 and the K-quants: the rows above the int-dot matvec (9-16). Q8_0's int-dot range drops from 16 to 8 rows where the
    tensor-core path is usable (Q8_0 [17408→5120] at 12 rows: 100 µs int-dot vs 72 µs skinny).
  * Narrow matrices keep the split-K kernels, which spread K over more workgroups. The skinny grid has N / 16-32
    workgroups and costs a fixed ~20 µs at K = 5120 (FP8 [5120→48] at 4 rows: 24 µs vs 8 µs for the split-K matvec). N <
    1024: never; N < 4096: up to 8 rows.
  * Activations: one f16 copy of x per input, shared by every matmul that reads it (Q/K/V, gate/up, the delta-rule input
    projections) and by gemm_cm2 above 16 rows (`QuantizedActivations` with `ActivationCode.F16`). Reading f32 rows
    instead (`SHARPLIGHT_SKINNY_X16=0`) is slower even at 2 rows (FP8 [17408→5120]: 75 vs 60 µs), because every workgroup
    re-reads x.
* **fp8 GEMM from 17 rows, int8 GEMM from 33 rows** (both were 65). `gemm_fp8` at 17-64 rows takes 67-80 µs on
  [5120→17408] vs 76-116 µs for gemm_cm2. MMQ from 33 rows: IQ4_XS 145 vs 310 µs, Q8_0 about 10 % faster
  (`SHARPLIGHT_FP8_GEMM_MIN_ROWS`, `SHARPLIGHT_MMQ_MIN_ROWS`).
* **Barrier batching.** `HazardTracker.Order` fills each barrier batch with later launches that conflict neither with the
  batch nor with launches deferred past it (32-launch lookahead), so the conversion / GEMM / reduction chains of sibling
  matmuls share barriers. Every read-after-write, write-after-read and write-after-write order is kept. Split-K scratch now
  rotates over three buffers; with one shared buffer the siblings serialised on it. `SHARPLIGHT_VK_REORDER=0` keeps
  program order (profiling also keeps it).

**Precision.** 2-16 rows now multiply f16 activations on tensor cores with f32 accumulation, as 9+ rows did before. FP8 at
17-64 rows uses E4M3 activations with one f32 scale per 128 values, as 65+ rows did before (and as vLLM does at every batch
size). Single rows keep the f32 matvec. Greedy CLI output (200 tokens, chat prompt) is token-identical before and after on
both the FP8 checkpoint and the Q8_0 GGUF.

### 19.2 Per-row-count step time, Qwen3.8-27B

BatchBench, `--ctx 2048 --modes greedy`. Minimum GPU ms per step (timestamps) over 30 steps and 2 interleaved process runs
per build; base = `main` before this section. The last column gives FP8 dispatches / barriers per step.

| rows | FP8 base | FP8 new | speedup | Q8_0 base | Q8_0 new | speedup | FP8 dispatches / barriers, base → new |
|---|---|---|---|---|---|---|---|
| 1 | 23.8 | 23.0 | 1.04x | 23.3 | 22.3 | 1.05x | 901 / 596 → 901 / 596 |
| 2 | 38.0 | 28.2 | 1.35x | 20.2 | 19.8 | 1.02x | 901 / 596 → 1206 / 853 |
| 3 | 51.1 | 29.1 | 1.75x | 22.3 | 22.4 | 0.99x | |
| 4 | 63.2 | 29.6 | 2.13x | 22.5 | 22.7 | 0.99x | |
| 8 | 115.3 | 31.4 | 3.67x | 28.3 | 28.5 | 0.99x | |
| 12 | 37.7 | 32.8 | 1.15x | 41.5 | 33.6 | 1.23x | 2022 / 1653 → 1334 / 917 |
| 16 | 39.0 | 34.7 | 1.12x | 41.4 | 35.3 | 1.17x | |
| 32 | 45.2 | 35.8 | 1.26x | 46.6 | 44.8 | 1.04x | 2022 / 1653 → 1830 / 1269 |
| 64 | 65.0 | 48.0 | 1.35x | 63.7 | 51.3 | 1.24x | |

The FP8 small-batch cliff is gone: the 8-row step went from 4.8x to 1.37x the 1-row step. Q8_0 up to 8 rows is unchanged
(int-dot matvec). There, 2-row steps are faster than 1-row steps because single rows use the f32 matvec, which keeps the
activations exact.

GPU-time profile of the FP8 step (`batch/prof19_fp8_*.txt`, 20 steps, ms per step):

| rows | build | dense GEMM / matvec | GEMM glue (to_f16 / quantize / split-K reduce) | recurrent | other | total |
|---|---|---|---|---|---|---|
| 4 | base | 61.9 | 0 | 0.9 | 1.9 | 64.7 |
| 4 | new | 26.3 | 1.2 | 0.9 | 2.0 | 30.4 |
| 16 | base | 32.7 | 3.8 | 2.1 | 2.3 | 40.9 |
| 16 | new | 29.6 | 1.7 | 2.3 | 2.0 | 35.7 |
| 64 | base | 44.5 | 4.6 | 12.2 | 3.9 | 65.2 |
| 64 | new | 31.2 | 3.6 | 12.6 | 3.8 | 51.2 |

At 4 rows the fused gate/up pair [5120→17408]x2 takes 125 µs (178 MB, 1.4 TB/s) and the down projection 92 µs (89 MB,
1.0 TB/s); the 1-row matvec reaches 1.61 TB/s.

### 19.3 Serving against vLLM, Qwen3.8-27B

`bench/s16_serve.sh` as in section 16.4 (~250-token prompts, 128 output tokens, ignore_eos, 2 rounds). All runs fell in one
period on an otherwise idle GPU (`*.gpu.txt`): Sharplight base and new on the FP8 checkpoint and the Q8_0 GGUF, and vLLM
0.33 on the FP8 checkpoint (identical weights). Aggregate tok/s, median of 2 rounds; mean ITL in ms.

| conc | vLLM FP8 | SL FP8 base | SL FP8 new | new / base | **new / vLLM** (§16.4) | SL Q8_0 base | SL Q8_0 new | FP8 ITL base → new | vLLM ITL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 45 | 38 | 40 | 1.05x | 0.89x (0.88x) | 40 | 39 | 25.9 → 24.8 | 22.2 |
| 4 | 157 | 59 | 123 | **2.08x** | **0.78x** (0.38x) | 153 | 149 | 66.6 → 31.2 | 24.4 |
| 16 | 525 | 331 | 359 | 1.08x | 0.68x (0.61x) | 299 | 332 | 45.0 → 41.5 | 26.0 |
| 64 | 1100 | 646 | 747 | 1.16x | 0.68x (0.54x) | 581 | 634 | 87.5 → 74.6 | 42.5 |

* **c = 4 doubles.** Its steps are 4 decode rows plus prefill chunks.
* **c = 16 / 64 gain less.** Their steps average 46 / 172 rows (decodes plus 250-row prompt chunks in the same step), so
  most of their GEMM work runs on the prefill-sized kernels. The pure 64-row decode step is now 48 ms of GPU time against
  vLLM's 42 ms ITL.
* **Q8_0** gains 11 % at c = 16 and 9 % at c = 64 (9-16-row and 33-64-row steps). c = 1 / 4 are within run-to-run noise
  (−1 to −3 %).
* **Single stream is unchanged**, since 1-row steps keep the matvec. SpecBench k = 0 below gives 39.1 vs 40.4 tok/s at
  depth 47 and 38.1 vs 38.7 at 4096; BatchBench gives 23.0 vs 23.8 ms per 1-row step.

### 19.4 MTP speculative decoding on FP8

SpecBench as in section 17.1 (natural-language prompt, greedy, tg = 128 / (t(129) − t(1)), median of 5), with base and new
alternating per draft count on an idle GPU (`spec/`). vLLM column: section 16.5 (same checkpoint and prompt, earlier
period).

| k | base @d47 | new @d47 | new: speedup over k = 0 | base @d4096 | new @d4096 | acceptance (new, d47 / d4096) | vLLM @d47 |
|---|---|---|---|---|---|---|---|
| off | 40.4 | 39.1 | | 38.7 | 38.1 | | 38.4 |
| 1 | 42.3 | 55.9 | 1.43x | 41.7 | 53.8 | 0.84 / 0.83 | 60.5 |
| 2 | 38.7 | 70.4 | 1.80x | 40.1 | 61.2 | 0.74 / 0.72 | 70.6 |
| 3 | 36.5 | **75.3** | **1.93x** | 36.5 | 68.7 | 0.62 / 0.61 | 74.7 |

MTP now pays on FP8 weights as it does on Q8_0 (section 17.1). At k = 3 it reaches vLLM's MTP throughput on the same
checkpoint (75.3 vs 74.7 tok/s); before, verification steps of 2-4 rows cost more than the accepted drafts saved.

### 19.5 Commands and tests

```bash
dotnet build Sharplight.slnx -c Release && dotnet build bench/BatchBench -c Release && dotnet build bench/SpecBench -c Release
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,2,3,4,8,12,16,32,64
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,4,16,64 --profile 4,16,64 --profile-mode greedy
python3 bench/profile_categories.py <profile output>
bench/SpecBench/bin/Release/net11.0/SpecBench <Qwen3.8-27B-FP8 dir> --drafts 0,1,2,3 --depths 47,4096
OUT_DIR=bench/results/s19 PORT=18190 bench/s16_serve.sh sharplight qwen27b-fp8-new <dir> "1 4 16 64"
OUT_DIR=bench/results/s19 PORT=18190 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
python3 bench/s16_summarize.py bench/results/s19; python3 -I bench/sanitize_results.py bench/results/s19
# kernel microbenchmarks (rows, weight types, K,N shapes)
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_ROWS=4 SHARPLIGHT_BENCH_TYPES=f8e4m3_b128 SHARPLIGHT_BENCH_SHAPES="5120,17408;17408,5120" \
  dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll -method '*MatMulPrefill'
```

Tests: `MatMulTests` (FP8 b128 / b32, NVFP4 and the other formats at 1-64 rows; a shared f16 copy across three matmuls),
`FusionTests` (dual gate/up, including f16 and FP8 at 16 rows), `QuantFormatTests` and `Fp8GemmTests` (17 and 40 rows).
They pass on the NVIDIA GPU, on the AMD iGPU (no cooperative matrices, so the fallback paths) and on llvmpipe.

### 19.6 Open issues

* **17-64-row GEMM bandwidth.** At 64 rows the fp8 GEMM runs the gate/up pair as two split-K launches (88 µs each, about
  1.0 TB/s) plus reductions. A dual fp8 GEMM with the SiLU·mul epilogue, in a stream-K or K-parallel layout like the
  skinny kernel's, would approach the 1.6 TB/s roof. bf16 / f16 weights still use gemm_cm2 at 17-64 rows (~1.3 TB/s).
* **Skinny GEMM at 1.0-1.4 TB/s.** Each workgroup re-reads the activations, and the subgroups' partial tiles go through
  shared memory. Next steps: larger K blocks with multi-stage prefetch, and splitting K across two workgroups for narrow
  matrices instead of falling back to split-K.
* **Delta rule at 64 decode rows** is about a quarter of the FP8 step (232 µs per layer). 64 sequences read and write
  3 MB of f32 state each per layer, at 1.7 TB/s. Only a smaller state type would reduce it (bf16 state, as an opt-in).
* **Glue launches.** f16 copies and split-K reductions are 4-7 % of the batched step. Producing the f16 copy in the
  preceding RmsNorm / elementwise epilogue would save one launch and one barrier per matmul input.
* **Serving at 16-64 streams** is at 0.68x of vLLM. The decode step is close (48 vs ~42 ms at 64 rows); the rest of the gap
  is in mixed prefill / decode steps, where a 512-row prompt chunk shares the step with the decodes (TTFT p95 4.6 s vs
  vLLM's 2.4 s at c = 64).

## 20. Mixed prefill / decode steps, fp8 GEMM rework, dual fp8 gate/up GEMM (2026-11-14)

Follow-up to section 19.6. Everything is generic at kernel-selection level. Kernels: `attention.slang` (`MIX_RUN`),
`recurrent.slang` (multi-sequence `conv1d_tokens`) and `gemm_fp8.slang` (accumulators, `DUAL`, grid order). Lowering:
`GraphLowerer.Attention.cs`, `GraphLowerer.Recurrent.cs`, `GraphLowerer.Fp8Gemm.cs`, `GraphLowerer.MatMul.cs` and
`GraphLowerer.Mmq.cs`. BatchBench gained `--mixed D+P[xN],...`, which times steps of D decode rows plus N prompt chunks
of P rows on fresh sequences, and `--batch`, which sets the rows per step.

Raw results are in [`bench/results/s20/`](../bench/results/s20):

* `batch/`: BatchBench and profiles.
* `spec/`: SpecBench.
* Top level: serving.
* `disturbed/`: runs that overlapped other GPU work and were repeated. The user's applications, including a Sharplight
  Studio session at 100 % GPU, ran during parts of this work. Every serving run now samples the GPU's other compute
  applications and is repeated when one was busy.
* `disturbed/pre-identity-fix/`: runs of an intermediate build, described in 20.2.

### 20.1 Where the serving time goes, and why the scheduler is unchanged

`BatchBench --mixed` times steps that carry D decode rows plus prompt chunks. On the 27B FP8 (`main`), a step costs the
decode-only step plus 0.23-0.30 ms of GPU time per prompt row. The chunk size (64-2047 rows) and the number of prompts in
the step (1-8) hardly matter (`batch/mixed_fp8_base_costmodel.txt`):

| decode rows | prompt rows (prompts) | GPU ms | ms per prompt row over decode-only |
|---|---|---|---|
| 64 | 0 | 45.4 | |
| 64 | 64 (1) / 448 (1) / 1984 (1) | 64.8 / 180.6 / 617.6 | 0.304 / 0.302 / 0.288 |
| 64 | 1000 (4) / 1750 (7) | 338.3 / 560.7 | 0.293 / 0.294 |
| 16 | 240 (1) / 1008 (1) / 2000 (8) | 89.9 / 296.6 / 571.0 | 0.229 / 0.260 / 0.268 |
| 1 | 511 (1) / 2047 (1) | 148.7 / 579.8 | 0.244 / 0.272 |

The total GPU time of a serving run is therefore about (steps × decode-step time) + (prompt rows × per-row cost). The
token budget per step (`--step-tokens`, `EngineOptions.MaxBatchTokens`) and the chunk size (`--prefill-chunk`,
`EngineOptions.PrefillChunk`) only move work between steps. They trade TTFT against ITL tails but cannot change throughput.

The closed-loop c = 64 run fits this model: 422 steps × ~50 ms plus 48 000 prompt rows × 0.28 ms is about 34.5 s, against
34.2 s measured. It runs in waves, because all 64 requests start and finish together: a few prefill-heavy steps, then
about 128 decode steps. vLLM (chunked prefill, `max_num_batched_tokens` 8192) has the same structure, 390 iterations at
about 42 ms per 64-row step. Its prompt rows cost roughly half as much as Sharplight's, which is also why its c = 64
TTFT p95 is about half of ours.

So there is no scheduler change and no new option. The levers are the per-row prefill cost and the decode step, both
kernel work. The two knobs stay in `EngineOptions` and the server; the shared hosting layer (Studio) keeps the defaults.

Two kernel-selection problems did show up in mixed steps, and both are fixed:

* **Attention.** A step with decode rows next to a prompt chunk (more than 64 rows, fewer than 8 rows per run) fell to the
  scalar per-head kernel: 1.09 ms per attention layer at 64 decode rows plus a 448-row chunk. Such steps now launch the
  split-K decode kernel for runs shorter than 16 rows and flash attention for the longer runs (80 + 127 µs).
  * The decode kernel uses one key range per row and writes its output directly, so no combine is needed.
  * Each kernel skips the other's rows (`MIX_RUN`).
  * Steps that flash attention covers alone keep flash only.
  * `SHARPLIGHT_ATTN_MIXED=0` restores one kernel per step.
* **conv1d.** The multi-sequence `conv1d_tokens` grid had one z-slice per run over the whole batch's row tiles, so 65
  runs launched 65x the blocks (184 µs per layer). It now tiles the batch's rows once (16 µs). Each block looks up its run
  by binary search and moves its sliding window to the next run where one begins.

### 20.2 fp8 GEMM

* **Per-column-fragment block accumulators.** The per-weight-block f32 accumulator now holds one column fragment at a
  time (`F8_FM` fragments instead of `F8_FM × F8_FN`). This frees registers and is 10-13 % faster from 256 rows. 27B
  shapes at T = 1024, in TFLOPS:

  | shape | before | after |
  |---|---|---|
  | [5120→17408] | 257 | 281 |
  | [17408→5120] | 233 | 261 |
  | [5120→6144] | 198 | 223 |

* **Token tiles first.** Consecutive workgroups now share a weight tile (grid x = token tiles). Each weight byte is then
  fetched from memory about once, even when the matrix does not fit the 128 MB L2: [17408→5120] at T = 2048 goes from
  1.93 to 1.44 ms. The same order on the int8 MMQ GEMM made no difference and is not used.
* **Dual fp8 GEMM for gate / up pairs at 17-64 rows** (one 64-row tile). One launch reads the E4M3 activations once and
  multiplies both weight matrices. Their 16-row fragments alternate in the B tile, so one thread holds gate and up of the
  same output and stores SiLU(gate)·up as the epilogue. Before, this was two GEMMs, each with split-K partials and a
  reduction, plus an elementwise launch. On the 27B: 139 µs at 32 rows (vs 2 × 84 µs plus reductions) and 147.5 µs at
  64 rows (vs 2 × 89 µs). `SHARPLIGHT_FP8_DUAL=0` restores the separate launches.
  * An intermediate build (`disturbed/pre-identity-fix/`) also ran it above 64 rows. Prefill was 4-6 % faster (pp512
    4375 vs 4212, pp2048 4344 vs 4114 tok/s), but the fused epilogue rounds differently from the separate launches. That changed greedy output
    on a 636-token FP8 prompt after about 120 tokens (a near tie). Prompt prefill therefore keeps the separate GEMMs, and
    stays bit-identical to `main` (see 20.7).
* **What limits the kernel** (recorded in `gemm_fp8.slang`). Without the global loads, the MMAs plus shared-memory
  traffic run at about 520 TFLOPS, against 282 for the whole kernel at T = 512. These were tried and are no faster:
  * double-buffered shared memory;
  * 64-row tiles for large T;
  * 128×256 tiles (register spills);
  * a deeper prefetch in the skinny GEMM.

Precision is unchanged: E4M3 activations with one f32 scale per 128 values from 17 rows (section 19.1), and the f32
matvec for single rows.

### 20.3 Step times

BatchBench on the 27B: minimum GPU ms per step (timestamps) over 2 interleaved process runs per build, on an otherwise
idle GPU; base = `main`.

Decode rows only (`--ctx 2048 --sizes ...`), FP8:

| rows | base | new | dispatches / barriers, base → new |
|---|---|---|---|
| 1 | 22.1 | 22.8 | 901 / 596 → 901 / 596 |
| 4 | 29.0 | 29.1 | 1206 / 853 (unchanged) |
| 16 | 33.8 | 33.9 | 1334 / 917 (unchanged) |
| 32 | 34.8 | 34.1 | 1830 / 1269 → 1574 / 1141 |
| 64 | 45.6 | 44.2 | 1830 / 1269 → 1574 / 1141 |

Up to 16 rows the kernels are unchanged, so differences there are noise (±3 %). At 32 / 64 rows the dual GEMM saves 2-3 %
and 256 launches / 128 barriers per step.

Mixed steps (`--slots 72 --ctx 4096 --batch 2048 --mixed ...`, final build). "Prompt rows" are fresh sequences next to the
decode rows:

| decode + prompt rows (prompts) | FP8 base | FP8 new | speedup | Q8_0 base | Q8_0 new | speedup |
|---|---|---|---|---|---|---|
| 64 + 64 (1) | 64.2 | 61.6 | 1.04x | 75.9 | 73.0 | 1.04x |
| 64 + 448 (1) | 179.4 | 162.4 | **1.10x** | 215.7 | 199.8 | **1.08x** |
| 64 + 1000 (4) | 327.9 | 300.5 | 1.09x | 405.7 | 401.4 | 1.01x |
| 16 + 240 (1) | 85.8 | 86.8 | 0.99x | 105.6 | 105.8 | 1.00x |
| 16 + 1000 (4) | 284.0 | 264.1 | 1.08x | 351.2 | 349.2 | 1.01x |
| 4 + 508 (1) | 145.7 | 148.6 | 0.98x | 181.9 | 181.7 | 1.00x |
| 1 + 2047 (1) | 566.5 | 544.1 | 1.04x | 719.8 | 699.3 | 1.03x |

GPU-time profile of the FP8 64 + 448 step (ms per 10 steps; `batch/prof_mixed_fp8_base.txt` vs
`prof_mixed_fp8_final.txt`):

| category | base | new |
|---|---|---|
| attention | 190.7 | 37.1 |
| recurrent | 219.4 | 133.9 |
| dense GEMM | 1269.2 | 1199.3 |
| total | 1891.6 | 1589.1 |

The profiled steps are a little slower than unprofiled ones because profiling keeps program order.

### 20.4 Prefill

`sharplight --bench N --batch N` on the FP8 checkpoint, best of 3 interleaved runs per build (`prefill.txt`); vLLM from
section 16.3.

| | base | new | speedup | vLLM | new / vLLM |
|---|---|---|---|---|---|
| pp512 | 4038 | 4212 | 1.04x | 5171 | 0.81x |
| pp2048 | 3781 | 4114 | **1.09x** | 5944 | **0.69x** |

Section 16.3 had 0.55x / 0.51x; section 18's chunked delta rule and this section's GEMM work closed part of that. Q8_0
prefill (int8 MMQ) is unchanged.

### 20.5 Serving against vLLM

`bench/s16_serve.sh` as in section 19.3: ~250-token prompts, 128 output tokens, ignore_eos, a 64-slot server, 2 rounds.
Each concurrency got its own server start, and base, new and vLLM 0.33 alternated. Values are aggregate tok/s, TTFT
P50 / P95 in ms and mean ITL in ms, each the mean of 2 rounds; c = 16 is the mean of 4, because single runs varied by ±5 %
(337-362 tok/s for either build).

| conc | vLLM FP8 | SL base | SL new | new / base | SL / vLLM, base → new | TTFT P50 / P95, SL base → new | vLLM TTFT | ITL SL base → new | vLLM ITL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 43 | 38 | 38 | 1.00x | 0.87x → **0.87x** | 87 / 88 → 82 / 83 | 60 / 62 | 26.0 → 26.1 | 22.7 |
| 4 | 154 | 123 | 124 | 1.01x | 0.80x → **0.80x** | 226 / 260 → 216 / 261 | 183 / 193 | 31.2 → 30.9 | 24.9 |
| 16 | 562 | 350 | 352 | 1.01x | 0.62x → **0.63x** | 304 / 1013 → 296 / 1035 | 574 / 579 | 42.6 → 42.3 | 24.3 |
| 64 | 1095 | 704 | 741 | **1.05x** | 0.64x → **0.68x** | 546 / 4977 → 512 / 4775 | 1855 / 2505 | 79.0 → 74.8 | 42.8 |

* **c = 64: +5 %**, with TTFT P95 down by 0.2 s and mean ITL down by 4 ms. Its prefill-heavy steps run the faster fp8 GEMMs; its decode steps run
  the dual GEMM at 64 rows.
* **c = 1 / 4 / 16 are unchanged.** Their decode steps of 1-16 rows use kernels this section does not touch, and the 1-row
  matvec is already at about 1.6 TB/s.
* **Absolute numbers are a few percent below section 19.3** for both base and new, on a warmer, power-capped GPU (300 W,
  83-89 °C during the runs). vLLM measured the same as in section 19. The ratios to vLLM above are therefore slightly
  lower than in section 19 for the same code: base 0.64x at c = 64 here vs 0.68x there.

### 20.6 MTP on FP8

SpecBench as in section 19.4 (natural-language prompt, greedy, median of 5), base and new run back to back
(`spec/`). The new build here is the intermediate one of 20.2, which differs from the final build only in prefill above 64
rows; the verification steps run the same kernels. Values are tok/s:

| k | base @d47 | new @d47 | base @d4096 | new @d4096 | acceptance (new, d47 / d4096) |
|---|---|---|---|---|---|
| off | 38.1 | 38.7 | 37.6 | 38.0 | |
| 1 | 54.6 | 58.2 | 52.7 | 54.7 | 0.86 / 0.80 |
| 2 | 70.2 | 73.8 | 62.6 | 70.2 | 0.78 / 0.73 |
| 3 | 71.3 | 74.1 | 65.5 | 67.8 | 0.57 / 0.57 |

Verification steps of 2-4 rows use the same kernels as before. The 3-5 % differences are within this period's run-to-run
spread: k = 3 at d47 ranged 64-81 tok/s per run.

### 20.7 Correctness

* Greedy CLI output (200 tokens, a 636-token prompt and a short chat prompt) is token-identical between `main` and this
  build on the FP8 checkpoint and on the Q8_0 GGUF.
* With `--speculate-mtp 3` the output is identical to non-speculative decoding, on both builds and both formats.

Two earlier versions broke this and were changed, not tolerated:

* the dual fp8 GEMM above 64 rows (20.2);
* a `conv1d_tokens` that summed each output via direct loads instead of the sliding window. The compiler rounds the two
  differently, and that changed MTP output on the Q8_0.

Tests:

* `FusionTests`: dual gate/up at 1-140 rows, including 70 / 140 rows of FP8.
* `MultiSequenceTests`: a step of more than 64 rows that mixes decode rows with prompt chunks.
* `Fp8GemmTests`, `MatMulTests`, `PagedAttentionTests`.

The Cpu, Vulkan, Serving and Studio suites pass on the rebased branch (the Families run was interrupted). `MultiSequenceTests` for qwen2.5-0.5b on the CPU provider
fail on `main` as well (rms 1.7e-3 against a 1e-3 bound at step 2).

### 20.8 Commands

```bash
dotnet build Sharplight.slnx -c Release && dotnet build bench/BatchBench -c Release && dotnet build bench/SpecBench -c Release
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,2,3,4,8,12,16,32,64
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --slots 72 --ctx 4096 --batch 2048 --modes greedy --reps 5 \
  --mixed 64+0,64+64,64+448,64+250x4,16+0,16+240,16+250x4,4+0,4+508,1+0,1+511,1+2047
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --slots 72 --ctx 4096 --batch 2048 --mixed 64+448,1+2047 --profile 0,1
samples/Sharplight.Cli/bin/Release/net11.0/sharplight <Qwen3.8-27B-FP8 dir> --bench 2048 --ctx 2304 --batch 2048
bench/SpecBench/bin/Release/net11.0/SpecBench <Qwen3.8-27B-FP8 dir> --drafts 0,1,2,3 --depths 47,4096
OUT_DIR=bench/results/s20 PORT=18190 SLOTS=64 bench/s16_serve.sh sharplight qwen27b-fp8-new <dir> "64"   # one concurrency per server
OUT_DIR=bench/results/s20 PORT=18190 SLOTS=64 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "64" --gpu-memory-utilization 0.5
```

### 20.9 Open issues

* **Per-row prefill cost** decides serving at 16-64 streams and its TTFT (20.1). It is about 0.24 ms per row here against
  about 0.17 ms for vLLM at pp2048, and GEMMs are 79 % of it.
  * The fp8 GEMM is bound by global-memory fetch, at 282 TFLOPS against about 520 without it. A TMA-like asynchronous
    copy, or a workgroup-scope 32×32 fp8 MMA with fewer, larger loads, is the next thing to try.
  * The dual GEMM above 64 rows was worth 4-6 % of prefill but must round like the separate launches to keep greedy output
    unchanged. Bit-matching its epilogue would recover that gain.
* **16-row decode step** (33.9 ms vs vLLM's 24.3 ms ITL at c = 16). The skinny GEMM reaches 0.8-1.2 TB/s at 16 rows,
  against 1.6 TB/s for the matvec, because every workgroup re-reads x. The narrow [5120→48] bf16 projections still take
  split-K launches.
* **Delta rule at 64 decode rows** (section 19.6): unchanged, about a quarter of the step. An opt-in bf16 state behind
  `--reduced-precision` is the remaining lever.
* **Glue launches**: f16 copies, fp8 quantization and split-K reductions are 3 % of the mixed step.

## 21. ONNX models against onnxruntime: CPU and GPU (2026-11-14)

The goal is to run 15 representative ONNX models, all imported through `Sharplight.Import.Onnx`, as fast as onnxruntime on the
same machine. Sharplight's CPU provider is compared with ORT's CPU EP, and the Vulkan provider with ORT's CUDA EP on the
same GPU. The optimizations work at graph, op and kernel level. None of them is specific to a model.

| | |
|---|---|
| Machine | Ryzen 9 9950X (16 cores, AVX-512) and RTX PRO 6000 Blackwell Max-Q (driver 610.57.04), as in section 1 |
| onnxruntime | 1.31.1 (`onnxruntime-gpu` wheel with CUDA 13 / cuDNN 9; venv set up by `bench/onnx/env.sh`), `ORT_ENABLE_ALL` graph optimizations |
| CPU | both engines pinned to cores 0-15 (`taskset`). They use 16 threads: `SHARPLIGHT_CPU_THREADS=16` for Sharplight, `intra_op_num_threads=16` for ORT |
| GPU | `ort-cuda` is the CUDA EP default, which allows **TF32** for f32 matmuls and convolutions. `ort-cuda-strict` sets `use_tf32=0`. Inputs are bound on the device with IO binding, and outputs are copied back |
| Sharplight GPU | `sl-vulkan` uses the defaults (f32-accurate). `sl-vulkan-rp` adds `AllowReducedPrecision` and `AllowHalfBlockScales` (opt-in, section 21.3). Ops without a Vulkan kernel (NMS) run on the CPU provider as a fallback segment |
| Timing | every cell is its own process. After warm-up, each process reports min / median of N runs (N ≥ 20, or a 10 s budget). Engines run interleaved per model, and the table gives the minimum of the per-process minima over 2 rounds, then the median of medians |

Scripts are in `bench/onnx/`:

* `OnnxBench/` (C#): imports a reference manifest, compiles for CPU or Vulkan, and times runs. It reports GPU timestamps
  and an optional profile, and checks outputs against the manifest's ORT reference.
* `ort_latency.py`: times the same model under ORT.
* `parity_bench.py`: runs both and prints the tables below.

Model paths come from `SHARPLIGHT_ONNX_MODELS`. Raw JSONL and logs are in [`bench/results/s21/`](../bench/results/s21):

* `before.jsonl`: the baseline at `4f192ee`.
* `after-cpu.jsonl`, `after-gpu.jsonl`: this branch.
* `disturbed/`: runs that overlapped another agent's GPU benchmark. They were repeated and are not used in the tables.

### 21.1 What changed

Commits on the branch, in order:

* **Benchmark harness** (`bench/onnx/OnnxBench`, `ort_latency.py`, `parity_bench.py`). `OnnxBench` replaces SurveyRunner,
  which does not build under the code-style analyzers. It tries a symbolic batch first and falls back to a fixed batch.
  On Vulkan it attaches the CPU provider as a fallback device, so ssd-10's NonMaxSuppression now runs as a CPU segment
  instead of failing to compile.
* **MatMulNBits in block form** (`MatMulNBitsPacking.cs`). The weights used to be dequantized to f32 at import. Now they
  are packed into ggml Q4_0 / Q8_0 blocks (Q4_1 when zero points are present), so the existing quantized matvec / GEMM
  kernels run them. Rules:
  * Packing happens only when it is exact: the block scales must be representable in f16, as for phi-3.5 q4f16.
  * Models with f32 scales (bert / qwen `model_q4`) keep the f32 path unless the new opt-in
    `OnnxImportOptions.AllowHalfBlockScales` is set. Rounding those scales costs about 5e-4 relative RMS on BERT, which
    is above the conformance tolerance.
  * The MatMuls carry a new `f32_activations` attribute (`StandardOps.F32ActivationsAttribute`) unless the node has
    `accuracy_level` 4. The attribute keeps the int8-activation paths (CPU `IntDot`, Vulkan int-dot matvec and MMQ) and
    the Vulkan f16-activation tensor-core GEMMs off unless `AllowReducedPrecision` is set.
* **CPU conv** (`ConvOp`, `OutputEpilogue`, `CpuCompiledSegment.Fusion.cs`):
  * A residual `Add` and one activation are applied to each output tile while it is in cache. This covers
    Relu / Clip / LeakyRelu / HardSwish / Sigmoid / SiLU / GELU / Tanh / ..., and the `Mul(x, Sigmoid(x))` SiLU of YOLO
    and EfficientNet exports.
  * Convs with ≤ 1 MB of weights and K ≤ 1152 pack each output tile's im2col panel per worker instead of the whole B
    matrix up front.
  * Depthwise rows use AVX-512 / AVX2 lanes over the interior, for stride 1 and stride 2 (even-lane permute). Narrow maps
    keep the AXPY rows.
* **CPU MatMul** (`MatMulOp`):
  * Static f32 weights up to 64 MB are packed into the GEMM's B layout once, on the first GEMM run. Before, they were
    packed on every run and for every row block.
  * The same epilogue also takes a constant bias vector over the last axis, so `MatMul → Add(bias) → Add(residual) → act`
    is one op.
* **ONNX translation**: the decomposed LayerNorm of opset < 17 exports (ReduceMean / Sub / Pow / ReduceMean / Add /
  Sqrt / Div, plus the affine Mul / Add) and the decomposed erf GELU (`x · 0.5 · (1 + erf(x/√2))`, any grouping) each
  become one node.
  * They are matched backwards from the last node, so constants defined after the first ReduceMean are known.
  * The elementary nodes are left to dead-node elimination.
  * vit-base drops from 608 to 408 nodes and bert-base from 635 to 427.
* **Vulkan bmm**: grids under 512 32x32 tiles use 2x2 outputs per thread (256-thread groups). Attention P·V at batch 1
  (12 x 197x197 · 197x64) drops from 43 µs to 16 µs.

New switches:

* `SHARPLIGHT_CPU_NO_FUSION=1`: no conv / matmul epilogues.
* `SHARPLIGHT_CPU_NO_PREPACK=1`: no once-only weight packing.
* `OnnxImportOptions.AllowHalfBlockScales`: opt-in, described above.

### 21.2 CPU: Sharplight vs onnxruntime CPU EP (16 threads)

Before = `4f192ee`, after = this branch. Times are in ms: min of the per-process minima over 2 rounds, then the median
of medians. The CPU was shared with other agents' builds and tests (load average 9-13), so the medians are noisy. Use
the minima. Cells read `min / median`, before → after. `(!)` marks a run whose outputs failed the parity check
against the ORT reference. The ratio column is Sharplight's after-min divided by ORT's min.

| model | batch | sl-cpu | ort-cpu | SL/ORT CPU (min) |
|---|---|---|---|---|
| resnet50 | 1 | 12.48 / 43.62 → 8.16 / 14.39 | 5.71 / 6.62 | 1.43x |
| resnet50 | 8 | 80.81 / 86.00 → 54.53 / 62.35 | 20.30 / 22.60 | 2.69x |
| convnext-tiny | 1 | 18.95 / 22.57 → 20.05 / 90.98 | 13.16 / 13.86 | 1.52x |
| convnext-tiny | 8 | 163.18 / 173.72 → 89.23 / 108.11 | 79.93 / 96.84 | 1.12x |
| efficientnet-lite4 | 1 | 9.42 / 13.71 → 7.93 / 19.59 | 3.60 / 4.28 | 2.20x |
| mobilenetv2 | 1 | 6.53 / 7.53 → 2.06 / 11.18 | 1.56 / 1.70 | 1.32x |
| mobilenetv2 | 8 | 21.04 / 23.88 → 15.31 / 18.21 | 6.41 / 11.24 | 2.39x |
| yolov8n | 1 | 28.54 / 33.49 → 14.03 / 16.11 | 12.75 / 13.47 | 1.10x |
| yolov10n | 1 | 28.86 / 35.77 → 15.53 / 22.33 | 15.15 / 16.67 | 1.02x |
| ssd-10 (NMS) | 1 | 467.76 / 489.10 (!) → 305.86 / 355.81 (!) | 204.96 / 236.74 | 1.49x |
| bert-base (16 tok) | 1 | 13.16 / 28.71 → 11.40 / 22.40 | 11.38 / 21.82 | 1.00x |
| bert-base (16 tok) | 8 | 33.57 / 53.98 → 22.23 / 26.59 | 26.68 / 27.60 | 0.83x |
| vit-base | 1 | 37.69 / 41.33 → 26.13 / 28.06 | 21.16 / 22.48 | 1.24x |
| vit-base | 8 | 258.10 / 299.06 → 164.52 / 173.98 | 130.62 / 135.67 | 1.26x |
| whisper-tiny-encoder | 1 | 59.13 / 65.97 → 44.89 / 59.24 | 39.45 / 45.53 | 1.14x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 40.74 / 50.44 → 16.76 / 26.65 | 20.84 / 21.45 | 0.80x |
| phi3.5-mini q4f16 (GQA) | 1 | fail → 41.27 / 43.12 (!) | 50.87 / 51.83 | 0.81x |
| segformer-b0 | 1 | 92.78 / 103.16 → 91.64 / 105.64 | 41.40 / 44.12 | 2.21x |
| segformer-b0 | 8 | 560.67 / 619.11 → 423.83 / 465.26 | 476.53 / 531.81 | 0.89x |
| mobilesam-encoder | 1 | 395.02 / 639.59 → 215.44 / 393.41 | 196.34 / 209.50 | 1.10x |
| raft-large | 1 | 2239.13 / 2347.01 → 1803.68 / 2042.71 | 1546.91 / 1793.31 | 1.17x |

What moved:

* Convnets: resnet50 is 1.5x faster at b1, yolov8n / yolov10n are about 2x faster, and mobilenetv2 b1 is 3x faster.
  These gains come from the epilogues, tile packing and depthwise lanes.
* Transformers: vit is 1.4-1.6x faster, bert b8 1.5x and whisper 1.3x. These come from the decomposed-op translation,
  weight prepacking and the bias / GELU epilogues.
* MatMulNBits: phi-3.5 went from an out-of-memory failure (f32 dequantization of 3.8B parameters) to 41 ms, against
  ORT's 51 ms.

Sharplight is now at or ahead of ORT on bert, phi-3.5, qwen q4 and segformer b8. It is within 10-25 % on yolov8n,
yolov10n, whisper, vit and mobilesam.

### 21.3 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

Same format as 21.2. All GPU after-rows ran while no other agent's benchmark was on the GPU. `nvidia-smi` was sampled
every 15 s (`after-gpu.gpu-samples.txt`). The qwen / phi-3.5 rows are from a re-run after the fix described below. ORT's
own run-to-run spread is visible there: phi-3.5 `ort-cuda` measured 3.41 ms in the baseline and 4.19 ms here.

| model | batch | sl-vulkan | sl-vulkan-rp | ort-cuda | ort-cuda-strict | SL / ORT CUDA (TF32) |
|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.97 / 1.00 → 0.94 / 0.96 | 1.10 / 1.12 (!) → 1.08 / 1.11 (!) | 1.02 / 1.08 | 2.11 / 2.27 | 0.92x |
| resnet50 | 8 | 4.01 / 4.22 → 3.86 / 4.06 | 3.34 / 3.54 (!) → 3.17 / 3.38 (!) | 2.00 / 2.14 | 3.72 / 4.17 | 1.93x |
| convnext-tiny | 1 | 1.18 / 1.20 → 1.10 / 1.14 | 1.53 / 1.99 (!) → 1.43 / 1.50 (!) | 1.00 / 2.33 | 1.35 / 1.45 | 1.10x |
| convnext-tiny | 8 | 4.46 / 4.83 → 4.48 / 4.68 | 2.87 / 3.68 (!) → 2.76 / 2.87 (!) | 2.56 / 2.58 | 4.01 / 4.36 | 1.75x |
| efficientnet-lite4 | 1 | 0.94 / 0.97 → 0.95 / 0.97 | 1.05 / 3.57 → 1.05 / 1.09 | 0.68 / 0.89 | 8.18 / 8.88 | 1.41x |
| mobilenetv2 | 1 | 0.46 / 0.47 → 0.47 / 0.51 | 0.51 / 0.51 (!) → 0.51 / 0.55 (!) | 0.35 / 0.49 | 0.65 / 0.67 | 1.32x |
| mobilenetv2 | 8 | 0.92 / 0.94 → 0.92 / 1.01 | 0.95 / 0.96 (!) → 0.95 / 1.00 (!) | 0.75 / 0.90 | 1.35 / 1.39 | 1.22x |
| yolov8n | 1 | 1.48 / 2.01 → 1.46 / 1.53 | 1.90 / 1.95 (!) → 1.80 / 2.24 (!) | 1.31 / 1.34 | 2.75 / 3.00 | 1.11x |
| yolov10n | 1 | 1.43 / 1.82 → 1.34 / 1.36 | 1.82 / 1.92 (!) → 1.72 / 1.75 (!) | 1.17 / 1.21 | 2.31 / 2.51 | 1.15x |
| ssd-10 (NMS) | 1 | fail → 19.51 / 21.21 (!) | fail → 13.04 / 14.45 (!) | 10.99 / 12.07 | 22.54 / 24.80 | 1.78x |
| bert-base (16 tok) | 1 | 1.63 / 3.76 → 1.26 / 1.30 | 2.43 / 3.60 (!) → 1.70 / 1.84 (!) | 1.24 / 1.45 | 1.44 / 1.61 | 1.01x |
| bert-base (16 tok) | 8 | 4.57 / 8.35 → 3.72 / 4.03 | 4.53 / 4.86 (!) → 3.84 / 8.58 (!) | 2.70 / 4.84 | 3.32 / 4.43 | 1.38x |
| vit-base | 1 | 3.34 / 3.55 → 2.76 / 2.80 | 3.42 / 3.44 (!) → 2.38 / 2.41 (!) | 1.34 / 2.53 | 2.46 / 2.69 | 2.06x |
| vit-base | 8 | 13.53 / 14.97 → 12.75 / 14.73 | 5.82 / 10.82 (!) → 5.43 / 5.81 (!) | 3.98 / 4.33 | 9.23 / 9.99 | 3.21x |
| whisper-tiny-encoder | 1 | 2.90 / 3.13 → 2.84 / 3.02 | fail → fail | 1.17 / 1.32 | 2.08 / 2.15 | 2.42x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.56 / 3.77 → 3.19 / 4.16 | 3.59 / 7.04 → 2.50 / 4.29 (!) | 3.92 / 12.47 | 3.86 / 4.23 | 0.82x |
| phi3.5-mini q4f16 (GQA) | 1 | 11.10 / 12.03 (!) → 4.62 / 6.60 (!) | 11.38 / 11.93 (!) → 3.50 / 8.68 (!) | 4.19 / 7.21 | 3.46 / 6.87 | 1.10x |
| segformer-b0 | 1 | 3.43 / 6.21 → 3.02 / 3.49 | 3.63 / 3.75 (!) → 3.33 / 3.83 | 2.33 / 2.55 | 3.39 / 3.92 | 1.30x |
| segformer-b0 | 8 | 23.12 / 25.12 → 22.63 / 24.83 | 21.24 / 23.45 (!) → 20.95 / 22.55 | 19.90 / 22.54 | 23.38 / 25.44 | 1.14x |
| mobilesam-encoder | 1 | 6.38 / 7.23 → 6.23 / 6.95 | fail → fail | 6.15 / 9.04 | 8.00 / 8.93 | 1.01x |
| raft-large | 1 | 58.69 / 63.30 → 58.19 / 64.14 | 47.97 / 75.43 (!) → 45.01 / 54.94 (!) | 29.25 / 36.54 | 77.00 / 94.06 | 1.99x |

What moved:

* **bert**: b1 went from 1.63 to 1.26 ms (ORT TF32 1.24, strict 1.44), b8 from 4.57 to 3.72 ms. This comes from the
  LayerNorm translation and the bmm change.
* **vit**: b1 went from 3.34 to 2.76 ms, segformer b1 from 3.43 to 3.02 ms.
* **phi-3.5 q4f16**: 11.1 → 4.6 ms. The weights now stay Q4_0 instead of f32.
* **qwen q4**: 3.56 → 3.19 ms, against 3.86 ms for ORT.
* **ssd-10** now runs, at 19.5 ms with NMS on the CPU fallback (ORT 11.0 ms).
* **Convnets** are unchanged within noise. Nothing on their Vulkan path changed.

The first after-run showed qwen q4 failing parity (rms 1.0e-3 instead of 1.1e-5). Its weights became Q4_0, so prompts
of more than 8 rows took the f16-activation tensor-core GEMM. MatMuls marked `f32_activations` now use those GEMMs only
with `AllowReducedPrecision`, as f32 weights already did. Speed is unchanged, and the pre-fix rows are in
`superseded/`.

Against `ort-cuda-strict`, Sharplight's default is faster or equal on 14 of 21 rows. Against TF32 it is ahead on
resnet50 b1 and qwen, about equal on bert b1 and mobilesam, and up to 3.2x behind on the transformer rows (vit b8,
whisper), where ORT's f32 matmuls use TF32 tensor cores.

`ort-cuda` runs f32 convolutions and matmuls in **TF32** (10-bit mantissa), which is the EP's default. The
accuracy-equivalent comparison for Sharplight's default mode is `ort-cuda-strict`. Sharplight's strict convs use the 3-way
bf16 split on tensor cores, and its f32 matmuls use the SIMT GEMM.

### 21.4 Opt-in reduced precision

`sl-vulkan-rp` turns on `AllowReducedPrecision` (f16 single-part convs and matmuls, int8 activation dots) and
`AllowHalfBlockScales`. Its outputs do not pass the strict parity check, as expected. Two problems:

* At batch 1 it is often *slower* than the default (resnet50, convnext, bert, mobilenetv2).
* It loses the device (`VK_ERROR_DEVICE_LOST`) on whisper-tiny-encoder and mobilesam-encoder, in the baseline and on this
  branch.

It pays off at batch 8 (convnext 4.48 → 2.76 ms, vit 12.75 → 5.43 ms), on raft (58 → 45 ms) and on the q4 models
(qwen 3.19 → 2.50 ms, phi-3.5 4.62 → 3.50 ms). On the CPU, `AllowHalfBlockScales`
packs bert / qwen `model_q4` as Q4_1 / Q4_0 too.

### 21.5 Negative results

* **Split-bf16 tensor-core GEMM for f32 matmuls** (3-part bf16 split, as used for conv): it was correct, but slower than
  the SIMT GEMM at these sizes. vit b1 768x768 took 26.5 µs against 16.2 µs, and b8 358 µs against 287 µs. It was
  removed.
* **Unbounded tile packing in CPU conv**: packing per tile for every conv, rather than for ≤ 1 MB weights and K ≤ 1152,
  made resnet50 slower (11.5 against 8.0 ms). Large-K convs prefer the up-front B panels, which stay in L2 across row
  tiles.
* **Decomposed LayerNorm on Vulkan**: the elementwise fusion already merged most of it. The translation is worth only
  0.03 ms on vit. The gain there comes from the bmm change.

### 21.6 Correctness and tests

Accuracy stays within the existing parity checks. Every after-row of the default engines passed the manifest's ORT
parity check, except ssd-10 (detection ordering on ties, as before) and phi-3.5 (fp16 reference noise, rms 4.6e-2, as
before).

On the rebased branch:

* Cpu, Onnx (`-parallelMode none`), Package, Families and Vulkan pass: 1169 / 412 / 29 / 456 / 1065 tests, 0 failed.
* The ONNX conformance report is unchanged from `main`:
  * 1829 of 1884 node tests pass (97.1 %) on CPU and on Vulkan;
  * 161 / 161 / 147 of 192 ai.onnx operators pass (cpu / vulkan / vulkan without fallback).

New tests:

* `ConvTests.ConvEpilogues`: conv + residual + Clip, SiLU pattern, Relu with a second consumer, on the tile-packed,
  im2col, depthwise and 1x1 paths.
* Wide depthwise cases (stride 1 and 2, 3x3 and 5x5).
* `MatMulEpilogueTests`: bias, residual and activation on the row and GEMM paths, prepacked or not.
* `TranslationTests`: decomposed LayerNorm and GELU, positive and negative cases.

### 21.7 Commands

```bash
source bench/onnx/env.sh                         # ORT venv + CUDA libs
export SHARPLIGHT_ONNX_MODELS=~/models/onnx      # models and ref/ manifests
dotnet build bench/onnx/OnnxBench -c Release
python3 bench/onnx/parity_bench.py run after.jsonl --rounds 2 [--engines sl-cpu,ort-cpu] [--models vit-base,...]
python3 bench/onnx/parity_bench.py table after.jsonl before.jsonl
dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/<manifest>.json vulkan --profile
```

### 21.8 Remaining gaps

* **CPU convolution GEMM.** resnet50 is 1.4x / 2.7x behind ORT at b1 / b8, efficientnet 2.2x and segformer b1 2.2x.
  * The 8x32 microkernel reaches roughly 10-15 % of the 9950X's f32 peak inside these networks.
  * At batch > 1 each image is a separate GEMM.
  * Next steps: pack the conv weights once, block over output channels per thread, and treat N·H·W as one GEMM.
* **CPU transformer GEMM.** vit is still 1.25x behind ORT. GEMMs are ~60 % of its time.
  * The attention scale `Div` and Softmax are separate passes.
* **Vulkan f32 GEMM vs TF32.** vit, whisper and bert at b8 trail `ort-cuda`, because ORT uses TF32 tensor cores. Against
  `ort-cuda-strict`, Sharplight leads or ties on most rows.
* **NonMaxSuppression / RoiAlign / MaxPool with indices** have no Vulkan kernel. ssd-10 runs NMS as a CPU fallback
  segment.
  * NMS needs data-dependent output counts on the device.
  * CumSum and GridSample already have Vulkan kernels.
* **Reduced precision**: slower than strict at b1, and `DEVICE_LOST` on two encoders (21.4).

## 22. Skinny GEMM retune, dual fp8 gate/up above 64 rows, matmul bias in the matvec (2026-11-14)

Follow-up to section 20.9. All changes are kernel-selection or fusion rules that apply to every model:

* `GraphLowerer.SkinnyGemm.cs`: the tile configuration of the skinny tensor-core GEMM (2-16 rows).
* `GraphLowerer.Fp8Gemm.cs`, `gemm_fp8.slang`: the dual fp8 gate/up GEMM above 64 rows.
* `GraphLowerer.Fusion.cs` (`FoldMatMulBiases`), `GraphLowerer.MatMul.cs` (`LowerMatMulBias`), `matmul.slang` (`BIAS`):
  a weight bias added in the single-row matvec.

Base is `main` at `eac6efe`, built side by side; base and new runs alternate. The GPU was shared during all runs: the
user's other applications (one holding about 3.4 GB) and other agents' test suites were active during parts of them, so
the tables use minimum GPU times or best-of-N values, and each `*.gpu.txt` records what else ran. Raw results:
[`bench/results/s22/`](../bench/results/s22).

### 22.1 Skinny GEMM (2-16 rows)

The skinny GEMM gives each workgroup `fn` 16-column weight fragments and splits K over `sg` subgroups. Section 19 used
two fragments for most shapes and targeted 2048 subgroups in flight, within 32 KB of shared memory. A sweep over the 27B
and Mistral shapes (`SHARPLIGHT_SKINNY_CFG=fn,sg,bk`) with cold weights (22.8) gave the new rule:

* one fragment per workgroup below N = 10240, which doubles the workgroups and the loads in flight;
* 8 subgroups per workgroup while the grid has fewer than 1536 subgroups and K has at least 4 blocks per subgroup;
* the device's shared-memory limit (`maxComputeSharedMemorySize`) up to 48 KB, instead of a fixed 32 KB. Smaller devices
  shrink the K block, then the subgroup count, as before.

Kernel time in µs (minimum of 400 iterations, 4 weight copies so that every launch reads from DRAM) and weight bandwidth:

| weights, rows | shape [K→N] | base µs | new µs | TB/s base → new |
|---|---|---|---|---|
| FP8, 4 | [17408→5120] | 92 | 70 | 0.97 → 1.27 |
| FP8, 4 | [6144→5120] | 31 | 23 | 1.01 → 1.37 |
| FP8, 4 | [14336→4096] | 72 | 49 | 0.82 → 1.20 |
| FP8, 4 | [4096→14336] | 59 | 47 | 1.00 → 1.25 |
| FP8, 4 | [5120→12288] | 68 | 54 | 0.93 → 1.17 |
| FP8, 16 | [17408→5120] | 95 | 82 | 0.94 → 1.09 |
| FP8, 16 | [5120→6144] | 39 | 31 | 0.81 → 1.01 |
| FP8, 16 | [14336→4096] | 74 | 62 | 0.79 → 0.95 |
| f16, 4 | [5120→6144] | 68 | 45 | 0.93 → 1.40 |
| f16, 4 | [4096→14336] | 102 | 76 | 1.15 → 1.55 |

The dual gate/up configuration (one fragment, 4 subgroups, K block 64) was already the best of the sweep and is unchanged.
Tried and not faster:

* loading the cooperative-matrix operands directly from global memory, without staging in shared memory;
* a K block of 128 or more (register spills);
* splitting K across workgroups (at most ~15 % on the kernel, before the extra reduction launch).

27B decode steps (BatchBench `--ctx 2048 --sizes 1,2,4,8,16,32,64`), minimum GPU ms per step over 2 interleaved process
runs per build (`batch/rows_*`):

| rows | FP8 base | FP8 new | | Q8_0 base | Q8_0 new | |
|---|---|---|---|---|---|---|
| 1 | 22.8 | 22.4 | 0.97x | 22.3 | 22.5 | 1.04x |
| 2 | 27.8 | 25.1 | **1.11x** | 19.7 | 20.4 | 0.97x |
| 4 | 28.6 | 25.8 | **1.11x** | 22.6 | 22.2 | 1.02x |
| 8 | 30.4 | 28.8 | 1.06x | 27.3 | 28.5 | 0.96x |
| 16 | 33.3 | 33.1 | 1.01x | 34.3 | 33.1 | 1.04x |
| 32 | 33.4 | 33.7 | 0.99x | 42.4 | 43.1 | 0.98x |
| 64 | 43.5 | 43.9 | 0.99x | 49.5 | 49.1 | 1.01x |

* **FP8 at 2-8 rows** (MTP verification, 2-8 streams) is 6-11 % faster: those steps run every projection on the skinny
  GEMM.
* **FP8 at 16 rows** barely moves. Its down projection went from 95.6 to 83.4 µs, but the step is dominated by the dual
  gate/up GEMM (139-143 µs per layer), whose configuration was already the best.
* **Q8_0** runs its 2-16-row steps on the int-dot matvec and the int8 GEMM, so its column is a control: ±4 % noise.
* 1 row runs the f32 matvec in both builds; 32 / 64 rows run the fp8 GEMM.

### 22.2 Dual fp8 gate/up above 64 rows

Section 20.2 kept the dual fp8 GEMM at 17-64 rows because above that its fused epilogue rounded differently from the two
separate GEMMs plus SiLU·mul. The difference had two sources, and both are removed:

* **Accumulation.** The driver contracted the separate GEMM's `acc += c * (x * w)` to an FMA, but not the dual GEMM's.
  Both now write `fma(c, x * w, acc)` explicitly. The separate GEMM's output is bit-identical to before (same hash of a
  probe run), and the dual one now matches it.
* **Split-K.** Where the separate GEMMs split K (small grids), their partial sums round differently from the dual GEMM's
  single pass. Above 64 rows the dual GEMM now runs only where `QuantizedGemmSplit` would not split the separate GEMMs.

`Fp8GemmTests.DualMatchesSeparateBitwise` checks bit equality at 512 and 636 rows. `SHARPLIGHT_FP8_DUAL_MAX_ROWS=64`
restores section 20's rule. Greedy output is unchanged (22.7).

27B FP8 prefill (`sharplight --bench N --batch N`), best of 3 interleaved runs, tok/s (`prefill/`):

| | base | new | | vLLM (§16.3) | new / vLLM |
|---|---|---|---|---|---|
| pp512 | 4310 | 4317 | 1.00x | 5171 | 0.83x |
| pp2048 | 4149 | 4268 | **1.03x** | 5944 | 0.72x |

The dual GEMM runs at 651 µs (~280 TFLOPS) per layer at 512 rows, the same rate as the separate GEMMs. It saves one
launch, one activation read and the elementwise pass. The 400+ TFLOPS target needs a different fetch scheme (20.2).

### 22.3 Matmul bias in the single-row matvec

Models with projection biases (gpt-oss q/k/v/o and router, Qwen2 q/k/v) ran each biased projection as a matmul plus an
elementwise add. `FoldMatMulBiases` now folds an f32 `[N]` weight bias into a weight matmul whose output it alone reads.
Single-row steps add the bias in the store of the split-K matvec (`BIAS`). Other row counts, and weight types that decode
on the int-dot matvec, keep two launches. `SHARPLIGHT_NO_BIAS_FOLD=1` disables the fold.

On gpt-oss-20b MXFP4 decode, dispatches per step go from 629 to 533 and barriers from 436 to 364.

### 22.4 Single stream against llama.cpp

`sharplight --bench 512 --batch 512` (512-token prefill, then decode), best of 2 interleaved runs per build; then
`llama-bench -ngl 99 -fa 1 -p 512 -n 64 -d 0,512 -r 3` (b11177, CUDA) on the same GPU. Values are tok/s (`single/`,
`llama/`). llama.cpp prefill is at depth 0 and its decode at depth 512, which matches the Sharplight run.

| model | prefill base | prefill new | llama.cpp | decode base | decode new | llama.cpp | decode new / llama.cpp |
|---|---|---|---|---|---|---|---|
| gpt-oss-20b MXFP4 | 13596 | 12752 | 14144 | 307 | **322** | 314 | 1.03x |
| OLMoE-1B-7B Q4_K_M | 28485 | 28443 | 25406 | 772 | 775 | 746 | 1.04x |
| ERNIE-4.5-21B-A3B Q4_K_M | 11195 | 11425 | 9252 | 332 | 333 | 309 | 1.08x |
| Nemotron-3-Nano-30B-A3B Q4_K_M | 9356 | 9348 | 8372 | 313 | 313 | 306 | 1.02x |
| Ministral-3-3B Q4_K_M | 19034 | 19733 | 15609 | 382 | 377 | 359 | 1.05x |
| Mistral-7B Q4_K_M | 10371 | 9759 | 11334 | 242 | 242 | 234 | 1.03x |

* **gpt-oss decode is 5 % faster** from the bias fold (best GPU ms per step 3.17 → 3.03). The other models have no
  projection biases, or decode on the int-dot matvec; their differences are noise. Prefill is unchanged for all of them
  (they do not use the fp8 GEMM), so the prefill differences are noise as well.
* **The llama.cpp column is low.** Another agent's test suite used the GPU during these runs, and the README's quiet
  runs had gpt-oss decode at 345 and ERNIE at 365 tok/s. An earlier quieter llama.cpp run during this work gave 340-347
  decode and 17.3k pp512 on gpt-oss. The README comparison therefore stays the reference, with gpt-oss decode updated to
  322 (0.93x of 345).

### 22.5 MTP

SpecBench as in section 20.6 (natural-language prompt, greedy, median of 5, tok/s), base and new back to back (`spec/`):

| k | FP8 base @d47 | FP8 new @d47 | FP8 base @d4096 | FP8 new @d4096 | Q8_0 base @d47 | Q8_0 new @d47 | Q8_0 base @d4096 | Q8_0 new @d4096 |
|---|---|---|---|---|---|---|---|---|
| off | 37.8 | 40.2 | 36.2 | 39.4 | 39.6 | 39.7 | 38.8 | 39.0 |
| 1 | 56.6 | 59.3 | 54.2 | 55.0 | 76.6 | 76.0 | 71.9 | 73.2 |
| 2 | 69.6 | 72.0 | 56.1 | 64.7 | 85.6 | 86.3 | 80.1 | 75.9 |
| 3 | 70.3 | 70.0 | 67.6 | 66.0 | 91.9 | 93.0 | 91.3 | 96.0 |

* The accepted-draft counts are identical between the builds for every k and depth (e.g. FP8 k = 3: 399/709 and
  402/707), as greedy identity requires.
* FP8 verification steps (2-4 rows) run the retuned skinny GEMM. k = 1 / 2 gain 1-5 %, the non-speculative baseline
  ~6 %. k = 3 is flat within the spread of this busy period: per-run values ranged 63-91 tok/s.
* Q8_0 verification runs the int-dot matvec, which is unchanged, so its column differs only by noise.

### 22.6 Serving against vLLM

`bench/s16_serve.sh` as in section 20.5 (~250-token prompts, 128 output tokens, ignore_eos, a 64-slot server, 2 rounds,
a fresh server per concurrency, base and new alternating, then vLLM 0.33 with `--gpu-memory-utilization 0.5`). Values
are the mean of 2 rounds: aggregate tok/s, TTFT P50 / P95 in ms and mean ITL in ms.

| conc | vLLM FP8 | SL base | SL new | new / base | SL new / vLLM | TTFT SL base → new | vLLM TTFT | ITL SL base → new | vLLM ITL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 45 | 40 | 39 | 0.99x | 0.87x | 80 / 80 → 80 / 87 | 60 / 68 | 24.8 → 25.0 | 22.7 |
| 4 | 162 | 124 | 130 | **1.05x** | 0.80x | 220 / 252 → 217 / 251 | 186 / 197 | 31.0 → 29.3 | 23.5 |
| 16 | 514 | 346 | 335 | 0.97x | 0.65x | 302 / 1071 → 348 / 997 | 642 / 778 | 43.1 → 44.6 | 26.2 |
| 64 | 1155 | 741 | 741 | 1.00x | 0.64x | 529 / 4812 → 506 / 4891 | 2070 / 2291 | 74.9 → 74.8 | 40.7 |

* **c = 4 is 5 % faster**, with ITL down by 1.7 ms: its decode steps have 4 rows and run the retuned skinny GEMM (22.1).
* **c = 1** decodes on the f32 matvec, and **c = 16 / 64** on steps whose time the dual gate/up GEMM and the fp8 GEMM
  decide; neither changed, and the ±3 % differences are within the spread of single rounds (c = 16: 327-352 tok/s for
  either build). The GPU ran at its 300 W limit and 87 °C during the c = 16 / 64 runs.
* vLLM measured 2-5 % higher than in section 20.5, so the ratios are slightly lower for the same code.

### 22.7 Correctness

Greedy CLI output, 200 tokens, base vs new (`ident/` holds the generated text):

| model | 636-token prompt | short prompt | chat prompt | MTP k = 3 vs non-speculative (new) |
|---|---|---|---|---|
| Qwen3.8-27B FP8 | identical | identical | identical | identical (636-token and short) |
| Qwen3.8-27B Q8_0 | identical | identical | identical | identical (636-token and short) |
| gpt-oss-20b MXFP4 | | | identical | |
| Qwen2.5-0.5B Q8_0 | | | identical | |

SpecBench accepted-draft counts are equal between the builds for every k and depth (22.5).

Tests: `FusionTests.MatMulBias` (the bias fold at one and several rows), `Fp8GemmTests.DualMatchesSeparateBitwise`, and
the `MatMulTests`, `Fp8GemmTests` and `FusionTests` classes. On the branch rebased onto `main` (after section 21), the Cpu,
Vulkan, Families, Serving and Studio suites pass, run one at a time. One Serving run failed `EngineFeatureTests.FairQueuing_LightTenantOvertakesHeavyTenantsBacklog`
(an admission-order assertion) while other agents loaded the machine; it passed on 3 isolated reruns and on a full rerun.
The 636-token FP8 output of the rebased build is identical to the outputs above.

### 22.8 Benchmark options

`BenchTests.MatMulPrefill` (Vulkan tests, `SHARPLIGHT_BENCH=1`) gained three options:

* `SHARPLIGHT_BENCH_COPIES=n`: n copies of each weight, used in turn, with the time divided by n. Below about 64 MB a
  single weight stays in the 128 MB L2 between launches, which overstated the skinny GEMM's bandwidth by up to 2x.
* `SHARPLIGHT_BENCH_ITERS=n` (default 50). 400 gives stable minima, because the GPU clock ramps up over the first
  hundred launches.
* `SHARPLIGHT_BENCH_DUAL=1`: also time the dual gate/up launch for each shape.

### 22.9 Commands

```bash
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_TYPES=f8e4m3_b128 SHARPLIGHT_BENCH_ROWS=4 SHARPLIGHT_BENCH_COPIES=4 \
  SHARPLIGHT_BENCH_ITERS=400 SHARPLIGHT_BENCH_SHAPES="17408,5120;14336,4096" \
  dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll -class Sharplight.Vulkan.Tests.BenchTests
#   add SHARPLIGHT_SKINNY_CFG=fn,sg,bk to force a skinny GEMM configuration
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,2,4,8,16,32,64
samples/Sharplight.Cli/bin/Release/net11.0/sharplight <model> --bench 512 --ctx 768 --batch 512
LD_LIBRARY_PATH=$LLAMA_DIR $LLAMA_DIR/llama-bench -m <gguf> -ngl 99 -fa 1 -p 512 -n 64 -d 0,512 -r 3 -o jsonl
bench/SpecBench/bin/Release/net11.0/SpecBench <model> --drafts 0,1,2,3 --depths 47,4096
OUT_DIR=bench/results/s22 PORT=18321 SLOTS=64 bench/s16_serve.sh sharplight qwen27b-fp8-new-c16 <dir> "16"  # one concurrency per server
OUT_DIR=bench/results/s22 PORT=18321 SLOTS=64 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
```

### 22.10 Open issues

* **16-row decode step.** The skinny GEMM reaches 0.95-1.1 TB/s at 16 rows (1.2-1.55 TB/s at 4 rows), and the 16-row
  FP8 step is dominated by the dual gate/up GEMM. Reaching ~1.5 TB/s needs fewer re-reads of x per workgroup (x kept in
  registers across fragments) or a cross-workgroup K split with the reduction fused into the consumer.
* **FP8 prefill GEMM** is still at ~280 TFLOPS. The dual GEMM above 64 rows removes launches, not fetch cost; 400+ needs
  an asynchronous-copy or larger-MMA fetch scheme (20.9).
* **Not attempted in this round:** K-quant MMQ, gpt-oss MoE prefill (0.68-0.74x of llama.cpp), MoE gate/up fusion for
  decode, and the opt-in reduced-precision delta-rule state. The state option has to reach the `GatedDeltaRule` op, its
  CPU reference, the chunked and token-serial kernels, the rollback records and the state snapshots. The delta rule takes
  ~232 µs per layer at 64 decode rows, a quarter of that step.
* **Single-user TTFT** (~82 ms against vLLM's 60 ms for ~250-token prompts) is GPU time: the 250-row prefill step profiles
  at 69 ms. Command-buffer recording (20-50 ms for a new row count) happens only for the first shapes of a run, so TTFT
  follows the per-row prefill cost.

## 23. Direct-fragment skinny GEMM, aligned FP8 weight layout (2026-11-14)

Follow-up to section 22.10 (the 16-row decode step). All changes are kernel or layout rules that apply to every model:

* `matmul_skinny.slang` (`SK_DIRECT`), `GraphLowerer.SkinnyGemm.cs` (`UseDirectSkinny`): a variant of the skinny GEMM
  (2-16 rows) that loads its operands straight into the MMA fragment registers.
* `DeviceFeatures.MatrixMmaLayout` (`matrix_mma_layout`): the device feature that gates the variant.
* `WeightRepacker.cs`, `ShaderTypes.cs` (`f8e4m3_b128r`, `f8e4m3_b32r`), `quant_fp8.slang`, `gemm_fp8.slang`: a
  16-byte-aligned split layout for block-scaled FP8 weights.

Base is `main` at `5f3c3cf`, built side by side. Base and new runs alternate. The GPU was shared with the user's other
applications and other agents, so the tables use minimum GPU times (BatchBench, kernels) or the best median over
repeated runs (SpecBench). Raw results are in [`bench/results/s23/`](../bench/results/s23).

### 23.1 The direct skinny GEMM

The section-22 skinny GEMM stages x and the weights in a private shared-memory slice per subgroup, then loads the
cooperative-matrix fragments from there. At 16 rows its FP8 gate/up launch reached about 1.1 TB/s.

`SK_DIRECT` (one token fragment, f16 x, K % 64 == 0, block FP8 or Q8_0 weights) skips shared memory in the K loop:

* **Fragment layout.** On NVIDIA, a 16x16x16 f16 fragment is two 16x8x16 MMA operands. Lane (g = lane / 4,
  c = lane % 4) holds rows (A) or columns (B) g and g + 8, at K slots 2c, 2c + 1, 2c + 8 and 2c + 9. The kernel builds the
  fragments element by element from registers.
* **K permutation.** A dot product may visit K in any order, as long as A and B agree. So each lane's slots for the four
  MMAs of a 64-value block map to 16 contiguous values of its row. A lane then reads x and the weights with 16-byte
  loads, and the four lanes of a row cover 64 contiguous values.
* **FP8 loads.** FP8 codes are prefetched one block ahead as raw words and decoded to f16 just before their MMAs.
* **Reduction.** The subgroups still split K inside the workgroup (8 subgroups) and reduce through shared memory once at
  the end.
* **Layout check.** The layout comes from the hardware, not from the Vulkan specification, so the lowering only picks
  the variant when `matrix_mma_layout` is detected (NVIDIA, `matrix_f16`, 32-wide subgroups). The kernel verifies the
  layout once with a probe matrix and writes NaN if it differs, so a wrong detection fails tests instead of producing
  plausible output.
* **Disabling.** `SHARPLIGHT_SKINNY_DIRECT=0`, or `--disable-features matrix_mma_layout`, keeps the staged path.
* **Separate variants.** Compiling both paths into one kernel costs the direct path 60 registers (188 against 128) and
  most of its gain, so they are separate compile-time variants.

Kernel times (`BenchTests.MatMulPrefill`, minimum over 300 launches, 4 weight copies so that nothing stays in L2;
`micro/`), Qwen3.8-27B shapes:

| rows | [17408→5120] FP8, staged → direct | [5120→17408]x2 gate/up FP8, staged → direct |
|---|---|---|
| 2 | 69.6 → 63.5 µs | 139.3 → 124.9 µs |
| 4 | 71.7 → 65.5 µs | 141.3 → 126.9 µs |
| 8 | 75.8 → 67.6 µs | 145.4 → 126.9 µs |
| 16 | 84.0 → 71.6 µs (1.24 TB/s) | 155.7 → 129.0 µs (1.38 TB/s) |

Q8_0 [17408→5120] at 16 rows: 79.6 → 67.3 µs. f16 matmuls showed no gain (112.6 / 215 µs single / dual at 16 rows,
1.58-1.66 TB/s), because their staged loads already run near bandwidth.

The direct path therefore serves only block FP8 and Q8_0 weights. f16 / bf16 gain nothing, and the K-quants' decode is
not tuned for it. These formats keep the staged path and its rounding. An earlier build also took the K-quants direct,
and that flipped a near-tied token in the pythia-160m Q4_K_M reference test (`FamilyGenerationTests`, llama.cpp's
first 12 tokens).

Things that did not help (measured on intermediate builds):

* hardware E4M3 → f16 conversion;
* half-width (16x8x16) MMAs;
* a 128-value K block;
* giving each subgroup one contiguous K range instead of interleaved 64-value blocks, which was far slower: DRAM pages
  are better served by 8 subgroups that read neighbouring blocks.

A constant (fake) scale per block saved about 5 %, so the remaining cost is mostly on the memory side.

### 23.2 Aligned split FP8 layout

Block-scaled FP8 rows were stored as {f32 scale, 128 codes} blocks of 132 bytes, so codes were only 4-byte aligned.
Rows with K % 512 == 0 (b128) or K % 128 == 0 (b32) are now repacked at upload into two parts:

* first all codes of the row;
* then all its scales.

Codes become 16-byte aligned. The fp8 GEMM, the matvec and the skinny GEMM read the new layout. The prefill GEMM's
arithmetic is unchanged (same products and sums), so outputs are bit-identical. `SHARPLIGHT_REPACK_FP8=0` keeps the
file layout.

fp8 GEMM at T = 512 (`micro/`), without → with the repack:

| shape | time | TFLOPS |
|---|---|---|
| [5120→17408] | 315.4 → 307.2 µs | 289 → 297 |
| [17408→5120] (split-K) | 343.9 → 333.7 µs | 266 → 274 |
| [5120→6144] | 178.2 → 165.8 µs | 181 → 194 |
| [5120→10240] | 198.7 → 192.4 µs | 270 → 279 |

* **End-to-end prefill.** 27B FP8 pp512 / pp2048 does not change measurably: 4040-4260 vs 4040-4180 tok/s at 512 and
  4170-4193 vs 3948-4188 tok/s at 2048, three interleaved runs each (`prefill/repack_ab_*`). The roughly 3 % GEMM gain is
  inside the run-to-run noise of a shared GPU.
* **Staged skinny GEMM.** The repack makes the staged path 3-6 % slower (its word loads straddle the split differently).
  It applies on devices without `matrix_mma_layout`. The direct path is equal or slightly faster with the repack.

### 23.3 Decode steps (BatchBench)

BatchBench, `--ctx 2048 --modes greedy`, minimum GPU ms per step over 2 interleaved process runs per build (`batch/`):

| rows | FP8 base | FP8 new | | Q8_0 base | Q8_0 new | |
|---|---|---|---|---|---|---|
| 1 | 23.17 | 22.42 | (same path) | 22.15 | 23.08 | (same path) |
| 2 | 25.58 | 24.22 | 1.06x | 19.73 | 19.76 | int-dot |
| 3 | 26.14 | 24.44 | 1.07x | 22.34 | 22.42 | int-dot |
| 4 | 26.50 | 25.15 | 1.05x | 21.95 | 22.63 | int-dot |
| 8 | 28.75 | 26.42 | 1.09x | 27.75 | 28.47 | int-dot |
| 12 | 30.30 | 28.29 | 1.07x | 31.30 | 26.57 | **1.18x** |
| 16 | 34.30 | 29.61 | **1.16x** | 33.70 | 29.34 | **1.15x** |
| 32 | 33.77 | 33.04 | 1.02x | 42.88 | 43.10 | 1.00x |
| 64 | 43.94 | 43.02 | 1.02x | 49.46 | 49.71 | 1.00x |

* **Unchanged rows.** Single rows, and Q8_0 up to 8 rows (int-dot matvec), run the same kernels in both builds, so their
  differences are noise; base runs alone differ by up to 1.6 ms.
* **Against vLLM.** The 16-row FP8 step goes from 34.3 to 29.6 ms. vLLM's mean ITL at 16 streams is 26.2 ms (22.6), which
  also includes its prefill steps, so the step now runs at about 0.89x of that, up from 0.76x.
* **Q8_0 at 5-8 rows.** `SHARPLIGHT_MMVQ_MAX_ROWS=4` moves these rows from the int-dot matvec to the direct skinny GEMM.
  The step drops from 27.7 / 28.1 / 28.7 ms to 24.0 / 25.6 / 25.3 ms at 5 / 6 / 8 rows. It also changes the rounding of
  those rows, and with it the greedy MTP k = 3 output of the short Q8_0 prompt
  (`ident/q8_poem_mtp3_with_mmvq_cap4.txt`). So the default stays at 8, and the setting is a diagnostic switch. The
  lowering now lets the skinny GEMM, rather than the f32 matvec, take the rows above a lowered int-dot limit.

### 23.4 Speculative decoding (SpecBench)

SpecBench as in section 22.5 (natural-language prompt, greedy, depth 47), tok/s, best median of 3 runs per build
(`spec/`):

| model | k = 0 | k = 1 | k = 3 |
|---|---|---|---|
| 27B FP8 base → new | 40.5 → 40.4 | 63.3 → 68.5 (1.08x) | 87.6 → 94.4 (1.08x) |
| 27B Q8_0 base → new | 41.3 → 40.8 | 79.6 → 79.2 | 104.3 → 104.6 |

Accepted-draft counts are equal between the builds. Q8_0 verification (2-4 rows) stays on the int-dot matvec.

### 23.5 Correctness

Greedy CLI output, 200 tokens (`ident/`):

* Compared with `main`'s outputs in `bench/results/s22/ident`: 27B FP8 and Q8_0 (636-token, short and chat prompts),
  gpt-oss-20b and Qwen2.5-0.5B are identical. The recorded gpt-oss file in s22 is cut off after its 18th line; `main`
  at `a548393`, rebuilt, prints the same full text as this branch.
* MTP k = 3 against non-speculative: identical for FP8 and Q8_0, 636-token and short prompts.

Tests:

* New `SkinnyGemmTests`: single and dual (SiLU·mul) matmuls over f16, bf16, FP8 b128 / b32, NVFP4 and Q8_0 at 2, 5 and
  16 rows. They run on the NVIDIA GPU with the direct path and with `SHARPLIGHT_SKINNY_DIRECT=0`, on the integrated GPU
  and on llvmpipe (`SHARPLIGHT_TEST_DEVICE=1` / `2`, staged path).
* The NVFP4 dual case gets a wider bound (atol 3). Its random rows give products above 100, and the generic f16 GEMM
  (`SHARPLIGHT_SKINNY_GEMM=0`) deviates from the CPU reference by the same amount.
* `MatMulTests`, `FusionTests`, `Fp8GemmTests`, `QuantFormatTests` and `MoeTests` pass with the repack.
* On the branch rebased onto `main` (`2d65fd0`), run one at a time, these suites pass: Cpu, Vulkan, Families, Serving and
  Vision.
  * Families was OOM-killed twice while the machine was short of memory (once with its process at 42 GB). It passed on
    a run alone, at a peak RSS of 16.8 GB.
  * Studio's `DownloadsViewModelTests.Download_ReportsProgress_PersistsTheQueue_AndAnnouncesTheModel` fails
    intermittently: in 2 of 3 runs on this branch and in 1 of 4 on `main` (`5f3c3cf`). This change does not touch Studio.

### 23.6 Commands

```bash
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_TYPES=f8e4m3_b128 SHARPLIGHT_BENCH_ROWS=16 SHARPLIGHT_BENCH_COPIES=4 \
  SHARPLIGHT_BENCH_ITERS=300 SHARPLIGHT_BENCH_DUAL=1 SHARPLIGHT_BENCH_SHAPES="5120,17408" \
  dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll -method '*MatMulPrefill'
#   SHARPLIGHT_SKINNY_DIRECT=0 / SHARPLIGHT_REPACK_FP8=0 for the staged path / file layout
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,2,3,4,8,12,16,32,64
samples/Sharplight.Cli/bin/Release/net11.0/sharplight <model> --bench 512 --ctx 768 --batch 512
bench/SpecBench/bin/Release/net11.0/SpecBench <model> --drafts 0,1,3 --depths 47
```

### 23.7 Open issues

* **16-row step.** The FP8 dual gate/up GEMM is at 1.38 TB/s and [17408→5120] at 1.24 TB/s, against about 1.65 TB/s
  for f16 weights and the lm_head. The rest of the step is spread out: the delta rule, attention, and glue launches such
  as f16 copies and the small [5120→48] split-K matmuls (about 0.9 ms per step).
* **Staged path with the repack** is 3-6 % slower. That is the path for devices without `matrix_mma_layout` (AMD,
  Intel, llvmpipe), and the repack's prefill gain has not been measured on them.
* **Q8_0 at 5-8 rows** would gain 10-13 % on the direct GEMM, but it changes greedy MTP output (23.3).
* **Serving against vLLM** was not re-run in this round. The decode-step gains apply to its c = 4 / 16 decode steps.
* **Not attempted in this round:** the FP8 prefill GEMM memory pipeline (400+ TFLOPS), MoE grouped GEMM and fused gate/up,
  K-quant MMQ, and the opt-in reduced-precision delta-rule state.

## 24. MoE: raw MXFP4 expert fetch, split expert tiles, dual gate/up and slot-sum expert matvecs; opt-in bf16 delta-rule state (2026-11-14)

Follow-up to sections 14 and 22 (gpt-oss-20b prefill at 0.68x and decode at 0.93x of llama.cpp CUDA). The MoE changes
are kernel or fusion rules for every MoE model (MatMulId):

* `matmul.slang` (`gemm_cm2`, `MOE`), `common.slang`, `WeightRepacker.cs`: MXFP4 experts in the expert-grouped GEMM are
  fetched as raw codes and scales one step ahead and decoded after the MMAs; the aligned MXFP4 layout stores its scales
  block-major per 16-row group.
* `GraphLowerer.Moe.cs`: tiles of at most half the tile height run in a second launch on half-height tiles.
* `GraphLowerer.Fusion.cs` (`FoldDualMatMuls`, `FoldExpertSums`), `GraphLowerer.Moe.cs` (`LowerDualMatMulId`,
  `LowerExpertSum`), `mmvq.slang` (`moe_mmvq`, `DUAL`, `SLOT_SUM`): decode fusions on the integer-dot expert matvec.

Plus an opt-in bf16 state for the gated delta rule (24.6) and the re-runs against vLLM (24.5) and llama.cpp (24.7).

Base is `main` (`95cc0a2` for 24.1-24.3, `4b20f55` for the correctness runs), built side by side; base and new runs
alternate. The GPU was shared with the user's applications and other agents, so the tables use best-of-N or minimum GPU
times. Raw results: [`bench/results/s24/`](../bench/results/s24).

### 24.1 Expert GEMM for prefill

The grouped expert GEMM (`gemm_cm2` with `MOE`, 64-row tiles at gpt-oss pp512) took 340 µs for one gpt-oss projection
(32 experts, [2880→2880], 2048 routed rows, skewed routing; `BenchTests.MoePrefill`). Ablations of that kernel:

| variant | µs (min) |
|---|---:|
| as on `main` | 340 |
| no MXFP4 decode (codes used as halves) | 291 |
| no weight loads | 172 |
| no MMAs | 203 |
| no activation loads | 322 |
| constant scales | 266 |

So the weight fetch and its decode, not the MMAs, set the time. Changes, each measured on its own:

* **Raw fetch, deferred decode** (MXFP4 only): the K step's code words and scale bytes are loaded into registers one step
  ahead and decoded to f16 only when staged, after the MMAs of the current step (-30 µs). The K-quant and Q8_0 variants
  of the same scheme were slower (q4_k 1.37 → 1.42-1.46 ms per block, q8_0 1.28 → 1.65 ms) or spilled at 128-row tiles,
  so they keep the old path.
* **Block-major scales**: the aligned MXFP4 layout stores a 16-row group's scales as `[block][row]` instead of
  `[row][block]`, so the 16 rows of a staged block read 16 consecutive bytes (-8 µs).
* **Hoisted row addresses** for weights and sorted activations (-26 µs), and a prefetch depth of two steps (about -3 %).

Result: 340 → 279 µs, the whole gate/up/down block 1.072 → 0.89 ms. Dense GEMMs are unchanged (f16 and q4_k A/B).

**Split tiles.** With skewed routing, 64-row tiles are only 65 % full (32-row tiles 83 %, 128-row 43 %): each expert's last
tile is partly empty. The 64- and 128-row launches now skip routing tiles of at most half their height, and a second
launch runs those on 32- / 64-row tiles. Every output row keeps its K order, so the result does not change. In the model
(profile at pp512): one projection 260-272 µs → 178 + 68 µs. `SHARPLIGHT_MOE_SPLIT_TILES=0` turns it off.

| gpt-oss-20b, best of 3 interleaved runs | split off | split on |
|---|---:|---:|
| pp512 tok/s | 15893 | 16023 |
| pp2048 ms | 115 | 110 |

Qwen3.6-35B-A3B (256 experts, 32-row tiles at pp512) is unchanged.

### 24.2 Decode: dual gate/up and slot-sum expert matvecs

Up to 128 (token, slot) items, MatMulId runs the integer-dot expert matvec (`moe_mmvq`, q8_1 activations). A gpt-oss MoE
layer took 8 launches per decode step: quantize x, up, gate, the activation group, quantize the activation, down, the
bias and router-weight group, and the slot sum (`ReduceSum` over `Permute(·, 0, 2, 1)`).

* **Dual gate/up** (`FoldDualMatMuls` now also pairs MatMulId products with the same x, ids and weight shape): one launch
  computes both products and applies the elementwise group, including per-expert bias rows read in place
  (`GatherAxis(bias, ids)`). 2 × 13.1 µs + 2.2 µs → 24.5 µs.
* **Slot-sum** (`FoldExpertSums`): the down product, its group (bias, router weight [T, S, 1]) and the slot sum fold into
  one launch. One subgroup computes each slot of a workgroup's rows; the slot values meet in shared memory and are summed
  in slot order starting from zero, as the narrow `ReduceSum` does, so the result is bit-identical. A first version that
  summed the slots in registers was not: the compiler contracted the router-weight multiply into the sum (a fused
  multiply-add). The shared-memory hand-off prevents that. A version with one subgroup computing all slots of a row had a
  quarter of the threads and took 17.4 µs; one subgroup per slot takes 14.3 µs, against 13.3 + 3.6 + 2.0 µs before.
  `SHARPLIGHT_NO_EXPERT_SUM=1` turns it off.

gpt-oss decode: 8 → 4 launches per MoE layer; GPU time over 64 decode steps (profile) 243.9 → 230.9 ms.

### 24.3 Against llama.cpp CUDA

`sharplight --bench N` best of 3 runs per build (decode = best of the 64 steps after the prompt), then `llama-bench -ngl 99
-fa 1 -p 512,2048 -n 128 -r 3` right after (mean of 3, best sample in parentheses):

| model | | Sharplight `main` | Sharplight new | llama.cpp CUDA | new / llama.cpp |
|---|---|---:|---:|---:|---:|
| gpt-oss-20b MXFP4 | pp512 | 14154 | **16276** | 17081 (18489) | 0.95x |
| | pp2048 | 17144 | **19023** | 17309 (17515) | **1.10x** |
| | tg | 327 | **331** | 342 (343) | 0.97x |
| OLMoE-1B-7B Q4_K_M | pp512 | 30596 | 30911 | 27839 (30468) | 1.11x |
| | pp2048 | 43345 | 42995 | 27338 (28922) | 1.57x |
| | tg | 773 | **822** | 812 (834) | 1.01x |
| Qwen3.6-35B-A3B Q4_K_M | pp512 | 10938 | 10907 | 7175 (7652) | 1.52x |
| | pp2048 | 14460 | 14473 | 6623 (6700) | 2.19x |
| | tg | 267 | **277** | 238 (247) | 1.16x |

Earlier in this round, with the GPU about half busy, llama.cpp measured gpt-oss pp512 at 12440 (best 14560), so its pp512
samples vary by up to 25 % on this machine; 18489 is the best sample seen. gpt-oss pp2048 is now ahead of llama.cpp and
pp512 within its run-to-run spread; decode went from 0.93x to 0.97x.

### 24.4 K-quant MMQ and the FP8 GEMM memory pipeline

Neither produced a change in this round:

* **K-quant MMQ** (`SHARPLIGHT_MMQ=q4_k,q5_k,q6_k`) is still slower than the f16 path on the 27B Q4_K_M at pp512: GPU time
  201.0 vs 190.1 ms. [5120→6144] q4_k takes 308 vs 229 µs, [17408→5120] q6_k 902 vs 620 µs; only the largest q4_k shape is
  close (510 vs 499 µs). The per-32-block scaling (section 13) still costs about as much as the MMAs. Folding the 6-bit
  sub-block scales into integer accumulation needs one activation scale per 256 values, a precision change. Any int8
  path also changes greedy output against the f16 default, so it could only be an opt-in.
* **FP8 GEMM on workgroup-scope matrices** (`VK_NV_cooperative_matrix2`, `matrix_workgroup`): a prototype loaded the E4M3
  activations and weights through tensor layouts (32x32x32 shapes, 128 x 128 tiles, 128-value K steps) and scaled each
  block's product with `MapElement`. On [5120→10240] at 512 rows it took 311 µs against 190 µs for `gemm_fp8`, and still
  258 µs with the scaling removed. The device-side loads are slower than the kernel's own register-prefetched staging,
  so the prototype was dropped. Also, `MapElement` plus an add rounds differently from the fused multiply-add of
  `gemm_fp8`, so outputs would not be bit-identical either. Vulkan reports 48 KB of shared memory per workgroup on this
  driver, which limits the stages a kernel-managed pipeline can hold.

### 24.5 Serving against vLLM

`bench/s16_serve.sh` as in section 22.6 (~250-token prompts, 128 output tokens, ignore_eos, a 64-slot server, 2 rounds),
Qwen3.8-27B-FP8, Sharplight from this branch, then vLLM 0.33 with `--gpu-memory-utilization 0.5`. A first attempt ran
while another agent benchmarked the 27B on the same GPU (c = 1 at 27 tok/s) and was discarded. In the repeated run one
round each at c = 4 and c = 16 was disturbed again (91 and 261 tok/s); the table gives both rounds.

| conc | vLLM FP8 (2 rounds) | Sharplight (2 rounds) | SL best / vLLM | TTFT P50 / P95 SL | vLLM | ITL mean SL | vLLM |
|---|---|---|---|---|---|---|---|
| 1 | 43, 42 | 40, 38 | 0.93x | 80 / 87 | 61 / 80 | 25.3 | 23.2 |
| 4 | 142, 153 | 130, 91 | 0.85x | 232 / 365 | 192 / 232 | 35.7 | 25.9 |
| 16 | 504, 522 | 261, 381 | 0.73x | 379 / 1531 | 621 / 670 | 47.7 | 26.6 |
| 64 | 1157, 1153 | 725, 752 | 0.65x | 517 / 4669 | 1708 / 2301 | 75.7 | 40.3 |

The undisturbed rounds match section 22.6 (40 / 130 / 381 / 741-752): this round does not change the dense 27B
path. The gap to vLLM at 16-64 streams remains the batched decode step (sections 19-23).

### 24.6 Opt-in bf16 delta-rule state

With `--reduced-precision` (`LlmLoadOptions.AllowReducedPrecision`), the gated delta rule's recurrent state is stored as
bf16 instead of f32 (`AdapterOptions.RecurrentStateType` / `LlmLoadOptions.RecurrentStateType`, Qwen3.5-style models;
`SHARPLIGHT_RECURRENT_STATE=f32|bf16|f16` overrides it). Compute stays f32 in registers (`state16.slang`):

* The token-by-token kernel rounds the state after every token row, including replayed rollback rows. Output therefore
  does not depend on how rows are batched, and MTP verification matches plain decode under the opt-in.
* The chunked prefill kernel widens the state once per run and rounds it at the end.
* The CPU reference rounds per row.

Rollback records hold per-token inputs, not the state, so they are unchanged. State buffers and resets follow the
declared type. Conv, Mamba and RWKV states stay f32.

27B FP8, BatchBench `--ctx 2048 --modes greedy --sizes 16,64 --reps 40`, three interleaved rounds, the state type set
by the env switch (isolating it from the other reduced-precision paths), minimum step GPU time:

| rows | f32 | bf16 | f16 |
|---|---:|---:|---:|
| 16 | 37.69 ms | 36.66 ms (-2.7 %) | 37.90 ms |
| 64 | 61.70 ms | **56.40 ms (-8.6 %)** | 60.86 ms (-1.4 %) |

`delta_rule` itself drops from 225.8 to 108.0 µs per layer at 64 rows and from 36.2 to 25.9 µs at 16 rows. f16 is more
accurate, but its conversion cost removes most of the gain, so the opt-in uses bf16.

Teacher-forced against the f32 state (193 positions, `RecurrentStateAccuracyTests`, `SHARPLIGHT_TEST_LARGE=1`):

| state | top-1 agreement | mean KL | max logit difference / logit spread |
|---|---|---:|---:|
| bf16 | 193/193 | 1.4e-5 | 7.1e-3 |
| f16 | 193/193 | 2.4e-7 | 6.4e-4 |

Greedy output with the opt-in:

* Identical to f32 on Q8_0 (both prompts) and on FP8 with the short prompt.
* On FP8 with the 788-token prompt it diverges after about 100 tokens and stays coherent.
* MTP k = 3 matches non-speculative output under the opt-in (FP8, bf16 and f16).

The default state is f32, and its output is unchanged.

### 24.7 Single stream against llama.cpp (README table)

`sharplight --bench 512` and `--bench 16`, 3 runs each, then `llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` right after,
on this branch (rebased on `main` `4b20f55`). The table is in the README. Prefill 512 / decode ratios:

| model | pp512 | decode |
|---|---:|---:|
| Llama-3.2-1B Q8_0 | 1.09x | 1.03x |
| Gemma-3-1B Q8_0 | 1.28x | 1.09x |
| Ministral-3-3B Q4_K_M | 1.03x | 1.02x |
| Mistral-7B Q4_K_M | 0.91x | 1.13x |
| Qwen3.8-27B Q8_0 | 1.11x | 0.90x |
| OLMoE-1B-7B Q4_K_M | 1.04x | 1.01x |
| ERNIE-4.5-21B-A3B Q4_K_M | 1.21x | 1.00x |
| Nemotron-3-Nano-30B-A3B Q4_K_M | 1.23x | 1.05x |
| gpt-oss-20b MXFP4 (24.3) | 0.95x | 0.97x |

These llama.cpp means are lower than the 2026-11-13 README values for some rows (e.g. Llama-1B pp512 54082 vs 58918,
27B Q8_0 2921 vs 3058). llama.cpp's samples varied by up to 10 % within a run, so the ratios carry that uncertainty.
In this run gpt-oss measured 14848 (Sharplight) against 11640 (llama.cpp, samples 10711-12746) at pp512, with disturbed
llama.cpp decode samples (133-333 tok/s). The README uses the interleaved 24.3 numbers instead.

### 24.8 Correctness

Greedy CLI output, 200 tokens, long (~650-token) and short prompts, the merged branch against `main` at `4b20f55`
(`ident/`):

* Identical for gpt-oss-20b, OLMoE, Qwen3.6-35B-A3B, Qwen2.5-0.5B, Mistral-7B Q4_K_M, and the 27B Q4_K_M, Q8_0 and FP8.
* MTP k = 3 against non-speculative output on the 27B Q8_0 and FP8: identical text.
* A 1921-token prompt on gpt-oss (128-row expert tiles) gives the same output with and without split tiles.

Tests:

* New `MoeTests.RoutedExpertBlock`: gate/up with per-expert biases, activation, down with bias, router weights and slot
  sum; mxfp4 / q4_k / q8_0; 1-70 tokens; S = 2, 4, 8. It covers the dual and slot-sum launches and the prefill path.
  `MoeTests` passes on llvmpipe.
* New tests for the bf16 / f16 delta-rule state:
  * CPU and Vulkan against the reference, including channel decay, the additive rule and time-invariant gates;
  * split-invariance and chunked prefill;
  * rollback (Families);
  * `RecurrentStateAccuracyTests` (large).
  The Vulkan ones pass on the NVIDIA GPU and llvmpipe.
* On the merged branch, run one at a time under the suite lock: Cpu 1182 (29 skipped), Vulkan 1202 (22 skipped), Families
  628 (32 skipped; OOM-killed once while the machine had 8 GB free, passed alone), Serving 283 and Vision 94 (5 skipped)
  pass.

### 24.9 Commands

```bash
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_MOE="32,4,2880,2880" dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll \
  -method '*MoePrefill'                                          # expert GEMM block, mxfp4 / q4_k / q8_0
samples/Sharplight.Cli/bin/Release/net11.0/sharplight <model> --bench 512 --ctx 768 --batch 512   # + SHARPLIGHT_VK_PROFILE=1
#   SHARPLIGHT_MOE_SPLIT_TILES=0, SHARPLIGHT_NO_EXPERT_SUM=1 turn the new paths off
LD_LIBRARY_PATH=$LLAMA_DIR $LLAMA_DIR/llama-bench -m <gguf> -ngl 99 -fa 1 -p 512,2048 -n 128 -r 3 -o jsonl
OUT_DIR=bench/results/s24/serve PORT=18337 SLOTS=64 bench/s16_serve.sh sharplight qwen27b-fp8 <dir> "1 4 16 64"
OUT_DIR=bench/results/s24/serve PORT=18337 SLOTS=64 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
SHARPLIGHT_RECURRENT_STATE=bf16 bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 16,64 --reps 40
```

### 24.10 Open issues

* **gpt-oss pp512** is at 0.95x of llama.cpp's mean (0.88x of its best sample). The expert GEMM is still bound by the
  weight fetch. Next steps: K-quant experts in the raw-fetch scheme without the register spill seen at 128-row tiles,
  and one launch for both tile heights (the split costs a barrier).
* **Batched MoE decode** (2-16 tokens through the int-dot matvec) gets the dual and slot-sum launches. Above 128 items
  the expert GEMM keeps separate gate, up, activation, down and sum launches.
* **K-quant MMQ** and the **FP8 GEMM memory pipeline**: no gain this round (24.4).
* **Serving against vLLM** is unchanged at 0.65-0.93x (24.5).
* **bf16 delta-rule state**: the chunked prefill rounds once per run, while the token-by-token kernel rounds per row, so
  the two differ slightly under the opt-in (relative RMS up to 1.3e-2 in the tests).
* **MTP on Q8_0**: the state agent's prompts (a 788-token essay prompt) showed MTP k = 3 output differing from
  non-speculative output on the 27B Q8_0, already on `main`. This section's prompts and section 23's are identical. This
  needs a look at the Q8_0 verification path (int-dot matvec at 2-4 rows against the 1-row path).

## 25. ONNX parity round 2: reduced-precision device loss, conv epilogues on Vulkan, wide SIMT GEMM, CPU conv as one implicit GEMM, detection ops on the GPU (2026-11-14)

This is the second round of section 21: the same 15 models, machine, harness (`bench/onnx/parity_bench.py`) and engines.
Section numbers 23 and 24 belong to the LLM rounds that ran in parallel. Raw data is in
[`bench/results/s25/`](../bench/results/s25). The onnxruntime rows are the section 21 measurements
(`ort-*-from-s21.jsonl`, same machine and ORT 1.31.1, taken a few hours earlier). They were not re-run in this round.

How the numbers were taken:

* Before → after cells are section 21's `after` rows → this branch.
* GPU cells are the min / median of a process's runs. The minimum over two rounds is taken where a second round ran
  (`after-gpu-r2.jsonl`).
* The GPU was shared with the desktop and with another agent's benchmarks, so wall-clock medians are noisy. The table
  in 25.2 also gives the GPU-timestamp minimum, which is not affected by host load.
* **The CPU rows were measured with a load average of 20-25 from other agents' test suites.** They are upper bounds and
  not a clean comparison with section 21, which ran on an idle machine. The CPU conv gains are therefore given from the
  per-shape benchmark (25.3) as well.

### 25.1 What changed

* **Reduced precision lost the device** on whisper-tiny-encoder and mobilesam-encoder (Xid 109, a hang).
  * Cause: the split-K f16 GEMM took its row count from the f16 copy of the activations. A static `[1, T, K]` input's
    copy is already `[T, K]`, so `TokenArg(copy, T)` gave T·T rows. The split-K partial slices were then addressed
    far outside the scratch buffer.
  * How it was found: GPU-AV reported nothing; the launch bisect (`SHARPLIGHT_VK_MAX_LAUNCHES`) found the launch.
  * Fix: the row count now comes from the original input. Both models now run with reduced precision.
* **Why reduced precision was slower than the default at batch 1**, and the fixes. There were three causes:
  * conv_cm f16 used 128-pixel tiles, which give half the workgroups of the strict form. Small maps are bound by
    staging. It now uses 64-pixel tiles when 128-pixel tiles cannot fill the device.
  * conv_cm needed a full 64-pixel tile, so 7x7 maps ran the SIMT conv at about 5 TFLOPS. Any map of 32+ pixels now
    takes the tensor-core conv, in both strict and reduced precision.
  * f32-weight matmuls of a few hundred rows went to gemm_cm2 on 24-36 workgroups. They now stay on the SIMT GEMM
    when the tensor-core grid is under 96 workgroups.
  * Result: resnet50 b1 reduced precision 1.08 → 0.70 ms (strict 0.81). Reduced precision is now at or below strict on
    every model except yolov8n (1.41 against 1.37, within noise).
* **Vulkan `gemm_wide`**: a 128x128 SIMT tile for f32 matmuls of 512+ rows and 256+ columns.
  * Each thread computes 8x8 outputs in two 4-wide strips. K steps through double-buffered shared memory, and f32
    weights are loaded as coalesced float4.
  * `SHARPLIGHT_GEMM_WIDE=0` disables it.
  * vit b8 768→3072: 280 → 201 µs.
* **Vulkan conv epilogue fusion**: an elementwise group that reads only a conv output (plus at most one residual of the
  same shape) runs in the conv's final store.
  * This covers conv_cm, conv_gemm, conv_dw and the split-K reduce, e.g. Relu or Add + Relu in ResNets.
  * ResNet50 drops from about 250 launches to 100.
  * `SHARPLIGHT_NO_CONV_EPILOGUE=1` disables it.
* **Vulkan bmm split-K**: few output tiles over a long K now split K across workgroups, and `bmm_reduce` sums the
  splits.
  * Example: whisper's attention P·V, 6 heads x 24 tiles over 1500 keys.
  * 201 → 86 + 4 µs per layer.
* **Vulkan LayerNorm**: rows of 512-16384 values always get one workgroup per row. vit: 12.3 → 4.1 µs at 197 rows.
* **ONNX translation**: `Div(Q·Kᵀ, c)` and `Mul(Q·Kᵀ, c)` fold into the BatchMatMul alpha when the product has no
  other reader.
* **CPU conv** (`ConvOp`), rewritten:
  * One parallel implicit GEMM covers all images (N·HO·WO columns). It replaces a GEMM per image and group with two
    barriers each.
  * Tasks own weight panels × output tiles. They pack each tile's im2col micro-panel just before use (L1), walk K in
    chunks that keep their A block in L2, and write through the epilogue.
  * When the output channels must be split across tasks:
    * B is packed once up front;
    * or, for stride-1 convs, the **shifted form** runs instead. The input is zero-padded once, and each tap's B rows are
      read in place through a per-k offset table (`Gemm.TileGather`). Stride-1 weights are packed tap-major, so both
      forms share them.
  * `SHARPLIGHT_CPU_NO_SHIFTED_CONV=1` disables the shifted form.
* **Vulkan kernels for detection ops** (merged from a parallel branch):
  * Ops: NonMaxSuppression, RoiAlign, MaxPool with indices, NonZero, TopK with a symbolic k, and Concat /
    ScatterElements along a symbolic axis.
  * Mechanism: a kernel writes the data-dependent count into a host-readable slot, and the segment ends after that node.
  * Effect: ssd-10, fasterrcnn-12 and maskrcnn-12 now run without CPU fallback segments. The R-CNNs went from 280 → 98
    and 283 → 108 ms wall.

### 25.2 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

Cells are min / median in ms. SL / ORT is the after-min of `sl-vulkan` over `ort-cuda`, which uses TF32. `(!)` marks a
parity-check failure. The full table is in `table-gpu.md`.

| model | batch | sl-vulkan | sl-vulkan-rp (min) | ort-cuda | ort-cuda-strict | SL / ORT CUDA |
|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.94 / 0.96 → **0.81** / 0.89 | 1.08 (!) → 0.70 (!) | 1.02 | 2.11 | **0.79x** |
| resnet50 | 8 | 3.86 / 4.06 → 3.09 / 3.12 | 3.17 (!) → 1.93 (!) | 2.00 | 3.72 | 1.54x |
| convnext-tiny | 1 | 1.10 / 1.14 → 1.12 / 1.13 | 1.43 (!) → 1.12 | 1.00 | 1.35 | 1.12x |
| convnext-tiny | 8 | 4.48 / 4.68 → 4.05 / 4.42 | 2.76 (!) → 2.75 (!) | 2.56 | 4.01 | 1.58x |
| efficientnet-lite4 | 1 | 0.95 / 0.97 → 0.88 / 0.90 | 1.05 → 0.76 | 0.68 | 8.18 | 1.31x |
| mobilenetv2 | 1 | 0.47 / 0.51 → 0.42 / 0.69 | 0.51 (!) → 0.39 (!) | 0.35 | 0.65 | 1.21x |
| mobilenetv2 | 8 | 0.92 / 1.01 → 0.73 / 0.81 | 0.95 (!) → 0.68 (!) | 0.75 | 1.35 | **0.97x** |
| yolov8n | 1 | 1.46 / 1.53 → 1.37 / 1.77 | 1.80 (!) → 1.41 (!) | 1.31 | 2.75 | 1.05x |
| yolov10n | 1 | 1.34 / 1.36 → 1.26 / 1.27 | 1.72 (!) → 1.28 (!) | 1.17 | 2.31 | 1.08x |
| ssd-10 (NMS) | 1 | 19.51 (!) → 18.27 (!) | 13.04 (!) → 10.67 (!) | 10.99 | 22.54 | 1.66x |
| bert-base (16 tok) | 1 | 1.26 / 1.30 → 1.30 / 4.06 | 1.70 (!) → 1.29 | 1.24 | 1.44 | 1.05x |
| bert-base (16 tok) | 8 | 3.72 / 4.03 → 4.40 / 10.78 | 3.84 (!) → 3.62 | 2.70 | 3.32 | 1.63x |
| vit-base | 1 | 2.76 / 2.80 → 2.61 / 3.23 | 2.38 (!) → 2.17 (!) | 1.34 | 2.46 | 1.94x |
| vit-base | 8 | 12.75 / 14.73 → 10.13 / 11.78 | 5.43 (!) → 5.19 (!) | 3.98 | 9.23 | 2.55x |
| whisper-tiny-encoder | 1 | 2.84 / 3.02 → 2.38 / 2.98 | fail → 2.02 (!) | 1.17 | 2.08 | 2.03x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.19 / 4.16 → 3.44 / 5.42 | 2.50 (!) → 2.48 (!) | 3.92 | 3.86 | 0.88x |
| phi3.5-mini q4f16 (GQA) | 1 | 4.62 (!) → 4.55 (!) | 3.50 (!) → 3.76 (!) | 4.19 | 3.46 | 1.08x |
| segformer-b0 | 1 | 3.02 / 3.49 → 3.43 / 16.26 | 3.33 → 3.01 | 2.33 | 3.39 | 1.47x |
| segformer-b0 | 8 | 22.63 / 24.83 → 22.95 / 24.51 | 20.95 → 20.83 | 19.90 | 23.38 | 1.15x |
| mobilesam-encoder | 1 | 6.23 / 6.95 → 5.96 / 6.31 | fail → 4.68 (!) | 6.15 | 8.00 | **0.97x** |
| raft-large | 1 | 58.19 / 64.14 → 55.99 / 59.38 | 45.01 (!) → 44.15 (!) | 29.25 | 77.00 | 1.91x |

The `sl-vulkan-rp` parity failures are expected: reduced precision is an opt-in, and its f16 products miss the strict
f32 tolerance. The two `fail` cells (device loss) are fixed. ssd-10 and phi-3.5 fail parity in strict precision too, as
in section 21: ssd-10 orders tied detections differently, and phi-3.5's fp16 reference is noisy.

GPU-timestamp minimum of `sl-vulkan` in ms (not affected by host load):

| model | batch | §21 | now | speedup |
|---|---|---|---|---|
| resnet50 | 1 / 8 | 0.91 / 3.80 | 0.78 / 3.05 | 1.16x / 1.25x |
| efficientnet-lite4 | 1 | 0.92 | 0.81 | 1.14x |
| mobilenetv2 | 1 / 8 | 0.44 / 0.88 | 0.39 / 0.70 | 1.12x / 1.26x |
| convnext-tiny | 8 | 4.41 | 4.00 | 1.10x |
| yolov8n / yolov10n | 1 | 1.34 / 1.32 | 1.26 / 1.23 | 1.07x |
| bert-base | 8 | 3.27 | 3.04 | 1.07x |
| vit-base | 1 / 8 | 2.72 / 12.62 | 2.52 / 9.89 | 1.08x / 1.28x |
| whisper-tiny-encoder | 1 | 2.75 | 2.16 | 1.28x |
| mobilesam-encoder | 1 | 6.03 | 5.65 | 1.07x |
| segformer-b0 | 1 / 8 | 2.74 / 18.09 | 2.71 / 17.34 | 1.01x / 1.04x |

The bert-base b8 and segformer b1 wall times are worse than in section 21, but their GPU times are equal or better.
The difference is host-side time under the machine's load: the medians of those rows are 2-5x their minima.

### 25.3 CPU: Sharplight vs onnxruntime CPU EP (16 threads, loaded machine)

Cells are min / median in ms.

| model | batch | sl-cpu (§21 → now) | ort-cpu (§21) | SL / ORT |
|---|---|---|---|---|
| resnet50 | 1 | 8.16 / 14.39 → 10.52 / 22.80 | 5.71 | 1.84x |
| resnet50 | 8 | 54.53 / 62.35 → **42.32** / 53.44 | 20.30 | 2.08x |
| mobilenetv2 | 8 | 15.31 / 18.21 → 12.62 / 14.08 | 6.41 | 1.97x |
| ssd-10 (NMS) | 1 | 305.86 → **197.58** | 204.96 | **0.96x** |
| bert-base (16 tok) | 8 | 22.23 → 22.94 | 26.68 | 0.86x |
| vit-base | 1 / 8 | 26.13 / 164.52 → 25.93 / 167.72 | 21.16 / 130.62 | 1.23x / 1.28x |
| whisper-tiny-encoder | 1 | 44.89 → 45.19 | 39.45 | 1.15x |
| segformer-b0 | 8 | 423.83 → 450.82 | 476.53 | 0.95x |
| yolov10n | 1 | 15.53 → 16.48 | 15.15 | 1.09x |

The full table is in `table-cpu.md`. Rows of ≤ 10 ms are slower than in section 21 because of the background load,
not because of a code change; their medians are 2-7x their minima. The larger conv workloads gained even under load:
resnet50 b8 by 22 %, ssd-10 by 35 %.

Per-shape conv benchmark (`SpatialBenchTests.ConvShapes`, the ResNet50 and YOLO conv shapes, 16 threads,
`conv-shapes-*.txt`):

* b8: 22.39 → 17.85 ms (one implicit GEMM with in-task packing) → 19.14 ms (plus the shifted form when the output
  channels are split).
* b1: 3.06 → 3.04 ms.
* The convs now reach 1.6-2.2 TFLOPS, about a third to 45 % of the 9950X's f32 peak. Section 21 reported 10-15 % for
  the whole networks.

The two b8 runs were taken under different load. In a later interleaved check of the full model, the shifted form gave
resnet50 b1 8.2 ms against 9.2 ms without it, and b8 the same within noise, so it stays on.

### 25.4 Correctness and tests

The branch was rebased onto `main` (after section 23, then again onto the bring-up round 6 merge, with the symbolic-k
TopK moved onto the shared `radix_select.slang`). The suites ran one at a time:

| suite | result |
|---|---|
| Cpu | 1177 passed, 29 skipped (first rebase) |
| Package | 29 passed (first rebase) |
| Vulkan | 1176 passed, 18 skipped |
| Onnx (`-parallelMode none`) | 415 passed |
| Families | 589 passed, 30 skipped |

* The ONNX conformance report (CPU and Vulkan) is unchanged from `main`: node tests 1829 / 1884 on both providers
  (8 wrong: the six Bernoulli RNG tests, `nonzero_example`, `strnormalizer ... twodim`), extra suites 170 / 172.
  The detection-op branch also measured Vulkan without fallback at 1745 → 1763.
* Families was OOM-killed twice on the first rebase while other agents' suites ran; it passed on the final branch when
  run alone with more than 45 GB available.
* An Onnx run that overlapped another suite failed `Zoo_Cpu(nomic-moe)` with "model does not fit" (available RAM).
  The isolated re-runs passed.

New tests:

* `ConvEpilogueTests`: the fused activation and residual on every conv kernel.
* `BatchMatMulTests.SplitK*`: static and symbolic batch, and symbolic rows.
* `SimtGemmTests.WideTiles` and `MatMulTests.LargeRowsSplitK`, the regression test for the device loss.
* `ConvMatrixTests` (7x7 maps, both precisions).
* `TranslationTests.ScaledProduct_*`.
* Differential tests for NMS, RoiAlign, MaxPool indices, NonZero and symbolic TopK / Concat / ScatterElements.

### 25.5 Commands

```bash
source bench/onnx/env.sh; export SHARPLIGHT_ONNX_MODELS=~/models/onnx
dotnet build bench/onnx/OnnxBench -c Release
python3 bench/onnx/parity_bench.py run after-gpu.jsonl --rounds 1 --engines sl-vulkan,sl-vulkan-rp [--models ...]
python3 bench/onnx/parity_bench.py table merged.jsonl bench/results/s21/after-gpu.jsonl
SHARPLIGHT_BENCH=1 SHARPLIGHT_CPU_THREADS=16 taskset -c 0-15 \
  dotnet tests/Sharplight.Cpu.Tests/bin/Release/net11.0/Sharplight.Cpu.Tests.dll -method "*ConvShapes"
```

### 25.6 Open gaps

* **Vulkan f32 matmuls against TF32.** vit, whisper, bert b8 and raft trail `ort-cuda` by 1.6-2.6x. Against
  `ort-cuda-strict` (no TF32), Sharplight is ahead on resnet50, convnext b1, mobilenetv2, the YOLOs, ssd-10, raft and
  mobilesam and bert b1, and behind on vit b1 (2.61 against 2.46), whisper (2.38 against 2.08), vit b8 (10.13 against
  9.23) and bert b8 (4.40 against 3.32, wall time under host load; its GPU time is 3.04).
  * The SIMT GEMM peaks at about 32 TFLOPS. TF32 tensor cores are not exposed through Vulkan cooperative matrices on
    this driver.
  * A 3xTF32-like split through bf16 (the conv's 3-part form) was slower than SIMT at these sizes (section 21.5).
* **Attention on Vulkan**: the decomposed MatMul → Softmax → MatMul is not fused. whisper still spends about 40 % of
  its GPU time in the two bmm launches and the softmax.
  * Next: a flash-style f32 kernel for the decomposed pattern with a batch axis.
  * The fused `Attention` op currently allows the f16 `fa` kernel for f32 K/V under strict precision. It should be
    gated like the other reduced-precision paths before a fusion routes more models to it.
* **CPU convolution**: resnet50 is still 1.4-2x behind ORT.
  * There are no NHWC / blocked layouts with layout propagation yet.
  * Winograd was not attempted.
  * b1 is limited by small-channel packing, Pool.max and Reduce.Mean (about 0.4 ms each).
* **CPU transformers**: no fused CPU attention for batched encoders.
  * The tiled kernel takes neither a batch axis nor spans.
  * The GEMMs at about 200 rows (vit b1) are about 78 % of the time.
* **Detection models**: each data-dependent count ends a Vulkan segment and goes through host memory. The R-CNNs take
  about 100 ms wall against about 34 ms of GPU time.
* **Measurement**: the CPU rows need a re-run on a quiet machine.

## 26. ONNX parity round 3: fused encoder attention, split-f16 tensor-core GEMM (2026-11-14)

The third round of section 21 on the same machine, models and harness (`bench/onnx/parity_bench.py`). Raw data is in
[`bench/results/s26/`](../bench/results/s26). onnxruntime 1.31.1 (CPU EP, CUDA EP with TF32 and with TF32 off) was
re-run in this session, interleaved with Sharplight: every model runs all engines back to back, two rounds.

How the numbers were taken:

* Cells are min / median in ms over a process's runs, minimum over the two rounds.
* The GPU was shared with the desktop and two other agents. The GPU-timestamp minimum (26.3) is not affected by host
  load; wall-clock medians are.
* The CPU rows ran at a load average of 10-30 (median 21) from the other agents' builds and test suites (`cpu-load.log`), so they
  are upper bounds.
* New engine `sl-vulkan-split`: Sharplight with `CompileOptions.SplitHalfMatMul` (`OnnxBench --split-f16`).

### 26.1 What changed

* **Fused attention for ONNX encoders.** The translator recognises `MatMul → (scale) → (mask Add) → Softmax → MatMul`
  when nothing else reads the probabilities, in the rank-4 head layout of vit / bert / segformer exports and the
  rank-3 `[B·H, S, D]` layout of whisper exports, and emits one `Attention` op. com.microsoft `Attention` and
  `MultiHeadAttention` without requested scores take the same op. `OnnxImportOptions.FuseAttention = false` keeps
  the elementary ops.
  * The `Attention` op got a batched form (`sequence_rows` / `sequence_keys`: equal-length sequences that only see
    their own keys, `BatchedAttention`) and f32 additive masks (one bias per key row, or `[T, S]`).
  * CPU: the tiled kernel takes the batched form and masks, and reads full f32 K/V blocks in place instead of copying
    them per task. Whisper-tiny encoder attention (1500 rows, 6 heads) runs at about 2 TFLOPS
    (`BenchTests.EncoderAttentionThroughput`, default thread count, loaded machine).
* **Strict precision for f32 attention on Vulkan.** The flash kernel ran f32 K/V through single f16 products whenever
  the device had f16 matrices, also under strict precision. Now:
  * strict precision splits Q, K, V and P into f16 hi + lo parts and computes hi·hi + hi·lo + lo·hi (three
    tensor-core products, f32 accumulation, P scaled by 2^12 before its split). The split rounds on the bits, since
    the compiler folds a `float(half(x))` round trip to `x`, which had silently left the lo parts at zero;
  * reduced precision keeps single f16 products;
  * `SHARPLIGHT_ATTN_SPLIT_F16=0` sends strict f32 attention to the scalar kernel.
  * Accuracy against the CPU oracle (`FullAttentionTests`): within 5e-6 absolute and relative under strict precision.
* **Encoder flash kernel `fe`** for plain bidirectional attention (no spans, window, causal mask, score biases,
  softcap or sinks; optional mask, batched or cross form; padded head sizes up to 128).
  * 64 queries per workgroup of 4 subgroups. Each subgroup keeps its Q fragments and O accumulators in registers for
    the whole key loop and rescales O with an element-wise product by a matrix of row factors.
  * The next K/V tile is fetched into registers during the current one: two workgroup barriers per 32-key tile
    instead of about five in `fa`.
  * Key ranges split over several workgroups (combine kernel) when there are few query blocks.
  * whisper-tiny: 180 → 115 µs per layer (split f16, strict). `SHARPLIGHT_ATTN_FE=0` falls back to `fa`.
* **Split-f16 tensor-core GEMM, opt-in** (`CompileOptions.SplitHalfMatMul`). `gemm_cm2` with `GEMM_SPLIT` splits f32
  activations and weights into f16 hi + lo parts while staging and computes the same three products.
  * Relative RMS error against an f64 reference (`SplitHalfGemmTests`, weights ~0.03, activations ~1):

    | shape | f32 SIMT (strict) | split f16 | single f16 (reduced) |
    |---|---|---|---|
    | 1024 x 768 → 2048 | 3.5e-7 | 1.7e-6 | 2.6e-4 |
    | 1000 x 384 → 1536 | 1.8e-7 | 1.2e-6 | 2.6e-4 |
    | 1576 x 3072 → 768 | 5.0e-7 | 2.9e-6 | 2.6e-4 |

    About 5-7x the f32 error and 100-200x below single f16 products (TF32 has the same 10-bit mantissa as f16). The lo
    parts of values below 2^-3 lose bits to f16 subnormals, which is most of the gap to f32. Because it is not
    f32-exact, it stays opt-in.
  * It needs `DeviceFeatures.MatrixF16`; `SHARPLIGHT_GEMM_SPLIT_F16=0` disables it. Grids too small for tensor cores
    keep the SIMT GEMM, as under reduced precision.
  * Section 21.5 found a three-part bf16 split slower than the SIMT GEMM at these sizes. Two f16 parts (3 products
    instead of 6) are faster: vit b8 [768→3072] 201 → 147 µs, whisper [384→1536] 55 → 37 µs.
* **Vulkan CumProd**: the generated scan kernel in product mode instead of the CPU fallback.
* Tried and dropped: two vectors of query rows per CPU attention task (more reuse of each K/V value). It halved the
  task count, and vit's 197-row sequences then had too few tasks: 94 → 207 µs per layer; whisper was even.

### 26.2 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

Cells are min / median in ms; §25 is the `sl-vulkan` min of section 25. SL / ORT is `sl-vulkan` (strict, f32-accurate)
over `ort-cuda` (TF32). `(!)` marks a parity-check failure. The full table is in `table-gpu.md`.

| model | batch | §25 | sl-vulkan | sl-vulkan-split | sl-vulkan-rp | ort-cuda | ort-cuda-strict | SL / ORT CUDA |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.81 | 0.81 / 0.82 | 0.81 | 0.70 (!) | 1.01 | 2.11 | **0.80x** |
| resnet50 | 8 | 3.09 | 3.10 / 3.30 | 3.09 | 1.93 (!) | 1.99 | 3.71 | 1.56x |
| convnext-tiny | 1 | 1.12 | 1.13 / 1.22 | 1.12 | 1.13 | 1.01 | 1.40 | 1.12x |
| convnext-tiny | 8 | 4.05 | 4.08 / 4.15 | 3.20 | 2.66 (!) | 2.55 | 4.00 | 1.60x |
| efficientnet-lite4 | 1 | 0.88 | 0.84 / 0.86 | 0.84 | 0.76 | 0.67 | 7.93 | 1.25x |
| mobilenetv2 | 1 | 0.42 | 0.41 / 0.43 | 0.41 | 0.38 (!) | 0.37 | 0.65 | 1.11x |
| mobilenetv2 | 8 | 0.73 | 0.73 / 0.74 | 0.72 | 0.67 (!) | 0.75 | 1.34 | **0.97x** |
| yolov8n | 1 | 1.37 | 1.38 / 2.04 | 1.38 | 1.34 (!) | 1.31 | 2.76 | 1.05x |
| yolov10n | 1 | 1.26 | 1.27 / 3.76 | 1.27 | 1.25 (!) | 1.17 | 2.31 | 1.08x |
| ssd-10 (NMS) | 1 | 18.27 (!) | 17.72 (!) | 17.68 (!) | 9.04 (!) | 11.22 | 22.83 | 1.58x |
| bert-base (16 tok) | 1 | 1.30 | 1.22 / 3.70 | 1.22 | 1.18 (!) | 1.24 | 1.43 | **0.98x** |
| bert-base (16 tok) | 8 | 4.40 | 3.72 / 9.89 | 3.46 | 3.32 (!) | 3.17 | 3.37 | 1.17x |
| vit-base | 1 | 2.61 | 2.44 / 3.00 | 2.14 | 1.95 (!) | 1.35 | 2.47 | 1.80x |
| vit-base | 8 | 10.13 | 9.85 / 11.14 | 7.15 | 4.82 (!) | 4.00 | 9.33 | 2.47x |
| whisper-tiny-encoder | 1 | 2.38 | **1.81** / 2.06 | 1.65 | 1.40 (!) | 1.18 | 2.09 | 1.53x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.44 | 3.15 / 13.82 | 3.17 | 2.44 (!) | 3.81 | 3.83 | **0.83x** |
| phi3.5-mini q4f16 (GQA) | 1 | 4.55 (!) | 4.53 (!) | 4.55 (!) | 3.44 (!) | 3.49 | 3.50 | 1.30x |
| segformer-b0 | 1 | 3.43 | 2.81 / 7.06 | 2.87 | 2.68 | 2.40 | 3.46 | 1.17x |
| segformer-b0 | 8 | 22.95 | 20.66 / 23.50 | 20.58 | 18.61 | 33.11 | 23.28 | **0.62x** |
| mobilesam-encoder | 1 | 5.96 | 5.96 / 7.70 | 5.20 | 4.40 (!) | 6.28 | 8.16 | **0.95x** |
| raft-large | 1 | 55.99 | 54.98 / 60.49 | 55.03 | 42.06 (!) | 30.71 | 73.52 | 1.79x |

* As in section 25, the `sl-vulkan-rp` parity failures are expected: reduced precision is opt-in, and its f16 products
  miss the strict f32 tolerance.
* ssd-10 and phi-3.5 fail parity in every precision, as in section 21: ssd-10 orders tied detections differently, and
  phi-3.5's fp16 reference is noisy.
* `sl-vulkan-split` passes parity on every model that passes in strict precision.
* ORT's segformer b8 min (33.11, median 115.51) was disturbed and is not a real lead. `ort-cuda-strict` gives 23.28.

Against `ort-cuda-strict` (no TF32, the like-for-like comparison with f32 products), strict Sharplight is now ahead or
level on every model except bert b8 (3.72 against 3.37 wall; GPU time 2.99), vit b8 (9.85 against 9.33) and phi-3.5.
whisper went from 2.38 to 1.81 ms (strict ORT: 2.09), and vit b1 from 2.61 to 2.44 (2.47).

### 26.3 GPU timestamps and split-f16 accuracy

GPU-timestamp minimum in ms, which host load does not affect, and the worst relative RMS of any output against
onnxruntime's CPU EP (`table-gpu-timestamps.md`):

| model | batch | §25 strict | strict | split f16 | reduced | worst rms (strict / split / reduced) |
|---|---|---|---|---|---|---|
| whisper-tiny-encoder | 1 | 2.16 | **1.70** | 1.55 | 1.30 | 3.1e-6 / 3.2e-6 / 4.7e-4 |
| vit-base | 1 | 2.52 | 2.39 | 2.09 | 1.91 | 1.7e-6 / 2.2e-6 / 4.9e-4 |
| vit-base | 8 | 9.89 | 9.65 | **6.95** | 4.72 | 1.6e-6 / 1.1e-5 / 5.9e-4 |
| bert-base | 8 | 3.04 | 2.99 | 2.89 | 2.82 | 2.2e-6 / 2.8e-6 / 2.5e-4 |
| segformer-b0 | 1 | 2.71 | 2.55 | 2.55 | 2.38 | 2.4e-6 / 2.2e-6 / 9.3e-5 |
| segformer-b0 | 8 | 17.34 | 16.19 | 15.97 | 14.03 | 3.0e-6 / 2.4e-6 / 1.7e-4 |
| convnext-tiny | 8 | 4.00 | 4.00 | **3.15** | 2.62 | 3.3e-6 / 7.4e-6 / 6.4e-4 |
| mobilesam-encoder | 1 | 5.65 | 5.62 | 5.01 | 4.24 | 1.3e-6 / 3.4e-6 / 5.9e-4 |

* Strict precision gains come from fused attention: whisper 21 %, segformer 6-7 %, vit 2-4 % (attention is a smaller
  share of vit, and its 197-key sequences already ran well as bmm).
* Split f16 takes another 9 % on whisper and 28 % on vit b8, and 21 % on convnext b8. Its error stays within 1.1e-5,
  2-7x strict and 20-50x below reduced precision. ORT's TF32 default rounds its inputs to the same 10-bit mantissa as
  single f16, so the split should also be more accurate than that default. This was not measured.
* The models where split f16 does nothing (resnet50, mobilenet, YOLO) have their matmul time in convs or in grids too
  small for tensor cores.

### 26.4 CPU: Sharplight vs onnxruntime CPU EP (16 threads, loaded machine)

Cells are min / median in ms; §25 is the `sl-cpu` min of section 25. During the run the 1-minute load average was
10-30 (median 21, `cpu-load.log`), from other agents' builds and test suites. Both engines were interleaved, so the
SL / ORT ratios are fairer than the absolute numbers. Medians are 1.5-6x the minima. The full table is in
`table-cpu.md`.

| model | batch | §25 | sl-cpu | ort-cpu | SL / ORT |
|---|---|---|---|---|---|
| resnet50 | 1 | 10.52 | 8.01 / 11.40 | 5.92 | 1.35x |
| resnet50 | 8 | 42.32 | 41.47 / 56.98 | 21.39 | 1.94x |
| convnext-tiny | 8 | 102.09 | 93.42 / 111.91 | 76.27 | 1.22x |
| mobilenetv2 | 8 | 12.62 | 14.16 / 55.11 | 8.18 | 1.73x |
| ssd-10 (NMS) | 1 | 197.58 | 206.94 / 301.35 | 212.38 | **0.97x** |
| bert-base (16 tok) | 1 | 13.06 | 13.66 / 37.17 | 11.43 | 1.20x |
| bert-base (16 tok) | 8 | 22.94 | 21.86 / 65.89 | 27.47 | **0.80x** |
| vit-base | 1 | 25.93 | 24.70 / 38.45 | 22.25 | 1.11x |
| vit-base | 8 | 167.72 | **148.61** / 271.10 | 132.91 | 1.12x |
| whisper-tiny-encoder | 1 | 45.19 | **22.70** / 53.11 | 54.08 | **0.42x** |
| segformer-b0 | 1 | 113.94 | 88.77 / 99.66 | 42.52 | 2.09x |
| segformer-b0 | 8 | 450.82 | **343.59** / 378.87 | 481.19 | **0.71x** |
| mobilesam-encoder | 1 | 229.03 | 199.73 / 585.75 | 204.09 | **0.98x** |
| raft-large | 1 | 1841.66 | 1570.58 / 1636.30 | 1606.12 | **0.98x** |

* Fused CPU attention:
  * whisper-tiny's encoder halved, 45.2 → 22.7 ms. That is 0.42x of ORT in this run, and 0.58x of ORT's idle-machine
    39.45 ms from section 21.
  * segformer b8 is 24 % faster and vit b8 11 %.
* segformer b1 (2.1x behind ORT) was not profiled in this round.
* The conv models (resnet50, mobilenetv2, efficientnet) did not change this round. Their rows move with the load.

### 26.5 Correctness and tests

The branch was rebased onto `main` (after bring-up round 8). The suites ran one at a time under the suite lock. A last
rebase onto the NuGet packaging commits was followed by a full build and a Package re-run (29 passed).

| suite | result |
|---|---|
| Cpu | 1193 tests, 0 failed, 31 skipped |
| Vulkan | 1229 tests, 0 failed, 22 skipped |
| Onnx (`-parallelMode none`) | 426 passed |
| Package | 29 passed |
| Families | 660 tests, 0 failed, 32 skipped |

* The first Families run lost 8 CPU generation tests to "model does not fit" while another agent's 26 GB benchmark
  held RAM. It passed when re-run with 50 GB available.
* ONNX conformance report (CPU and Vulkan): node tests 1829 / 1884 on both providers (the same 8 wrong and 47 import
  failures as in section 25), extra suites 170 / 172.
  * Vulkan without the CPU fallback: 1745 → 1772 node tests (`docs/onnx-support.md` had last been generated before
    the section 25 detection ops); ai.onnx ops fully on Vulkan 147 → 152.
  * CumProd, NonMaxSuppression, RoiAlign, Compress and MaxPool (indices) no longer touch the CPU.

New tests:

* `AttentionPatternTests` (ONNX): both export layouts fuse, masks, fusion off, probabilities with a second reader,
  fused against elementary on CPU and Vulkan.
* `BatchedAttentionTests` (CPU): batched form and both mask forms on the tiled and general kernels.
* `FullAttentionTests` (Vulkan): strict and reduced accuracy of the split flash kernel, masks, batched cross attention.
* `SplitHalfGemmTests` (Vulkan): the error table above.
* `DataOpTests.CumProd` (Vulkan).
* `BenchTests.EncoderAttentionThroughput` (CPU, `SHARPLIGHT_BENCH=1`).

### 26.6 Commands

```bash
export SHARPLIGHT_ONNX_MODELS=~/models/onnx ORT_PYTHON=~/venvs/ort/bin/python3   # plus the CUDA libs of bench/onnx/env.sh
dotnet build bench/onnx/OnnxBench -c Release
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-split,sl-vulkan-rp,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 2 --engines sl-cpu,ort-cpu
python3 bench/onnx/parity_bench.py table gpu.jsonl
SHARPLIGHT_BENCH=1 dotnet tests/Sharplight.Cpu.Tests/bin/Release/net11.0/Sharplight.Cpu.Tests.dll -method "*EncoderAttentionThroughput"
```

### 26.7 Open gaps

* **Vulkan f32 GEMMs against TF32.** In strict precision, vit (1.8-2.5x), raft (1.8x), convnext b8 and resnet50 b8
  (1.6x) and whisper (1.5x) still trail `ort-cuda`.
  * The opt-in split-f16 GEMM closes part of the gap: vit b8 is at 1.8x and whisper at 1.4x.
  * Making split f16 the strict default would need f32-level accuracy. The lo parts lose bits to f16 subnormals,
    which a per-tile exponent offset could fix.
  * Split f16 for convs (the implicit-GEMM conv kernels) and for bmm was not done.
  * The f32 SIMT GEMM was not changed in this round.
* **QKV projection fusion**: the three Q / K / V matmuls of the encoders still run as three GEMMs (vit b1 has
  197-row grids that underfill the GPU).
* **CPU convolution**: resnet50 is still at 1.35x / 1.9x of ORT and mobilenetv2 b8 at 1.7x. Not attempted in this round:
  NCHWc or NHWC layouts with layout propagation, Winograd for the 3x3 convs (51 % of resnet50's CPU time in
  `SHARPLIGHT_CPU_PROFILE`), and output-channel blocking of the 1x1 convs (about 37 %).
* **CPU transformer GEMMs**: vit and bert b1 are at 1.1-1.2x. Their time is in the ~200-row GEMMs, which were not
  reblocked.
* **Detection models**: each data-dependent count still ends a Vulkan segment and goes through host memory (unchanged
  from section 25).
* **Vulkan fallbacks**: Det / Inverse, integer CumSum and volumetric GridSample still run on the CPU.
* **Measurement**: the CPU rows ran at load 10-30 and need a re-run on a quiet machine.

## 27. ONNX parity round 4: CPU conv scheduling, fused Q/K/V projections, Vulkan Det / Inverse / volumetric GridSample (2026-11-14)

Same machine and harness as section 26 (`bench/onnx/parity_bench.py`, 2 interleaved rounds, ORT 1.31 in the same
session). Raw data is in `bench/results/s27/`.

### 27.1 What changed

* **CPU convolution scheduling** (`ConvOp`).
  * The task split compares splitting the weight panels with splitting the output tiles by operand traffic. Blocks
    that stay in L2 are counted once per thread. The plan aims at about 6 tasks per thread. `SHARPLIGHT_CPU_CONV_PLAN=tiles`
    restores the old split.
  * Strided 1x1 convs without leading padding subsample their input once and then run like the in-place 1x1 form.
    resnet50's 256x56 → 128 stride-2 conv went from 98 to 63 µs.
  * The stride-2 im2col gather and the subsample use an even-lane AVX-512 permute (`Simd.CompressEven`).
  * The shifted stride-1 form runs each tap through the strided GEMM kernel when a group has at least 16 channels.
  * The copy-out of a C block applies the epilogue in the same pass.
* **Max pooling** runs per block of output rows: a vectorized vertical max, then the horizontal window. resnet50's
  pool went from 258 to 39 µs.
* **Parallel loops** (`CpuThreadJob`) give each participant one contiguous range and claim several items at a time;
  idle participants steal from other ranges. Consecutive ops that split their outputs alike then run each region on
  the same core. An empty `CpuParallel.For` over 1024 items went from 38 to 12 µs. `SHARPLIGHT_CPU_AFFINITY=0` goes
  back to one shared counter.
* **JIT**: executables compile methods with loops fully optimized on first use
  (`TieredCompilationQuickJitForLoops=false`). Before, short runs and first runs measured tier-0 kernels: a 64-channel
  3x3 conv took 0.66 ms instead of 0.11 ms.
* **Strided copies out of transposed views** copy 16 x 16 tiles instead of one element per cache line. vit's
  class-token Concat went from 380 to 111 µs.
* **Fused sibling projections** (`OnnxImportOptions.FuseProjections`, default on, for both providers).
  * Several MatMuls of one activation by constant weights with the same K become one MatMul over the concatenated
    weights, followed by one slice per product. This covers an encoder's Q / K / V, or K / V where Q reads another
    input (segformer).
  * When each product's only reader is a bias Add, the biases are concatenated too.
  * The CPU keeps reshapes of the slices as views, including merges of the symbolic batch axis. A missing
    element-count check in the strided reshape, which this exposed, is fixed.
  * Vulkan: the tensor-core GEMM splits K in two when two halves fill the last wave of about 192 tiles much better.
    vit's [768 → 2304] product at 1576 rows went from 141 to 117 µs in split-f16 mode.
  * `OnnxBench --no-fuse-projections` turns the fusion off for A/B runs.
* **Vulkan kernels instead of the CPU fallback**:
  * Det and Inverse: one thread per matrix, LU / Gauss-Jordan with partial pivoting in f32 (the CPU uses f64);
  * volumetric GridSample: linear and nearest, all padding modes;
  * CumSum / CumProd over int64 inputs, read and converted per element.
* **Profiling**: `SHARPLIGHT_CPU_PROFILE=min` sums the per-instance minimum of every op. This is robust on a shared
  machine. `CpuOp.Detail` prints the shapes of the slowest instances.
* Tried and dropped:
  * Winograd F(2x2, 3x3) on NCHW. It was accurate (RMS 1.30e-6 against 1.34e-6 for the direct conv). It was 20-25 %
    faster for 64-channel 56x56 convs at b8, about 7 % at 128 channels, and slower with more channels or at b1: in
    NCHW the input and output transforms cost about as much as the multiplies they save.
  * Phase-plane shifted strided convs: 2x slower at b8.
  * Forcing the shifted form for single-block convs: its padded copy costs about 20 µs.
  * Reblocking the ~200-row encoder GEMMs, in two variants:
    * makespan-optimal row blocks;
    * several weight panels per task with K chunks outermost.

    Neither beat the old plan in `BenchTests.MatMulThroughput` beyond its ±10 % noise. In the model, vit's
    [3072 → 768] product takes 640 µs against 400 µs in isolation: each layer streams its 28 MB of weights from DRAM.

### 27.2 CPU: Sharplight vs onnxruntime CPU EP (16 threads)

Cells are min / median in ms. §26 is the `sl-cpu` min of section 26. The 1-minute load average was 7-34 (median 13,
`cpu-load.log`). That is lower than in section 26 (median 21), so the interleaved SL / ORT ratio is the comparable
number. The segformer rows are a re-run after the reshape fix (`cpu-segformer.log`). The full table is in
`table-cpu.md`.

| model | batch | §26 | sl-cpu | ort-cpu | SL / ORT (§26) | SL / ORT |
|---|---|---|---|---|---|---|
| resnet50 | 1 | 8.01 | 5.87 / 6.90 | 6.17 | 1.35x | **0.95x** |
| resnet50 | 8 | 41.47 | 34.22 / 47.80 | 22.16 | 1.94x | 1.54x |
| convnext-tiny | 1 | 20.40 | 14.02 / 21.11 | 13.55 | - | 1.03x |
| convnext-tiny | 8 | 93.42 | 81.50 / 103.46 | 84.14 | 1.22x | **0.97x** |
| efficientnet-lite4 | 1 | 8.25 | 6.92 / 8.79 | 4.36 | - | 1.59x |
| mobilenetv2 | 1 | 2.24 | 1.98 / 2.80 | 1.61 | - | 1.23x |
| mobilenetv2 | 8 | 14.16 | 10.78 / 18.77 | 7.04 | 1.73x | 1.53x |
| yolov8n | 1 | 15.66 | 9.76 / 16.64 | 13.86 | - | **0.70x** |
| yolov10n | 1 | 19.79 | 12.07 / 17.16 | 15.84 | - | **0.76x** |
| ssd-10 (NMS) | 1 | 206.94 | 180.12 / 195.47 | 219.94 | 0.97x | **0.82x** |
| bert-base (16 tok) | 1 | 13.66 | 9.61 / 11.89 | 12.18 | 1.20x | **0.79x** |
| bert-base (16 tok) | 8 | 21.86 | 20.50 / 28.61 | 27.70 | 0.80x | **0.74x** |
| vit-base | 1 | 24.70 | 19.77 / 27.57 | 23.04 | 1.11x | **0.86x** |
| vit-base | 8 | 148.61 | 132.85 / 180.56 | 139.73 | 1.12x | **0.95x** |
| whisper-tiny-encoder | 1 | 22.70 | 21.99 / 32.42 | 43.35 | 0.42x | **0.51x** |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 18.03 | 17.61 / 23.32 | 21.63 | - | **0.81x** |
| segformer-b0 | 1 | 88.77 | 38.73 / 44.90 | 42.09 | 2.09x | **0.92x** |
| segformer-b0 | 8 | 343.59 | 249.47 / 273.26 | 480.67 | 0.71x | **0.52x** |
| mobilesam-encoder | 1 | 199.73 | 143.35 / 390.45 | 219.09 | 0.98x | **0.65x** |
| raft-large | 1 | 1570.58 | 1230.48 / 2321.20 | 1767.96 | 0.98x | **0.70x** |

* Every model at batch 1 except efficientnet-lite4 and mobilenetv2 is now level with or ahead of ORT. The
  depthwise-heavy models (efficientnet, mobilenetv2) and resnet50 b8 are the gaps left.
* resnet50 b1 (sum of per-op minima 7.0 → 5.8-6.0 ms) gained from the pool, the strided 1x1 convs, the stem's
  vectorized stride-2 gather and the traffic-based split.
  * At b8 the remaining gap is the 3x3 convs: about 13.9 ms against 8.5 ms in ORT's profiler, where NCHWc reaches
    about 3.6 TFLOPS.
* segformer b1 2.09x → 0.92x: its K / V projections from the reduced sequence are now one GEMM, and its convs and
  pooling gained from the scheduling changes.
* vit b1: sum of per-op minima 22.7 → 21.0 ms.
  * The fused [768 → 2304] product takes 3.7 ms over 12 layers, against 5.6 ms for 36 separate products.
  * The class-token Concat went from 380 to 111 µs.
* ORT's medians are 1.1-2x its minima under this load, Sharplight's 1.1-1.9x.

### 27.3 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

Cells are min / median in ms. SL / ORT is `sl-vulkan` (strict, f32-accurate) over `ort-cuda` (TF32). `(!)` marks a
parity-check failure (expected for reduced precision, and for ssd-10 and phi-3.5 in every precision, as in section
26). The full table is in `table-gpu.md`.

| model | batch | §26 | sl-vulkan | sl-vulkan-split | sl-vulkan-rp | ort-cuda | ort-cuda-strict | SL / ORT CUDA |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.81 | 0.82 / 1.00 | 0.80 | 0.69 (!) | 1.01 | 2.12 | **0.81x** |
| resnet50 | 8 | 3.10 | 3.09 / 3.50 | 3.10 | 1.93 (!) | 2.00 | 3.73 | 1.55x |
| convnext-tiny | 8 | 4.08 | 4.11 / 4.64 | 3.22 | 2.68 (!) | 2.56 | 4.04 | 1.61x |
| mobilenetv2 | 8 | 0.73 | 0.73 / 0.76 | 0.73 | 0.67 (!) | 0.75 | 1.35 | **0.97x** |
| bert-base (16 tok) | 1 | 1.22 | 1.22 / 1.43 | 1.21 | 1.19 (!) | 1.24 | 1.44 | **0.99x** |
| bert-base (16 tok) | 8 | 3.72 | 3.55 / 4.59 | 3.39 | 3.34 (!) | 2.73 | 3.37 | 1.30x |
| vit-base | 1 | 2.44 | 2.47 / 2.87 | 2.15 | 1.97 (!) | 1.34 | 2.46 | 1.84x |
| vit-base | 8 | 9.85 | 9.77 / 11.38 | 7.19 | 4.86 (!) | 4.11 | 9.19 | 2.38x |
| whisper-tiny-encoder | 1 | 1.81 | 1.84 / 2.09 | 1.60 | 1.30 (!) | 1.16 | 2.08 | 1.59x |
| segformer-b0 | 8 | 20.66 | 21.12 / 24.84 | 20.46 | 18.54 | 42.55 | 46.83 | **0.50x** |
| mobilesam-encoder | 1 | 5.96 | 5.89 / 6.85 | 5.20 | 4.39 (!) | 6.08 | 7.98 | **0.97x** |
| raft-large | 1 | 54.98 | 56.17 / 62.50 | 56.38 | 43.32 (!) | 29.46 | 74.54 | 1.91x |

GPU-timestamp minima (`table-gpu-timestamps.md`, unaffected by host load) moved by at most ±3 % against section 26 on
every model. Fusing the projections is neutral on Vulkan:

* In vit's per-dispatch profile, at b8 in strict precision, the fused [768 → 2304] SIMT GEMM takes 160 µs against
  3 x 59 µs plus split-K reduces for the separate products.
* Attention then copies the strided K and V into dense buffers (2 x 4 µs). The Vulkan attention kernels need dense
  K / V rows.
* The whole-graph minima stayed level: vit b8 9.65 → 9.64 ms strict, 6.95 → 7.01 split; bert b8 2.99 → 3.02.

### 27.4 Correctness and tests

New tests:
* `ProjectionFusionTests`: the fused graph has one MatMul and three slices, and computes what the separate products
  compute on the CPU and Vulkan;
* `DeviceLinalgOpsTests`: Vulkan Det / Inverse / volumetric GridSample / int64 CumSum against the CPU over a symbolic
  batch;
* `TransposedCopyTests`;
* `StridedReshapeTests`;
* new conv and pool cases in the CPU tests.

The Vulkan conformance tests of Det, Inverse, volumetric GridSample and CumSum pass without the CPU fallback.

The branch was rebased onto `main`, and the suites ran one at a time under the suite lock:

| suite | result |
|---|---|
| Cpu | 1214 tests, 0 failed, 36 skipped |
| Vulkan | 1229 tests, 0 failed, 22 skipped |
| Onnx (`-parallelMode none`) | 436 passed |
| Package | 32 passed |
| Families | 660 tests, 0 failed, 32 skipped |

Conformance report (cpu and vulkan): 1829 of 1884 node tests and 170 of 172 extra tests pass on both providers, as in
section 26. On Vulkan without the CPU fallback, 1778 node tests (94.4 %, was 1745) and 128 extra tests (was 127) pass.

### 27.5 Commands

```bash
export SHARPLIGHT_ONNX_MODELS=~/models/onnx ORT_PYTHON=~/venvs/ort/bin/python3   # plus the CUDA libs of bench/onnx/env.sh
dotnet build bench/onnx/OnnxBench -c Release
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-split,sl-vulkan-rp,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 2 --engines sl-cpu,ort-cpu
python3 bench/results/s27/gpu_timestamp_table.py gpu.jsonl ../s26/gpu.jsonl
SHARPLIGHT_CPU_PROFILE=min dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/vit-base-patch16-224.json cpu --batch 1
SHARPLIGHT_BENCH=1 dotnet tests/Sharplight.Cpu.Tests/bin/Release/net11.0/Sharplight.Cpu.Tests.dll -method "*MatMulThroughput"
```

### 27.6 Open gaps

* **CPU convolution at b8 and depthwise convs**: resnet50 b8 1.54x, mobilenetv2 1.2-1.5x, efficientnet-lite4 1.6x.
  * The 3x3 convs need a blocked layout (NCHWc or NHWC, propagated between convs, with transforms at the graph
    boundaries) to reach ORT's ~3.6 TFLOPS.
  * Winograd only pays off in such a layout.
* **CPU encoder GEMMs from DRAM**: in the model, the ~200-row GEMMs run at about 60 % of their speed with the
  weights in cache. Prefetching the next layer's weights or keeping activations resident would be the next step. The
  task planner was not the limit.
* **Vulkan f32 GEMMs against TF32** are unchanged: vit 1.8-2.4x, raft 1.9x, resnet50 / convnext b8 1.6x, whisper
  1.6x.
  * Not attempted: split f16 at f32-level accuracy (per-tile exponent offset for the subnormal lo parts), split f16
    for convs and bmm, and the SIMT GEMM itself.
* **Vulkan attention with strided K / V** would remove the two copies that the fused projections add.
* **Detection models**: data-dependent counts still round-trip through the host (unchanged since section 25).
* **Vulkan fallbacks left**: LinearAttention, DeformConv, Unique, string / ai.onnx.ml ops and TfIdfVectorizer.

## 28. Row-invariant MTP verification, MoE tile launches without a barrier, norm-written f16 copies; serving against vLLM (2026-11-14)

Follow-up to section 24 (LLM performance; section 26 is the ONNX round). Base is `main` at `66a55a8` (correctness runs: the branch rebased on `44622a2` against that `main`), built side by side; base and new runs alternate. The GPU was shared
with the user's applications and other agents throughout, so the tables use minimum GPU times or best rounds and give
every round. Raw results: [`bench/results/s28/`](../bench/results/s26).

### 28.1 MTP output differed from plain decoding on long prompts (fixed)

On a 832-token prompt, the 27B Q8_0 with `--speculate-mtp 3` produced different text from plain decoding (the plain output
matched `main`). Cause: a verification step runs 1 + k rows of one sequence, and several kernels choose their arithmetic by
row count:

* Q8_0 weights: one row takes the f32 split-K matvec, 2-8 rows the integer-dot matvec on q8_1-quantized activations
  (logit differences around 0.2);
* FP8 weights: 2-16 rows take the skinny tensor-core GEMM on f16 activations;
* decode attention: the number of key splits depends on the row count;
* the f32 matvec itself: its variants for 2, 4 and 8 rows were contracted differently by the driver from the 1-row
  variant (last-bit differences, enough to flip a near-tie after a few hundred tokens).

So the rows of a verification step did not reproduce the logits of the single-row steps they replace, and on long
outputs a near-tie eventually flipped a token.

The fix keeps verification steps on the single-row arithmetic:

* `CompileOptions.RowInvariantRows` (`SHARPLIGHT_ROW_INVARIANT`): steps of up to that many rows (capped at the matvec range,
  8 on Vulkan) compute every row bit-identically to a step of that row alone. `LlmModel.CreateBatchDecoder` sets it to
  k + 1 when it verifies k drafts.
* `GraphLowerer.RowInvariance.cs`: in such compiles, the f32 matvec serves 2, 4 and 8 rows with the single-row kernel and K
  split, the integer-dot matvec and the skinny GEMM start above the range, formats whose single rows use the integer-dot
  matvec keep its single-row layout, and decode attention splits keys as for one row (`DEC_INVARIANT_ROWS`).
* `matmul.slang` (`dot8`, `MV_FMA_CHAIN`): row-invariant compiles write the sum of each 8-weight group as an explicit fma
  chain in every matvec variant. Plain compiles keep the plain sum: the chain matched the driver's contraction on the 27B
  models but not on gpt-oss-20b and OLMoE (a first version used it everywhere and changed their short-prompt output).

Tests: `RowInvarianceTests` (Vulkan): single-row steps against 2-, 3- and 4-row steps, bit for bit, for f32, f16, bf16, q8_0,
q4_k, q6_k and block FP8 weights, single and as gate / up pairs. The K-quants rely on the driver contracting their
int-dot variants alike; it does at these shapes but not at all (q4_k [2048→48] differs in the last bit), so verification
on K-quant weights is not guaranteed bit-exact. `VerificationRowInvarianceTests` (Families): verify-step rows against
single-row session steps over a prompt and 1-row / 4-row decode, bit for bit, on Qwen3-0.6B Q8_0 and (large) the 27B
Q8_0 GGUF and FP8 checkpoint.

MTP throughput with row-invariant verification (`SpecBench`, depth 47, tg tok/s, k = 0 / 1 / 3, two interleaved
rounds; acceptance identical between builds; `spec/`):

| build | round 1 | round 2 |
|---|---|---|
| FP8 base | 39.7 / 66.6 / 90.5 | 39.2 / 67.6 / 89.0 |
| FP8 new | 39.8 / 67.8 / 86.1 | 38.8 / 66.5 / 82.0 |
| Q8_0 base | 40.2 / 74.7 / 97.7 | 39.3 / 78.5 / 103.0 |
| Q8_0 new | 37.3 / 77.2 / 95.1 | 39.8 / 76.0 / 98.1 |

Verification on the single-row arithmetic costs about 5-8 % at k = 3 on FP8 (the 4-row step no longer takes the skinny
GEMM) and is within noise on Q8_0. k = 1 is unchanged.

### 28.2 MoE: the two expert-tile launches without a barrier

Section 24 split the expert GEMM into a full-height launch and a half-height launch for an expert's short last tile. Both
write disjoint rows of the same output, but the hazard tracker ordered them with a barrier. `KernelLaunch.WriteGroup` lets
launches declare disjoint writes; `HazardTracker` does not order writes within one group (reads still order against them).
gpt-oss-20b pp512 (CLI `--bench 512`, interleaved, best of 4 inside each run): base 13271 / 13951 / 13721, new 14417 /
15689 / 15197 tok/s (+10 %). With `SHARPLIGHT_NORM_HALF_COPY=0` the new build gives 15909 / 15474 / 15649, so the gain is
the barrier. Decode is unchanged (316-331 tok/s in both).

### 28.3 Fewer glue launches: the norm writes the f16 copy; tall dual fp8 tiles

* `GraphLowerer.HalfCopies.cs`, `RowNormKernelSource.cs`: a row norm whose output feeds matmuls over non-f32 weights also
  stores the f16 copy the tensor-core GEMMs read, for steps in the skinny range (2-16 rows); larger steps convert into the
  same buffer. The stored value is `f32tof16` of the f32 output, so it is bit-identical to the conversion kernel.
  `SHARPLIGHT_NORM_HALF_COPY=0` disables it. 27B FP8 at 16 rows: 1334 → 1158 dispatches and 917 → 789 barriers per step.
  The GPU time did not change measurably on the Blackwell (three interleaved rounds with the switch: 27.2 / 28.2 / 30.0 ms
  off, 27.8 / 28.4 / 27.5 ms on). The separate conversions were 4.5 % of the profiled step, but profiling serializes
  launches; unprofiled, they overlap with their neighbours. Letting the norm write the copy at every row count made the
  171-row mixed step 1.3 ms slower, hence the 16-row cap.
* `GraphLowerer.Fp8Gemm.cs`: the dual fp8 gate / up GEMM uses 128-row tiles above 256 rows where they pad no more rows
  than 64-row tiles (`SHARPLIGHT_FP8_DUAL_TALL=0` keeps 64). 512-row prefill step on the 27B FP8: 139.4 / 141.0 → 134.1 /
  135.4 ms (-3 to -4 %). 384 rows: 110.4 / 111.8 → 106.9 / 107.9 ms. At 128 and 171 rows the tall tiles were slower
  (44.7 → 49.9 and 71.0 → 80.0 ms), because the grid has half the workgroups.
* Tried and dropped: the skinny GEMM up to 64 rows (`SHARPLIGHT_SKINNY_MAX_ROWS=64`): 32 rows 32.3 → 39.1 ms, 64 rows 42.4
  → 60.1 ms.

### 28.4 Where the serving steps go

The serving gap to vLLM at 16-64 streams is the cost of mixed steps (decode rows of every stream plus prompt chunks of
new requests), not host overhead. Server metrics of section 24.5: c = 1 averages 25.2 ms per step at 2.9 rows (BatchBench
1-row step: 23-24 ms), c = 16 41 ms at 46 rows, c = 64 80 ms at 172 rows. BatchBench `--mixed` reproduces such steps
(27B FP8, minimum GPU time, `batch/mixed_prof_new.txt`):

| step | GPU ms |
|---|---:|
| 16 decode rows | 29.3 |
| 16 decode + 32 prompt rows | 32.5 |
| 63 decode rows | 42.2 |
| 63 decode + 108 prompt rows | 77.4 |

Profile of the 171-row mixed step (10 steps):

* GEMMs 70 %: dual gate/up 290.8 µs per layer (64-row tiles, 3 row tiles; about 210 TFLOPS), [17408→5120] 163.7 µs,
  [5120→10240] 98.2 µs; the bf16 [5120→48] split-K pair 2.3 %.
* Gated delta rule 20 %: the token-by-token kernel for the 63 decode rows 234.5 µs per layer, memory-bound on the f32
  state (3 MB per sequence and layer, read and written). The opt-in bf16 state halves that traffic; vLLM keeps this
  state in the model dtype (bf16).
* Glue (activation quantization, split-K reductions, norms) about 5 %.

So the remaining levers are the fp8 GEMM's efficiency at 100-500 rows (section 22.10: about 280 of about 520 TFLOPS) and
the state precision. In this round, skinny-GEMM and tile-height changes did not help in the 32-171 row range (28.3).

### 28.5 Single stream: 27B Q8_0 decode and Mistral-7B prefill

27B Q8_0, one-row decode profile (`batch/prof_q8_1row.txt`, minimum per launch; weights 1.0625 bytes per value):

| matvec | µs | TB/s |
|---|---:|---:|
| gate/up [5120→17408]x2 (dual) | 114.2 | 1.66 |
| down [17408→5120] | 67.0 | 1.41 |
| [5120→10240] | 39.1 | 1.42 |
| [6144→5120] | 24.7 | 1.35 |
| LM head [5120→248320] | 1010.8 | 1.34 |

The dual matvec is close to the bandwidth limit, while the single ones run at 1.34-1.42 TB/s; norms and the 96 small
[5120→48] matvecs add about 0.8 ms per token. llama.cpp's 48.0 tok/s (20.8 ms per token) corresponds to about 1.45 TB/s
over the whole model. Closing the gap needs a different K split or work per workgroup in the single matvecs, which changes
the summation order and therefore greedy output. Under this round's output-identity rule it was not changed. The same
holds for Mistral-7B Q4_K_M pp512 (0.91x), whose GEMM tiling sets the accumulation order. Both are listed as open issues.

### 28.6 Serving against vLLM

`bench/s16_serve.sh` as in section 24.5 (~250-token prompts, 128 output tokens, ignore_eos, a 64-slot server), 27B FP8,
port 18377. Rounds 1-2: Sharplight, then vLLM 0.33 (`--gpu-memory-utilization 0.5`), then Sharplight with the opt-in
bf16 delta-rule state (`SHARPLIGHT_RECURRENT_STATE=bf16`). A first attempt ran while another application loaded the GPU (c = 1 at
22 tok/s) and was discarded. A third round (vLLM, then Sharplight, `serve/round3/`) was disturbed for both engines
(vLLM c = 1 at 38.9 instead of 47) and is not used in the ratios.

Aggregate output tok/s (rounds 1, 2), TTFT P50 / P95 and mean ITL in ms (round 1):

| conc | vLLM | Sharplight | SL bf16 state | SL best / vLLM best | bf16 / vLLM | TTFT SL | vLLM | ITL SL | vLLM |
|---|---|---|---|---:|---:|---|---|---:|---:|
| 1 | 47, 47 | 38, 36 | 39, 37 | 0.81x | 0.83x | 82 / 86 | 61 / 102 | 25.8 | 20.8 |
| 4 | 165, 165 | 135, 133 | 138, 141 | 0.82x | 0.85x | 231 / 275 | 183 / 185 | 28.2 | 23.1 |
| 16 | 552, 548 | 380, 383 | 382, 393 | 0.69x | 0.71x | 293 / 996 | 595 / 597 | 39.1 | 24.7 |
| 64 | 980, 1004 | 769, 748 | 799, 804 | 0.77x | 0.80x | 491 / 4532 | 1733 / 7258 | 72.4 | 40.0 |

Sharplight is at its section 24.5 level (40 / 130 / 381 / 752 then, 38 / 135 / 383 / 769 now). vLLM measured faster at
1-16 streams (43 / 153 / 522 then) and slower at 64 (1157 then), so the ratios moved mostly with vLLM's run-to-run spread on
the shared GPU. The bf16 state adds 2-5 % at 16-64 streams. Single-stream ITL is 25.8 ms against a 22.5-23.5 ms
BatchBench step; the server's step average includes the prompt steps of each new request (section 28.4).

### 28.7 Correctness

Greedy CLI output, 200 tokens, a short and a ~650-token prompt (`ident/p_*.txt`), the branch rebased on `main` at
`44622a2` against that `main` (`ident/`; an earlier run against `66a55a8` gave the same result):

* Identical for the 27B FP8, Q8_0 and Q4_K_M, gpt-oss-20b, OLMoE, Mistral-7B Q4_K_M and Qwen2.5-0.5B.
* MTP k = 3 against non-speculative output on the 27B FP8 and Q8_0, short, long and the 832-token prompt that diverged
  before (`ident/p_full.txt`): identical text. On `main` the Q8_0 output of the 832-token prompt diverged.

Tests, run one at a time under the suite lock on the branch rebased on `44622a2`: Cpu 1214 (36 skipped), Vulkan 1277
(22 skipped), Families 663 (34 skipped), Serving 283 and Vision 94 (5 skipped) pass. (On an earlier rebase, Families was
OOM-killed once while other applications and jobs held memory, and passed when re-run.) New: `RowInvarianceTests` and `NormHalfCopyTests` (Vulkan), `VerificationRowInvarianceTests`
(Families, two large cases with `SHARPLIGHT_TEST_LARGE=1`).

### 28.8 Commands

```bash
# Identity (base = git archive of main, built separately)
sharplight <model> -p "$(cat p_long.txt)" -n 200 --ctx 2048 [--speculate-mtp 3]
# MTP throughput
bench/SpecBench/bin/Release/net11.0/SpecBench <model> --drafts 0,1,3
# Batched and mixed steps, profiles
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 1,16,64 --reps 30
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --mixed 63+0,63+108x1,16+0,16+112x1,0+512x1 --profile 1
# gpt-oss prefill
sharplight gpt-oss-20b-MXFP4.gguf --bench 512 --ctx 768
# Serving
OUT_DIR=bench/results/s28/serve PORT=18377 SLOTS=64 bench/s16_serve.sh sharplight qwen27b-fp8 <dir> "1 4 16 64"
OUT_DIR=bench/results/s28/serve PORT=18377 SLOTS=64 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
```

### 28.9 Open issues

* **Serving at 16-64 streams** (0.69-0.80x of vLLM): mixed steps are dominated by the fp8 GEMM at 100-500 rows (about
  210 TFLOPS on the dual gate/up at 171 rows) and the f32 delta-rule state (28.4). Next: a more efficient fp8 GEMM pipeline
  and, if accepted as a default, the bf16 state.
* **MTP verification cost**: k = 3 on FP8 is 5-8 % slower because the 4-row verification steps use the single-row matvec.
  A row-invariant skinny GEMM (the same K order for 1-4 rows) would recover it.
* **K-quant verification** is bit-exact only where the driver contracts the int-dot variants alike (28.1).
* **27B Q8_0 decode and Mistral-7B pp512** (0.90x / 0.91x of llama.cpp): reaching llama.cpp's bandwidth needs a different
  summation order in the single matvecs and the K-quant GEMM, which changes greedy output (28.5).
* **Norm-written f16 copies** cut launches and barriers but not Blackwell GPU time; their value on GPUs with costlier
  dispatch is unmeasured. Elementwise and matmul epilogues (attention gate, SwiGLU output) still feed separate conversions.

## 29. Serving steps are power-limited; narrow fp8 tiles and deeper skinny prefetch tried and dropped (2026-11-14)

Follow-up to section 28 (LLM performance, serving against vLLM). Base is `main` at `ba21180`, built side by side; base
and new runs alternate. The GPU was shared with the user's applications and with
other agents' test suites; runs disturbed by them are kept in the raw results but not used. Raw results:
[`bench/results/s29/`](../bench/results/s29). No kernel change of this round made the end-to-end steps faster, so the
branch ends with the source tree of `main`; this section records what was measured.

### 29.1 Narrow fp8 GEMM tiles below 64 rows (reverted)

The fp8 GEMM's regular 64-row tile (64 x 128 outputs, 8 subgroups) uses 156-158 registers per thread, so one workgroup
runs per SM, and the 17-64-row batches of the 27B FP8 give grids of 160-384 workgroups (split-K included) on the RTX PRO
6000's 188 SMs: the last wave is often half empty. A 64 x 64 tile on 4 subgroups (`2,2,2,2`, 164-168 registers, 19 KB
shared memory) fits three workgroups per SM and doubles the grid. The experiment (commit `5c71bb5c`, reverted on the
branch) chose the narrow tiles when the regular grid filled less than 80 % of its waves, with the compute-unit count
read from `VK_NV_shader_sm_builtins` / `VK_AMD_shader_core_properties` / `VK_ARM_shader_core_builtins`. The K split
stayed the regular one, so outputs were bit-identical (greedy 27B FP8 output identical to `main` on the short, a
37-token and the ~650-token prompt, `ident/`), and all suites passed with it.

Kernel times with the rule against the regular tiles (`micro/narrow_auto_summary.txt`, minimum of two interleaved rounds
of 100 back-to-back launches, µs):

| shape | regular grid | 24 rows | 40 rows | 64 rows |
|---|---:|---|---|---|
| [5120→17408]x2 dual | 272 | 133 → 127 | 137 → 127 | 139 → 129 |
| [5120→17408] | 272 | 75 → 69 | 78 → 73 | 80 → 76 |
| [5120→12288] | 384 | 63 → 57 | 65 → 61 | 68 → 66 |
| [5120→6144] | 192 | 30 → 26 | 37 → 28 | 41 → 35 |
| [17408→5120], [5120→10240], [6144→5120] | 160-320 | regular kept | | |

Forcing the narrow tiles everywhere (`micro/narrow_forced_ab.txt`) made the 320-workgroup grids 8-22 % slower. In the
profiled 16 + 32-row mixed step the dual GEMM went from 143.4 to 129.0 µs per layer and the step from 34.2 to 33.7 ms
GPU time (launches serialized; `batch/prof_m48_*.txt`). Unprofiled, the steps did not get faster: same binary,
`SHARPLIGHT_FP8_NARROW=0` against the rule, minimum GPU ms over three interleaved rounds of 20 steps
(`batch/narrow_switch_*.txt`; the dual-only variant `batch/narrow_dual_*.txt` behaved the same; base against narrow
build: `batch/mixed_*.txt`):

| step (decode + prompt rows) | regular | narrow (rule) | dual only: regular | dual only: narrow |
|---|---:|---:|---:|---:|
| 16 + 16 | 29.25 | 28.49 | 28.29 | 27.97 |
| 16 + 32 | 31.05 | 31.77 | 30.08 | 29.70 |
| 16 + 48 | 34.24 | 34.32 | 33.47 | 33.92 |
| 32 + 32 | 36.60 | 37.49 | 36.17 | 36.60 |
| 40 + 0 | 35.87 | 35.86 | 35.17 | 35.40 |
| 64 + 0 | 42.55 | 43.44 | 42.07 | 42.60 |

The differences (-0.8 to +0.9 ms) are within the round-to-round spread and lean towards slower at 48-64 rows. 29.2 gives
the likely reason.

### 29.2 Sustained steps run at the power limit

The card is the 300 W Max-Q edition. During back-to-back 32 + 32 and 64-row steps (`nvidia-smi` every 500 ms,
`batch/power_32+32_64.csv`) it draws 300 W with the software power cap as the active clock-limit reason, and the SM
clock is 1600-1770 MHz (3090 MHz maximum; about 2.3 GHz when idle, where short micro-benchmark bursts start). So the
serving steps are limited by energy, not by idle SMs: a tile that fills the last wave but re-reads the activations twice
as often and stages more shared memory most likely just runs the same work at a lower clock. vLLM runs under the same
cap. Wave-fill and occupancy tuning measured with micro-benchmarks therefore overstates end-to-end gains on this card;
the levers that remain are those that move fewer bytes (smaller state, fewer intermediate copies) or do fewer
operations.

### 29.3 Other experiments (dropped)

* Two K blocks in flight per subgroup in the direct skinny GEMM (2-16 rows, same arithmetic;
  `micro/skinny_depth2_dropped.txt`): 3-22 % slower at 4 and 16 rows, the extra registers cost occupancy.
* Narrow tiles above 64 rows against the 128-row tiles (`micro/narrow_above64_summary.txt`): 4-28 % slower at 80-256
  rows except [5120→6144] (8-16 % faster, under 1 % of a mixed step).

### 29.4 The delta-rule state is at the memory bandwidth

The token-by-token `delta_rule` kernel reads and writes each sequence's f32 state once per step: 48 value heads x 128 x
128 x 4 bytes = 3.1 MB per layer and sequence. At 48 decode rows it takes 161 µs per layer, 302 MB in 161 µs = 1.9 TB/s,
at or above the card's 1.79 TB/s nominal bandwidth (part of the state stays in the 128 MB L2 between the bench's
repeated steps); section 28's 63-row profile (234.5 µs) gives 1.7 TB/s. A bit-identical change of layout, fusion or
occupancy cannot make it faster; only a smaller state (the opt-in bf16 state, section 24.6) can. At 16 rows the kernel
is 6 % of the step (36 µs per layer); the 20 % of section 28.4 is the 63-row case.

### 29.5 Where the 16-row decode step goes

27B FP8, 16 decode rows, minimum per launch (`batch/prof16dec.txt`; 29.0 ms GPU per step, against vLLM's 24.7-25.8 ms
mean ITL at 16 streams in sections 28.6 and 29.6):

| launch | µs | per step | weight TB/s |
|---|---:|---:|---:|
| gate/up [5120→17408]x2 skinny (dual) | 129.0 | 64x | 1.42 |
| down [17408→5120] skinny | 73.2 | 64x | 1.26 |
| [5120→10240] skinny | 45.5 | 48x | 1.19 |
| [6144→5120] skinny | 26.0 | 64x | 1.22 |
| [5120→6144] skinny | 34.4 | 48x | 0.94 |
| LM head [5120→248320] bf16 skinny | 1602.5 | 1x | 1.59 |
| delta rule | 36.2 | 48x | |
| [5120→48] bf16 split-K (+ reduce) | 7.7 (+2.0) | 96x | |

The skinny GEMMs read the weights (with their block scales) at 0.9-1.4 TB/s, the LM head at 1.6 TB/s. The small
[5120→48] matmuls and their reductions are about 1 ms per step in the serialized profile.

### 29.6 Serving against vLLM

`bench/s16_serve.sh` as in section 28.6 (~250-token prompts, 128 output tokens, ignore_eos, a 64-slot server), 27B FP8,
port 18377, one round: Sharplight (the source tree of `main`), then vLLM 0.33 (`--gpu-memory-utilization 0.5`). Another
agent's GPU tests ran during both engines' c = 4-64 runs (the snapshots before those runs show 98-100 % utilization at
300 W), so the ratios carry that disturbance.

| conc | vLLM | Sharplight | SL / vLLM | TTFT P50 / P95 SL | vLLM | mean ITL SL | vLLM |
|---|---:|---:|---:|---|---|---:|---:|
| 1 | 45.5 | 39.0 | 0.86x | 80 / 111 | 56 / 63 | 25.2 | 21.7 |
| 4 | 158.5 | 140.8 | 0.89x | 214 / 229 | 178 / 183 | 27.1 | 24.1 |
| 16 | 533.1 | 393.3 | 0.74x | 286 / 943 | 581 / 594 | 37.7 | 25.8 |
| 64 | 1108.7 | 769.7 | 0.69x | 499 / 4360 | 1715 / 2332 | 72.4 | 42.6 |

Sharplight is at its section 28.6 level (38 / 135 / 383 / 769 tok/s then); vLLM measured 45-47 / 158-165 / 533-552 /
980-1109 over sections 28 and 29, so the ratio at 64 streams moves between 0.69x and 0.78x with vLLM's spread.

### 29.7 Correctness

* Greedy CLI output of the 27B FP8, 200 tokens, the short and ~650-token prompts of section 28 and a 37-token prompt
  (`ident/p_mid.txt`, its prefill runs the 17-64-row GEMMs the narrow tiles served): identical to `main` with the narrow
  tiles. Q8_0, K-quant and MXFP4 models never took the changed path (fp8 GEMM only).
* Tests with the narrow tiles, one at a time under the suite lock: Package 32, Cpu 1214 (36 skipped), Vulkan 1277 (22
  skipped), Families 663 (34 skipped) and Serving 283 pass. After the revert the source tree equals `main`.

### 29.8 Commands

```bash
# Kernel A/B (GPU timestamps; SHARPLIGHT_FP8_NARROW=0|1 existed only in commit 5c71bb5c)
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_TYPES=f8e4m3_b128 SHARPLIGHT_BENCH_ROWS=48 SHARPLIGHT_BENCH_COPIES=4 SHARPLIGHT_BENCH_ITERS=100 \
  SHARPLIGHT_BENCH_DUAL=1 SHARPLIGHT_BENCH_SHAPES="5120,17408" SHARPLIGHT_BENCH_REPORT=1 \
  dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll -method '*MatMulPrefill'
# Mixed steps (and --profile <index of the --mixed entry>)
bench/BatchBench/bin/Release/net11.0/BatchBench <model> --ctx 2048 --modes greedy --sizes 16 \
  --mixed 16+16x1,16+32x1,16+48x1,32+32x1,40+0,64+0 --reps 20
# Power and clocks during a run
nvidia-smi --query-gpu=timestamp,power.draw,power.limit,clocks.sm,clocks.mem,utilization.gpu,clocks_event_reasons.active \
  --format=csv,noheader -lms 500
# Serving
OUT_DIR=bench/results/s29/serve PORT=18377 SLOTS=64 ROUNDS=1 bench/s16_serve.sh sharplight qwen27b-fp8 <dir> "1 4 16 64"
OUT_DIR=bench/results/s29/serve PORT=18377 SLOTS=64 ROUNDS=1 bench/s16_serve.sh vllm qwen27b-fp8 <dir> "1 4 16 64" --gpu-memory-utilization 0.5
```

### 29.9 Open issues

* **Serving at 16-64 streams** (0.69-0.78x of vLLM): unchanged. The mixed steps are fp8 GEMM at 100-500 rows and the
  delta rule; the delta rule is at the bandwidth (29.4), and on this power-capped card a faster fp8 GEMM has to do less
  work or move fewer bytes per output, not only fill the GPU better (29.2).
* **16-row decode** (29.0 ms GPU against vLLM's 24.7-25.8 ms ITL): the skinny GEMMs read weights at 0.9-1.4 TB/s against
  the LM head's 1.6 TB/s (29.5); a deeper register prefetch did not help (29.3).
* **Small [5120→48] matmuls**: about 1 ms per 16-row step serialized; folding the two per layer into one launch needs a
  two-output variant of the split-K GEMM with separate epilogues.
* The open issues of section 28.9 (MTP verification cost, K-quant verification, single-stream Q8_0 decode and Mistral-7B
  pp512, norm-written copies) are unchanged.

## 30. ONNX parity round 5: f32-accurate split-f16 GEMMs and convs as the strict default, CPU channels-last convs, device-side detection counts (2026-11-14)

Same machine and harness as section 27 (`bench/onnx/parity_bench.py`, interleaved rounds, ORT 1.31 in the same
session). Raw data is in `bench/results/s30/`. The benchmarks ran on the branch merged with section 28's `main`. They
predate the later merges of `main` (section 29, model bring-up) and the two split-weight fixes of 30.4, which change
host memory and sharing, not the kernels. The GPU and CPU were shared with the user's applications and
other agents throughout.

### 30.1 What changed

* **Split-f16 tensor-core GEMM at f32 accuracy, now the strict default** for f32 MatMuls with constant weights and K a
  multiple of 32 on devices with f16 matrices (`CompileOptions.SplitHalfMatMul`, default true).
  * Each row of x and W is scaled by a power of two that puts its largest magnitude in [2^14, 2^15). Both f16 parts
    (hi and lo) then stay normal, which removes the subnormal lo parts that kept the section 26 form at 2-7x the f32
    error. x is split by a `split_rows` pre-pass, constant weights on the host.
  * The hi*hi + hi*lo + lo*hi products of each 32-value K step are summed from zero on the tensor cores and added to
    f32 accumulators, so the tensor cores' truncating accumulation no longer builds up over K.
  * Against an f64 reference (`SplitHalfGemmTests`, `split-f16-accuracy.txt`) the relative RMS is 1.5-1.8e-7, against
    1.2-5.0e-7 for the f32 SIMT GEMM and 2.6e-4 for plain f16. It holds with weights near 1e-7 or 1e3 and activations
    near 1e6 or 1e-6, where plain f16 overflows or loses its digits.
  * Split-f16 matmuls whose rows always exceed the matvec range drop the f32 copy of their weights. vit-base uploads
    0.35 GB instead of 0.69 GB.
  * `SHARPLIGHT_GEMM_SPLIT_F16=0`, `SplitHalfMatMul = false` or `OnnxBench --no-split-f16` keep the f32 SIMT GEMM.
    Devices without the `MatrixF16` feature use the SIMT GEMM.
* **Split-f16 strict convs** (`conv_cm` with 4 parts) instead of six bf16 products.
  * Weights are split on the host with per-output-channel scales.
  * The im2col values of each 16-pixel block and 32-value K step are scaled by a power of two (the block maximum over
    one subgroup's lanes) before the split. Each step's products are added to f32 accumulators, times the block's
    inverse scale.
  * Two subgroup rows (128 output channels) with 128-pixel tiles: segformer's 1024 → 256 1x1 conv at 128x128 went from
    321 to 170 µs.
  * resnet50 b8 3.06 → 2.24 ms GPU time, worst RMS 1.5e-6 → 9.6e-7. `SHARPLIGHT_CONV_SPLIT_F16=0` keeps the bf16 form.
* **Vulkan attention over strided K / V**: the encoder flash kernel and the scalar kernel address K / V rows through
  per-row and per-head strides, so the slices of a fused Q / K / V projection are read in place. The two dense copies
  per layer from section 27 are gone. `SHARPLIGHT_ATTN_STRIDED_KV=0` restores them.
* **Gather with many scattered constant indices** becomes one GatherAxis instead of a slice per index. raft-large had
  about 6800 one-element slices and concats per run (29 % of its GPU time).
* **CPU channels-last layout** (device feature `channels_last`, on with AVX2; `SHARPLIGHT_DISABLE_FEATURES=channels_last`
  or `CompileOptions.DisabledFeatures` turn it off).
  * Rank-4 f32 activations stay channels-last (NHWC strides over NCHW dims) across the ops that take the layout.
    Every other op sees an ordinary strided view and materializes it, so transforms only happen at region borders. A
    conv picks the layout when its input is already channels-last or all its readers take it.
  * `ChannelsLastConvOp`: a register-blocked implicit GEMM without im2col or activation packing (AVX-512 8x32, 12x16
    for up to 16 output channels, AVX2 6x16). Stems with fewer than 16 input channels run kernel-row spans over a
    padded copy. K chunks stay in L1, tasks are planned by a time model, and bias / residual / Relu / Clip are applied
    in registers.
  * Depthwise convs, windowed and global pooling, Resize, Concat and channel splits over channels-last views;
    elementwise ops with per-channel broadcasts.
* **Detection: data-dependent counts grouped by level.**
  * Nodes are ordered by count level, so the independent counts of a detection head (per-class NonZero /
    NonMaxSuppression) end one segment per dependency level instead of one each. fasterrcnn-12 went from 174 to 5
    segments.
  * Consecutive segments on one Vulkan device copy their boundary values device to device
    (`IDeviceForwardingSegment`) instead of through host memory.
  * `PlacementOptions.GroupDataDependentCounts`, `SHARPLIGHT_COUNT_LEVELS=0` and `SHARPLIGHT_DEVICE_FORWARDING=0`
    restore the old paths. `OnnxBench --vary-input` feeds a new input each run.
* **Vulkan kernels instead of the CPU fallback**: LinearAttention (all four update rules, one thread per state column),
  DeformConv, Unique (with a device-written count), TfIdfVectorizer, the ai.onnx.ml tree ensembles, SVM one-vs-one, numeric LabelEncoder, and a direct DFT for lengths past the basis matmul.
* **JIT for package consumers**: the CPU kernels and loop methods carry `MethodImplOptions.AggressiveOptimization`.
  They are compiled fully optimized on first use whatever the application's JIT settings, so NuGet consumers no
  longer depend on this repository's `TieredCompilationQuickJitForLoops=false`.
  * Before, an application with the .NET defaults ran resnet50's first five CPU runs at about 48 ms instead of 10 ms.
  * With `DOTNET_TC_QuickJitForLoops=1` forced on, resnet50's first five runs now take 7.0-7.5 ms, the same as with
    it off (6.9-8.7 ms).
  * `OnnxBench` records `firstRunMs` and `earlyRunsMs`.
* Not done: Winograd F(4x4, 3x3) on the channels-last layout, split f16 for batched matmuls, and SIMT GEMM tuning for
  devices without f16 matrices.

### 30.2 CPU: Sharplight vs onnxruntime CPU EP (16 threads)

Cells are min / median in ms; §27 is the `sl-cpu` min of section 27. Other work kept the 1-minute load average at
8-28 in both CPU sessions (`prev-run/cpu-load.log`, `cpu-load.log`). The table takes the minimum over both sessions
(4 interleaved rounds per engine, same CPU code; `table-cpu-both.md`). The interleaved SL / ORT ratio is the
comparable number.

| model | batch | §27 | sl-cpu | ort-cpu | SL / ORT (§27) | SL / ORT |
|---|---|---|---|---|---|---|
| resnet50 | 1 | 5.87 | 4.61 / 6.31 | 6.26 | 0.95x | **0.74x** |
| resnet50 | 8 | 34.22 | 23.13 / 33.78 | 25.91 | 1.54x | **0.89x** |
| convnext-tiny | 1 | 14.02 | 13.23 / 23.59 | 14.05 | 1.03x | **0.94x** |
| convnext-tiny | 8 | 81.50 | 76.23 / 121.69 | 85.38 | 0.97x | **0.89x** |
| efficientnet-lite4 | 1 | 6.92 | 4.19 / 7.49 | 3.94 | 1.59x | 1.06x |
| mobilenetv2 | 1 | 1.98 | 0.98 / 1.97 | 1.69 | 1.23x | **0.58x** |
| mobilenetv2 | 8 | 10.78 | 6.96 / 16.90 | 6.69 | 1.53x | 1.04x |
| yolov8n | 1 | 9.76 | 7.79 / 13.06 | 13.51 | 0.70x | **0.58x** |
| yolov10n | 1 | 12.07 | 9.32 / 19.45 | 16.91 | 0.76x | **0.55x** |
| bert-base (16 tok) | 1 | 9.61 | 9.82 / 14.98 | 12.53 | 0.79x | **0.78x** |
| bert-base (16 tok) | 8 | 20.50 | 20.41 / 31.52 | 30.18 | 0.74x | **0.68x** |
| vit-base | 1 | 19.77 | 21.31 / 32.04 | 24.12 | 0.86x | **0.88x** |
| vit-base | 8 | 132.85 | 140.08 / 196.57 | 145.55 | 0.95x | **0.96x** |
| whisper-tiny-encoder | 1 | 21.99 | 21.49 / 48.54 | 43.02 | 0.51x | **0.50x** |
| segformer-b0 | 1 | 38.73 | 23.27 / 51.80 | 45.36 | 0.92x | **0.51x** |
| segformer-b0 | 8 | 249.47 | 236.00 / 309.20 | 569.12 | 0.52x | **0.41x** |
| mobilesam-encoder | 1 | 143.35 | 130.88 / 190.81 | 222.35 | 0.65x | **0.59x** |

* The conv models gained from the channels-last layout: resnet50 b8 1.54x → 0.89x, mobilenetv2 b1 1.23x → 0.58x,
  efficientnet-lite4 1.59x → 1.06x, segformer b1 0.92x → 0.51x.
* mobilenetv2 b8 (1.04x) and efficientnet-lite4 (1.06x) are the CPU gaps left.
* The transformer models (bert, vit, whisper) did not change. Their differences to section 27 are within the load
  noise.
* `prev-run/` also holds the first GPU session, run before the last conv-tile change. The GPU tables below are from
  the second session.

### 30.3 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

Cells are min / median in ms (wall). SL / ORT is `sl-vulkan` (strict, f32-accurate) over `ort-cuda` (TF32). `(!)` marks a
parity-check failure (expected for reduced precision, and for ssd-10 and phi-3.5 in every precision, as before).
§27 is section 27's strict `sl-vulkan` min. The bert-base b8 row is a 3-round re-run (`gpu-rerun.log`); the first
session's wall times for it were disturbed (6.2 ms against 2.89 ms GPU time). The full table is in `table-gpu.md`.

| model | batch | §27 | sl-vulkan | sl-vulkan-simt | sl-vulkan-rp | ort-cuda | ort-cuda-strict | SL / ORT (§27) | SL / ORT |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.82 | 0.77 / 1.19 | 0.84 | 0.70 (!) | 1.02 | 2.12 | 0.81x | **0.75x** |
| resnet50 | 8 | 3.09 | 2.28 / 2.37 | 3.10 | 1.93 (!) | 2.01 | 3.73 | 1.55x | 1.13x |
| convnext-tiny | 8 | 4.11 | 3.65 / 4.54 | 4.04 | 2.66 (!) | 2.56 | 4.15 | 1.61x | 1.42x |
| efficientnet-lite4 | 1 | 0.84 | 0.75 / 0.89 | 0.84 | 0.75 | 0.69 | 8.03 | - | 1.09x |
| mobilenetv2 | 8 | 0.73 | 0.69 / 1.00 | 0.73 | 0.68 (!) | 0.75 | 1.36 | 0.97x | **0.92x** |
| yolov10n | 1 | 1.26 | 1.19 / 1.24 | 1.27 | 1.25 (!) | 1.21 | 2.42 | - | **0.98x** |
| ssd-10 (NMS) | 1 | 17.65 | 12.38 / 15.51 (!) | 16.63 (!) | 8.16 (!) | 11.20 | 22.56 | - | 1.11x |
| bert-base (16 tok) | 1 | 1.22 | 1.27 / 1.63 | 1.21 | 1.17 (!) | 1.40 | 1.50 | 0.99x | **0.91x** |
| bert-base (16 tok) | 8 | 3.55 | 3.46 / 3.73 | 3.50 | - | 2.72 | 3.31 | 1.30x | 1.27x |
| vit-base | 1 | 2.47 | 2.36 / 2.67 | 2.59 | 2.10 (!) | 1.38 | 2.60 | 1.84x | 1.71x |
| vit-base | 8 | 9.77 | 8.06 / 9.17 | 9.77 | 4.81 (!) | 3.97 | 9.27 | 2.38x | 2.03x |
| whisper-tiny-encoder | 1 | 1.84 | 1.78 / 1.94 | 1.85 | 1.33 (!) | 1.19 | 2.07 | 1.59x | 1.49x |
| segformer-b0 | 8 | 21.12 | 20.98 / 29.95 | 20.93 | 18.57 | 19.52 | 22.89 | 0.50x | 1.07x |
| mobilesam-encoder | 1 | 5.89 | 5.26 / 5.82 | 5.92 | 4.40 (!) | 6.14 | 8.16 | 0.97x | **0.86x** |
| raft-large | 1 | 56.17 | 37.82 / 49.74 | 48.19 | 34.88 (!) | 29.35 | 72.99 | 1.91x | 1.29x |
| fasterrcnn-12 (counts) | 1 | - | 9.68 / 11.79 | 10.30 | 7.50 | 7.91 | 13.19 | - | 1.22x |
| maskrcnn-12 (counts) | 1 | - | 9.48 / 11.27 | 11.62 | 9.04 | 7.56 | 13.74 | - | 1.25x |

GPU-timestamp minima (`table-gpu-timestamps.md`, unaffected by host load), in ms:

| model | batch | §27 strict | §27 split (opt-in, less accurate) | strict default now | SIMT now | worst RMS default / SIMT |
|---|---|---|---|---|---|---|
| resnet50 | 8 | 3.06 | 3.06 | 2.24 | 3.05 | 9.6e-07 / 1.5e-06 |
| convnext-tiny | 8 | 4.04 | 3.17 | 3.60 | 3.98 | 3.1e-06 / 3.3e-06 |
| efficientnet-lite4 | 1 | 0.81 | 0.81 | 0.72 | 0.81 | 1.2e-06 / 1.2e-06 |
| ssd-10 (NMS) | 1 | 16.84 | 16.83 | 11.96 | 16.40 | (parity fails in every mode) |
| vit-base | 1 | 2.43 | 2.13 | 2.32 | 2.42 | 1.7e-06 / 1.7e-06 |
| vit-base | 8 | 9.64 | 7.01 | 7.73 | 9.64 | 1.7e-06 / 1.6e-06 |
| whisper-tiny-encoder | 1 | 1.75 | 1.50 | 1.65 | 1.75 | 3.0e-06 / 3.1e-06 |
| mobilesam-encoder | 1 | 5.64 | 5.02 | 5.07 | 5.66 | 1.3e-06 / 1.3e-06 |
| raft-large | 1 | 53.31 | 53.41 | 34.78 | 45.10 | 2.0e-02 / 3.5e-02 |
| fasterrcnn-12 (counts) | 1 | - | - | 8.35 | 9.84 | 1.9e-07 / 1.9e-07 |

* The f32-accurate split GEMM is the strict default now, and its worst model RMS is at or below the SIMT GEMM's on
  every model. It is slower than section 27's less accurate opt-in form (vit b8 7.73 against 7.01 ms) because of the
  row-scaling pre-pass and the per-step promotion.
* The split convs account for resnet50 b8 (3.06 → 2.24 ms) and for 9-11 % on efficientnet, mobilenetv2 and yolov8n.
* raft-large 53.3 → 34.8 ms comes from the Gather rewrite (its ~6800 slices were 29 % of the GPU time) and the split
  convs and GEMMs. Its SIMT run also gains from the Gather rewrite (45.1 ms).
* The transformer encoders stay at 1.5-2.0x of TF32 ORT. Their split GEMMs are bound by staging tiles and the
  activation split pre-pass.
* segformer b8: ORT's CUDA EP ran 19.5 ms in this session against 42.6 ms in section 27. Sharplight is unchanged
  (21 ms), so the ratio moved from 0.50x to 1.07x.

Detection A/B (`detection-ab.log`, 3 interleaved rounds, min wall / GPU ms; "old" = `SHARPLIGHT_COUNT_LEVELS=0
SHARPLIGHT_DEVICE_FORWARDING=0`, "levels" = grouped counts with host copies):

| model | segments old → new | wall old | wall levels | wall new | GPU old → new |
|---|---|---|---|---|---|
| fasterrcnn-12 | 174 → 5 | 118.6 | 20.8 | 11.9 | 36.2 → 10.1 |
| maskrcnn-12 | 187 → 7 | 136.6 | 22.4 | 13.6 | 35.7 → 11.1 |
| retinanet-9 | 1 → 1 | 14.4 | 13.2 | 13.6 | 11.0 → 10.7 |

The 1-6 ms gap between wall and GPU time with new inputs is the command-buffer re-recording that a changed count needs.

### 30.4 Correctness and tests

New tests:
* `SplitHalfGemmTests`: relative RMS of the SIMT GEMM, the split GEMM and plain f16 against f64 across weight and
  activation scales;
* `ConvMatrixTests`: split-f16 conv cases;
* `ChannelsLastTests`: channels-last convs (stems, strided, dilated, depthwise, chunked K), conv chains, channel
  split / concat and Resize against the NCHW path;
* `CountLevelTests`: grouped count segments and device forwarding;
* `DeviceExtendedOpsTests`: Vulkan LinearAttention / GatedDeltaRule, DeformConv, Unique and the long DFT against the
  CPU. The tree-ensemble, SVM, LabelEncoder and TfIdf kernels are covered by the conformance tests.

The branch was merged with the latest `main`, and the suites ran one at a time under the suite lock:

| suite | result |
|---|---|
| Cpu | 1265 tests, 0 failed, 36 skipped |
| Vulkan | 1312 tests, 0 failed, 22 skipped |
| Onnx (`-parallelMode none`) | 442 passed |
| Package | 32 passed |
| Families | 833 tests, 0 failed, 36 skipped |

Two fixes came out of the Families runs:
* **Shared split parts.** A draft head compiled next to its decoder uploaded a second copy of the shared LM head's
  split-f16 parts (`DraftHead_Reuses_DecoderWeights`). The split parts now get one derived source per weight tensor
  (`SplitWeightCache`), so `SharedWeights` shares them like the original tensors.
* **No host copy of uploaded split parts.** Each compiled program kept its split parts in host memory, a second host
  copy of every f32 weight on the split path. With the suite's models compiled side by side, the test process grew to
  80 GB (21 GB with `SHARPLIGHT_GEMM_SPLIT_F16=0`), and CPU placements of later tests no longer fit.
  `TransientWeightSource` produces the bytes for the upload and drops them after the last read. The suite now peaks
  at 13-22 GB, the same as with the split GEMM off.

Conformance report (cpu and vulkan): 1829 of 1884 node tests and 170 of 172 extra tests pass on both providers, as in
section 27. On Vulkan without the CPU fallback, 1809 node tests (96.0 %, was 1778) and 158 extra tests (was 128)
pass.

### 30.5 Commands

```bash
export SHARPLIGHT_ONNX_MODELS=~/models/onnx ORT_PYTHON=~/venvs/ort/bin/python3   # plus the CUDA libs of bench/onnx/env.sh
dotnet build bench/onnx/OnnxBench -c Release
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-simt,sl-vulkan-rp,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 2 --engines sl-cpu,ort-cpu
python3 bench/results/s30/gpu_timestamp_table.py gpu.jsonl ../s27/gpu.jsonl
dotnet tests/Sharplight.Vulkan.Tests/bin/Release/net11.0/Sharplight.Vulkan.Tests.dll -class "*SplitHalfGemmTests"
DOTNET_TC_QuickJitForLoops=1 dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/resnet50-v1-12.json cpu
```

### 30.6 Open gaps

* **Vulkan transformer encoders against TF32 ORT**: vit 1.7-2.0x, whisper 1.5x, bert b8 1.3x.
  * The split GEMM is bound by staging its tiles. Fusing the activation split into the producing LayerNorm / GELU
    would remove the pre-pass.
  * The tensor-core main loop needs work too: one f16 product runs at about 100 TFLOPS.
* **Vulkan convnext b8 (1.4x)** and resnet50 b8 (1.13x).
* **Split f16 for batched matmuls**: raft's correlation volume runs on the SIMT bmm.
* **The f32 SIMT GEMM** on devices without f16 matrices is unchanged.
* **Detection**: a new count re-records the command buffers (1-6 ms per new image). Reading counts from device memory
  with indirect dispatch would remove that. Faster / Mask R-CNN are at 1.2x of ORT.
* **CPU**:
  * Winograd F(4x4, 3x3) on the channels-last layout for 64 / 128-channel 3x3 convs;
  * mobilenetv2 b8 and efficientnet-lite4 at 1.04-1.06x;
  * encoder GEMMs stream each layer's weights from DRAM.

## 31. ONNX parity round 6: matmul store epilogues, f32 encoder attention and device-sized tiles for GPUs without matrix units, faster detection re-recording (2026-11-15)

Same machine and harness as section 30 (`bench/onnx/parity_bench.py`, interleaved rounds, ORT 1.31 in the same
session). Raw data is in `bench/results/s31/`. The GPU and CPU were shared with other workloads throughout.

When the first GPU pass started, another workload held the GPU at 100 % (`gpu-load-start.txt`). The encoder and conv
models therefore ran again in a second pass of 3 interleaved rounds (`gpu-rerun.jsonl`), and their table cells come
from that rerun. The rerun minima match the first pass within 0.05 ms, except convnext b1 (1.29 → 1.18).

### 31.1 What changed

* **Matmul store epilogues**: an elementwise group that consumes a weight matmul (bias, GELU, layer scale, residual
  add) runs in the split-f16 GEMM's store and in its split-K reduction, instead of as a separate launch.
  * The group is generated like the conv epilogue: `mm_epi(m0, a0, a1, a2)` reads up to three extra tensors (scalar,
    per-column vector or full output).
  * Matvec row counts (up to 8) keep the separate launch.
  * `SHARPLIGHT_NO_MATMUL_EPILOGUE=1` restores the separate launch.
* **Vulkan encoder flash attention (`fe`)**:
  * packed f16 tile staging and coalesced output stores;
  * a score layout without bank conflicts;
  * P·V sums per key tile, added to the rescaled f32 output. This also fixes the precision of a single key range.
  * whisper-tiny's attention went from 1.62 to 1.45 ms of GPU time per inference.
* **64-token split-f16 GEMM tiles up to 256 rows**:
  * The 128-token tiles hold two accumulator sets in about 220 registers, so one workgroup fits per SM. A few hundred
    rows left those grids to deep K splits.
  * vit-base b1 went from 2.18 to 2.00 ms of GPU time.
  * bert-base's 128-row matmuls move from the SIMT GEMM to the tensor cores.
  * Wide outputs (vocabulary heads) keep 128-token tiles.
* **LayerNorm over channel-strided views in place** (`layernorm_cs`):
  * The channels-last Transpose that feeds ConvNeXt's LayerNorm used to be copied to a dense tensor first.
  * It is now read in place: 32 rows per workgroup, through shared-memory transposes, coalesced on both sides.
  * convnext-tiny GPU time: b1 1.27 → 1.18 ms, b8 3.48 → 3.45 ms.
  * `SHARPLIGHT_NO_STRIDED_LAYERNORM=1` restores the copy.
* **`conv_narrow` for convs with up to 8 output channels** (flow, depth and mask heads):
  * The implicit GEMMs' smallest tiles cover 32 output channels, so these convs mostly computed padding.
  * conv_narrow puts 32 pixels in a workgroup and splits the input channels over its subgroups, with uniform weight
    reads.
  * raft-large's [256 → 2 3x3] flow head went from 129 to 22 µs. It runs 12 times per inference.
  * `SHARPLIGHT_NO_NARROW_CONV=1` disables it.
* **Re-recording for a new data-dependent count**: 2-3 ms → 0-1 ms for the 230 launches of the R-CNN box heads.
  * Every symbol lookup re-parsed its count expression (5-10 µs per launch). `Bindings` now memoize the evaluated
    expressions.
  * The barrier placement is cached by the set of launches that run, which a new count rarely changes.
* **Devices without matrix units** (integrated GPUs, llvmpipe, the `sl-vulkan-simt` engine):
  * **`fes` attention.** Bidirectional attention ran the per-query kernel, which re-reads every key for each query:
    16.5 ms per vit-base layer on the integrated GPU.
    * `fes` is an f32 flash kernel. Each workgroup handles 64 queries, one per thread, with the K / V tiles in shared
      memory.
    * It serves the same forms as `fe` (masks, batched, cross, key ranges) for head sizes up to 64.
    * `SHARPLIGHT_ATTN_FES=0` disables it.
  * **`DeviceInfo.ConcurrentWorkgroups`** (also in `DeviceProfile`) is a sizing hint.
    * It is 64 for integrated GPUs and CPU implementations, and 0 for discrete GPUs (unknown, sized as before).
    * The SIMT GEMM's split-K target and the conv and bmm tilings use it.
    * Small devices take the 128 x 128 `gemm_wide` tile from 128 rows. On a device that runs a few dozen
      workgroups at once, K splits only added partial-sum traffic.

### 31.2 GPU: Sharplight Vulkan vs onnxruntime CUDA EP

* Cells are min / median ms over the rounds.
* §30 is section 30's `sl-vulkan` min.
* (!) marks a parity failure against the stored reference. The reduced-precision engine (`sl-vulkan-rp`) fails parity
  by design on several models. phi3.5 fails on every engine, as in section 30.

| model | batch | §30 | sl-vulkan | sl-vulkan-simt | sl-vulkan-rp | ort-cuda | ort-cuda-strict | SL / ORT (§30) | SL / ORT |
|---|---|---|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.77 | 0.74 / 0.75 | 0.81 | 0.71 (!) | 0.99 | 2.10 | 0.75x | **0.75x** |
| resnet50 | 8 | 2.28 | 2.27 / 2.31 | 3.09 | 1.93 (!) | 1.97 | 3.67 | 1.13x | 1.15x |
| convnext-tiny | 1 | - | 1.18 / 1.20 | 1.22 | 1.21 | 1.01 | 1.41 | - | 1.17x |
| convnext-tiny | 8 | 3.65 | 3.50 / 3.62 | 3.96 | 2.65 (!) | 2.54 | 3.97 | 1.42x | 1.38x |
| efficientnet-lite4 | 1 | 0.75 | 0.75 / 0.82 | 0.83 | 0.75 | 0.67 | 7.72 | 1.09x | 1.12x |
| mobilenetv2 | 8 | 0.69 | 0.69 / 0.71 | 0.72 | 0.67 (!) | 0.75 | 1.34 | 0.92x | **0.92x** |
| yolov8n | 1 | - | 1.27 / 1.40 | 1.35 | 1.31 (!) | 1.31 | 2.76 | - | **0.97x** |
| yolov10n | 1 | 1.19 | 1.19 / 1.24 | 1.26 | 1.24 (!) | 1.18 | 2.30 | 0.98x | 1.01x |
| ssd-10 (NMS) | 1 | 12.38 | 12.01 / 12.59 (!) | 16.51 (!) | 8.02 (!) | 10.94 | 22.38 | 1.11x | 1.10x |
| bert-base (16 tok) | 1 | 1.27 | 1.19 / 1.21 | 1.17 | 1.15 (!) | 1.23 | 1.42 | 0.91x | **0.97x** |
| bert-base (16 tok) | 8 | 3.46 | 3.26 / 3.41 | 3.46 | 3.29 (!) | 2.58 | 3.20 | 1.27x | 1.26x |
| vit-base | 1 | 2.36 | 2.04 / 2.06 | 2.37 | 1.89 | 1.34 | 2.46 | 1.71x | 1.52x |
| vit-base | 8 | 8.06 | 7.24 / 7.59 | 9.31 | 4.55 (!) | 3.88 | 8.98 | 2.03x | 1.87x |
| whisper-tiny-encoder | 1 | 1.78 | 1.53 / 1.56 | 1.66 | 1.15 (!) | 1.15 | 2.03 | 1.49x | 1.33x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | - | 3.11 / 3.29 | 3.12 | 2.45 (!) | 3.84 | 3.81 | - | **0.81x** |
| phi3.5-mini q4f16 (GQA) | 1 | - | 4.61 / 5.16 (!) | 4.56 (!) | 3.52 (!) | 3.38 | 3.36 | - | 1.36x |
| segformer-b0 | 1 | - | 2.86 / 2.91 | 2.85 | 2.77 | 2.29 | 3.30 | - | 1.25x |
| segformer-b0 | 8 | 20.98 | 20.68 / 21.07 | 21.24 | 18.84 | 19.18 | 22.32 | 1.07x | 1.08x |
| mobilesam-encoder | 1 | 5.26 | 4.99 / 5.94 | 5.67 | 4.29 (!) | 6.00 | 7.80 | 0.86x | **0.83x** |
| raft-large | 1 | 37.82 | 36.14 / 36.58 | 46.92 | 33.70 (!) | 28.52 | 70.01 | 1.29x | 1.27x |
| fasterrcnn-12 (counts) | 1 | 9.68 | 8.68 / 10.21 | 10.19 | 7.56 | 7.55 | 13.07 | 1.22x | 1.15x |
| maskrcnn-12 (counts) | 1 | 9.48 | 9.57 / 10.41 | 11.03 | 8.32 | 7.43 | 13.19 | 1.25x | 1.29x |

* **The encoders gained 6-14 %:**
  * vit b1 2.36 → 2.04 ms and vit b8 8.06 → 7.24 ms;
  * whisper 1.78 → 1.53 ms;
  * bert b8 3.46 → 3.26 ms.
* **Against ORT with TF32 off**, the same models run at 0.81-0.83x (vit b8, b1), 0.75x (whisper) and 1.02x (bert b8).
* **Most of the remaining gap to TF32 ORT is structural.**
  * The strict split GEMM forms three f16 products per multiply-add (hi·hi + hi·lo + lo·hi). TF32 forms one product
    with 10 mantissa bits.
  * At equal efficiency that costs about 1.5x the time.
  * vit b8's [768 → 3072] GEMM runs at about 146 f16-equivalent TFLOPS.
* **With reduced precision** (opt-in), vit b8 takes 4.55 ms and whisper 1.15 ms: 1.17x and 1.0x of TF32 ORT.

Detection with a new image every run (`detection-ab.log`):

* `--vary-input`, 30 runs, 3 interleaved rounds;
* "old" is `main` before this round;
* times in ms.

| model | wall min, old → new | wall median, old → new | GPU min | mean re-recording, old → new |
|---|---|---|---|---|
| fasterrcnn-12 | 9.52 → 9.09 | 10.24-11.51 → 9.68-10.60 | 8.27-8.34 | 0.1-1.1 → 0.0 |
| maskrcnn-12 | 10.65 → 10.01 | 14.12-16.30 → 10.99-11.38 | 9.03-9.08 | 1.1-1.5 → 0.0-0.1 |

### 31.3 Integrated GPU (no matrix units)

Setup (`igpu-ab.log`):

* AMD Radeon integrated graphics (RDNA2, 2 CUs, RADV);
* batch 1, strict precision;
* min wall ms of 5 runs, over 2 rounds interleaved with `main`.

All runs pass parity.

| model | main | this round | speedup |
|---|---|---|---|
| whisper-tiny-encoder | 2424 | 152 | 16x |
| vit-base | 404 | 126 | 3.2x |
| resnet50 | 72.8 | 53.6 | 1.36x |
| convnext-tiny | 67.3 | 53.8 | 1.25x |
| mobilenetv2 | 8.62 | 7.44 | 1.16x |
| bert-base (16 tok) | 36.7 | 35.2 | 1.04x |

* **whisper and vit gain from `fes`.** The per-query kernel took 16.5 ms per vit layer; `fes` takes 1.3 ms.
* **The GEMMs and convs gain from sizing by `ConcurrentWorkgroups`.**
  * vit's [768 → 3072] at 197 rows went from 4.5 to 2.8 ms (`gemm_wide`, no split).
  * resnet50's 3x3 convs dropped by 20-30 %.

### 31.4 CPU: Sharplight vs onnxruntime CPU EP (16 threads)

No CPU kernel changed in this round. Setup:

* one session, 2 interleaved rounds;
* the 1-minute load average was 2-12 (`cpu-load.log`);
* cells are min / median ms; §30 is section 30's `sl-cpu` min.

| model | batch | §30 | sl-cpu | ort-cpu | SL / ORT |
|---|---|---|---|---|---|
| resnet50 | 1 | 4.61 | 4.09 / 9.01 | 5.24 | **0.78x** |
| resnet50 | 8 | 23.13 | 21.29 / 39.27 | 20.34 | 1.05x |
| convnext-tiny | 1 | 13.23 | 12.72 / 18.78 | 12.14 | 1.05x |
| convnext-tiny | 8 | 76.23 | 60.17 / 95.62 | 74.96 | **0.80x** |
| efficientnet-lite4 | 1 | 4.19 | 3.17 / 4.36 | 3.71 | **0.86x** |
| mobilenetv2 | 1 | 0.98 | 0.60 / 0.76 | 1.55 | **0.38x** |
| mobilenetv2 | 8 | 6.96 | 5.55 / 8.56 | 5.77 | **0.96x** |
| yolov8n | 1 | 7.79 | 7.88 / 15.47 | 12.27 | **0.64x** |
| yolov10n | 1 | 9.32 | 7.56 / 10.87 | 14.87 | **0.51x** |
| bert-base (16 tok) | 1 | 9.82 | 9.07 / 13.29 | 10.78 | **0.84x** |
| bert-base (16 tok) | 8 | 20.41 | 17.81 / 25.09 | 24.75 | **0.72x** |
| vit-base | 1 | 21.31 | 23.91 / 36.09 | 21.75 | 1.10x |
| vit-base | 8 | 140.08 | 120.25 / 171.04 | 124.83 | **0.96x** |
| whisper-tiny-encoder | 1 | 21.49 | 19.80 / 35.74 | 37.47 | **0.53x** |
| segformer-b0 | 1 | 23.27 | 21.56 / 29.40 | 39.96 | **0.54x** |
| segformer-b0 | 8 | 236.00 | 209.85 / 245.21 | 453.68 | **0.46x** |
| mobilesam-encoder | 1 | 130.88 | 112.76 / 161.14 | 198.77 | **0.57x** |

* The Sharplight medians run 1.4-2x their minima, against 1.05-1.1x for ORT. At this load only the minima are
  comparable.
* Three CPU rows are above ORT: resnet50 b8 (1.05x), convnext b1 (1.05x) and vit b1 (1.10x).

### 31.5 Correctness and tests

New tests:

* `MatMulEpilogueTests`: bias + GELU and bias · γ + residual, at the matvec, split-K and plain GEMM row counts.
* `RowOpTests.LayerNormChannelsLastView`.
* `ConvTests.NarrowOutputs` and `NarrowOutputsSlicedDilated`: f32, f16 and bf16 weights, strides, dilation and a
  sliced input.
* `FullAttentionTests.WithoutMatrixUnits`: `fes` with the matrix features disabled, covering masks, the batched and
  cross forms, and key ranges.

The attention, conv, SIMT GEMM, row-op and matmul classes (253 tests) also pass on the integrated GPU
(`SHARPLIGHT_TEST_DEVICE=1`).

Suites on the branch merged with `main`, one at a time:

| suite | result |
|---|---|
| Cpu | 1265 tests, 0 failed, 36 skipped |
| Vulkan | 1333 tests, 0 failed, 22 skipped |
| Onnx (`-parallelMode none`) | 442 passed |
| Package | 32 passed |
| Families | 833 tests, 0 failed, 36 skipped |

Conformance report (cpu and vulkan): 1829 of 1884 node tests and 170 of 172 extra tests pass on both providers, as in
section 30.

### 31.6 Commands

```bash
export SHARPLIGHT_ONNX_MODELS=~/models/onnx ORT_PYTHON=~/venvs/ort/bin/python3   # plus the CUDA libs of bench/onnx/env.sh
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-simt,sl-vulkan-rp,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 2 --engines sl-cpu,ort-cpu
dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/vit-base-patch16-224.json vulkan --device 1 --runs 5
SHARPLIGHT_LOAD_TIMING=1 dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/survey/maskrcnn-12.json vulkan --vary-input
```

### 31.7 Open gaps

* **Vulkan encoders against TF32 ORT**: vit b8 1.87x, vit b1 1.52x, whisper 1.33x, bert b8 1.26x.
  * The split GEMM's three products cost about 1.5x of TF32 by construction.
  * Its 128-token tiles run one workgroup per SM (about 220 registers).
  * The activation split is a pre-pass (`split_rows`, 0.2-0.5 ms per vit b8 inference). Fusing it into the
    producing LayerNorm / GELU would remove it.
* **convnext b8 (1.38x)**:
  * its 25088-row [96 ↔ 384] GEMMs have only 3-12 K steps and run at 27-40 f16-equivalent TFLOPS;
  * the split-K reductions add 0.3 ms.
* **raft (1.27x)**:
  * the split convs (1.5x structural);
  * the correlation volume on the SIMT bmm (1.1 ms);
  * about 290 small concats per inference.
* **Detection**: Faster / Mask R-CNN run at 1.15-1.29x of ORT on GPU time.
* **CPU fallback**: ScatterND over a symbolic axis (qwen2.5 MatMulNBits) is the last one in the benchmark set.
* **CPU**:
  * Winograd F(4x4, 3x3) on the channels-last layout;
  * vit b1 (1.10x) and resnet50 b8 (1.05x);
  * encoder GEMMs stream each layer's weights from DRAM.
* **Devices without matrix units**:
  * `fes` serves head sizes up to 64 only.
  * `ConcurrentWorkgroups` is a per-device-type heuristic, not a query of the compute units.

## 32. Quantized Hugging Face checkpoints: NVFP4, MXFP4, GPTQ, AWQ, bitsandbytes, MLX (2026-11-14)

Same machine as section 1. Single stream on Qwen3-8B: one real Hub checkpoint per format, plus Sharplight's own Q4_K_M GGUF
as the baseline. The Sharplight CLI was run as `--bench N --ctx N+256 --batch 512` for N = 16 / 512 / 2048 / 4096. There were 3
repetitions, with the models interleaved inside every repetition (A/B on a shared GPU). vLLM 0.33 used the offline API
(`bench/vllm_single.py`, 5 repetitions, median), with flash-attn, CUDA graphs and a bf16 KV cache (f16/auto for the f16 AWQ
checkpoint). Raw data and both scripts are in `bench/results/s32/`.

Checkpoints:

| Format | Checkpoint | vLLM kernel | Sharplight weight type | Weights on GPU |
|---|---|---|---|---|
| NVFP4 (ModelOpt) | nvidia/Qwen3-8B-NVFP4 | CUTLASS W4A4 (FP4 tensor cores, activations quantized to FP4) | `NVFP4_GS` (W4A16; int-dot matvec on q8_1 activations) | 6.83 GB |
| AWQ GEMM, g128 | Qwen/Qwen3-8B-AWQ | Marlin | `Q4Z_G128` | 6.18 GB |
| GPTQ 4-bit, g128 | JunHowie/Qwen3-8B-GPTQ-Int4 | Marlin | `Q4Z_G128` | 6.18 GB |
| bitsandbytes NF4 (double quant) | unsloth/Qwen3-8B-bnb-4bit | not supported by vLLM 0.33 | `NF4_B64` | 6.40 GB |
| GGUF Q4_K_M | Qwen3-8B-Q4_K_M.gguf | | `Q4_K` / `Q6_K` | 5.0 GB |

All weights stay quantized on the GPU. The only formats dequantized at load are the ones without a fitting block type:
bitsandbytes FP4, LLM.int8 and MLX at bit widths other than 4. These decode to BF16 at load and are not benchmarked here.

### 32.1 Throughput (tok/s; Sharplight pp = best of 3 runs, tg = median of the per-run medians; vLLM = median of 5)

| Checkpoint | Engine | pp512 | pp2048 | pp4096 | tg@16 | tg@4096 |
|---|---|---|---|---|---|---|
| NVFP4 | vLLM | 18119 | 36251 | 36436 | 113.4 | 93.9 |
| NVFP4 | Sharplight | 9314 | 7422 | 6704 | **157.7** | **148.2** |
| AWQ | vLLM | 11624 | 11903 | 11378 | 168.0 | 158.4 |
| AWQ | Sharplight | 8949 | 8028 | 6576 | **195.9** | 156.6 |
| GPTQ | vLLM | 10866 | 11465 | 11111 | 159.6 | 154.2 |
| GPTQ | Sharplight | 8904 | 7774 | 6746 | **187.6** | **175.2** |
| bnb NF4 | Sharplight | 7903 | 6755 | 6130 | 131.8 | 111.6 |
| Q4_K_M (baseline) | Sharplight | 8208 | 7096 | 6240 | 225.4 | 168.6 |

Decode now beats vLLM on all three formats vLLM runs, except AWQ at depth 4096, where the two are level. Prefill lags
by 20-40 % against Marlin (GPTQ / AWQ) and by 2-5x against vLLM's W4A4 NVFP4 path. The Blackwell GPU's Vulkan driver
exposes no FP4 (or FP8) cooperative-matrix types, so NVFP4 prefill dequantizes to f16 in the tensor-core GEMM, the same
path as the integer formats. Matching CUTLASS W4A4 would need FP4 coopmat, or a W4A4 integer-MMA emulation; the latter
would also change the numerics vLLM gets from activation quantization.

### 32.2 Before / after the decode kernels of this round (best GPU ms per decode step at N = 16)

"Before" is the same branch with the formats imported and decoded through the generic `load8` / `load16h` paths. "After"
adds the 8-bit integer-dot matvec (`mmvq`) for the zero-point and NVFP4 types and moves the NF4 lookup into the grid
buffer.

| Checkpoint | before | after | tg@16 before → after (tok/s) |
|---|---|---|---|
| NVFP4 | 6.98 | 4.66 | 117.4 → 157.7 |
| AWQ | 5.92 | 4.06 | 140.2 → 195.9 |
| GPTQ | 5.66 | 4.19 | 139.1 → 187.6 |
| bnb NF4 | 15.96 | 6.18 | 54.5 → 131.8 |
| Q4_K_M (unchanged code) | 4.38 | 4.13 | 183.4 → 225.4 |

The two runs were hours apart on the shared GPU. The unchanged Q4_K_M row shows the "after" run had a quieter GPU
(about 6 % on GPU time, more on wall-clock tok/s), so the GPU-time columns are the fair comparison. In the bnb NF4
row, the move from a local `static const` table (divergent indexing spilled to local memory) to the grid buffer also
doubled prefill (4274 → 7903 tok/s at pp512).

### 32.3 Exactness

* **Dequantization**: the tiny checkpoints were made with the libraries themselves where they install (llmcompressor /
  compressed-tensors, AutoAWQ, bitsandbytes, ModelOpt, MLX). GPTQ and Quark packing follow the gptqmodel / quark source.
  These checkpoints, plus two real Qwen3-0.6B NVFP4 checkpoints, decode on the CPU and on Vulkan **bit-exact** against the
  library's own dequantization, after rounding to the library's output dtype (`QuantizedImportTests`).
* **Logits** against an f32 transformers forward over the library dequantization: within 1 % of the logit spread on the
  CPU (f32 activations) and 4 % on Vulkan (q8_1 activations in the int-dot matvec, f16 tensor-core prefill), with the same
  top-1 (`QuantizedCheckpointTests`).
* **Greedy regression**: 200-token CLI outputs are identical to main for Qwen2.5-0.5B (HF, CPU and Vulkan; Q8_0),
  Llama-3.2-1B Q4_K_M (CPU and Vulkan), Qwen3-8B Q4_K_M, gpt-oss-20b and the 27B FP8.

### 32.4 Open gaps

* NVFP4 prefill (no FP4 coopmat on Vulkan, see above). `NVFP4_GS` stores the per-tensor f32 global scale in every
  64-value block (5.0 bits per weight against 4.5). Folding it into the UE4M3 block scales is only exact when it is a
  power of two, and that does not hold in practice.
* GPTQ / AWQ prefill: 6.6-8.9k tok/s against Marlin's 11-12k. The `Q4Z` decode in the f16 GEMM is the same cost as Q4_0
  plus one subtract per half2. The gap is in the GEMM itself, shared with Q4_K_M (8.2k).
* bitsandbytes FP4, LLM.int8 (outliers stay in the BF16 decode: the int8 weights times the row scale, matching
  `int8_vectorwise_dequant`) and non-4-bit MLX are dequantized at load.
* Not supported: HQQ, AQLM, EETQ, quanto, EXL2, torchao, VPTQ, HIGGS, SpQR, AutoRound-native, FP-Quant, Quark integer and
  MXFP6, AWQ `gemv_fast`, Marlin-repacked checkpoints, and the MLX FP4 / MXFP4 modes.

## 33. ONNX parity round 7: opt-in fast matrix precision (TF32 class), ScatterND over symbolic axes on Vulkan, phi-3.5 reference, 36 conformance fixes (2026-11-15)

Same machine and harness as sections 30 and 31: `bench/onnx/parity_bench.py`, interleaved rounds, and ORT 1.31 (CPU EP, CUDA EP with
TF32 and with TF32 off) re-run in the same session. Raw data is in `bench/results/s32/`. The GPU and CPU were shared with
other workloads throughout. During the first GPU pass, the GPU was at 38-96 % utilization before Sharplight's runs, and
medians were often twice the minima (`gpu-load-*.txt`, `gpu.jsonl`). The six target models and Faster R-CNN therefore ran
again in a second pass of 4 interleaved rounds (`gpu-rerun.jsonl`). The tables below take those rows from the rerun.

### 33.1 What changed

* **Fast matrix precision (opt-in)**: `CompileOptions.MatrixPrecision = MatrixPrecision.Fast` (or
  `SHARPLIGHT_MATRIX_PRECISION=fast`) trades f32 accuracy for speed, in the accuracy class of ORT's default TF32.
  Strict stays the default. The setting applies to f32 GEMMs, convs and attention.
  * **GEMM and conv**: the split-f16 tensor-core paths keep their row and block scales but drop the lo parts. Each value
    becomes one f16 with 11 significant bits, which is TF32's precision. Each multiply-add is one tensor-core product
    with f32 accumulation, instead of the three products of the strict split (hi·hi + hi·lo + lo·hi). The scales keep
    values far outside f16's range finite and accurate.
  * Weights are split once on the host (`SplitScaleTable`, hi parts only). Activations go through the same `split_rows`
    pre-pass, which writes half the bytes. The GEMM uses 64-value K blocks and the 8-warp layout. It also uses the
    tensor-core path on grids too small for the strict split.
  * **Attention**: f32 encoder attention runs the single-product f16 flash kernel (`fe`) instead of its split-f16 form.
  * Devices without f16 matrix units keep the strict kernels. The older `AllowReducedPrecision` mode (plain f16 without
    scales) takes precedence when both are set.
  * Benchmarks: OnnxBench `--fast-precision`, and the parity-bench engine `sl-vulkan-fast`.
* **ScatterND over a symbolic axis on Vulkan**: the parallel ScatterND kernel takes the runtime extent of a symbolic
  leading axis as a push constant. Only a ScatterND with a reduction over such an axis still falls back to the CPU.
  qwen2.5-0.5b q4 (`model_q4`, KV-cache updates) now runs as one Vulkan segment, worst relative RMS 1.1e-5.
* **phi-3.5-mini q4f16 parity**: it failed on every engine, ORT CUDA included, because the survey reference was ORT
  CPU's fp16 run. That run is 4.8e-2 relative RMS from an f32 copy of the same graph, and ORT CUDA's fp16 run is 5.5e-2
  from the copy. Sharplight computes these graphs in f32 and lands at 1.2e-4 from the copy (Vulkan and CPU).
  `bench/onnx/make_survey_refs.py` now builds this model's reference from the f32 copy (`f32ref`), and parity passes.
  qwen2.5-0.5b fp16 has the same cause and stays documented: against the f32 copy, ORT fp16 is 7.3e-2 off, Sharplight
  CPU 2.3e-4 and Vulkan 1.5e-3 (see `docs/onnx-support.md`).
* **Conformance**: 36 of the 57 remaining failures fixed (classified in `docs/onnx-support.md`):
  * zero-size inputs become empty constants, and reductions over an empty set fold to their identity;
  * `OnnxOutputLayout` maps NonZero, Unique-along-axis, StringSplit and unit-dim outputs back to the ONNX layout;
  * new ops: Unique along an axis, StringSplit, com.microsoft MurmurHash3 and WordConvEmbedding.

### 33.2 Fast precision against ORT's TF32, strict against ORT strict

Wall min latency in ms, Vulkan against the CUDA EP on the same GPU (rows from `gpu-rerun.jsonl`, 4 interleaved rounds):

| model | batch | SL fast | ORT TF32 | fast / TF32 | SL strict | ORT strict | strict / strict | SL strict / ORT TF32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit-base | 1 | 1.23 | 1.36 | 0.90x | 2.04 | 2.48 | 0.82x | 1.50x |
| vit-base | 8 | 3.65 | 4.22 | 0.86x | 7.99 | 10.22 | 0.78x | 1.89x |
| whisper-tiny encoder | 1 | 1.04 | 1.22 | 0.85x | 1.61 | 2.10 | 0.77x | 1.32x |
| bert-base (16 tokens) | 1 | 1.05 | 1.24 | 0.85x | 1.19 | 1.46 | 0.82x | 0.96x |
| bert-base (16 tokens) | 8 | 2.79 | 2.79 | 1.00x | 3.48 | 3.42 | 1.02x | 1.25x |
| convnext-tiny | 1 | 1.00 | 1.00 | 1.00x | 1.18 | 1.40 | 0.84x | 1.18x |
| convnext-tiny | 8 | 2.25 | 2.57 | 0.88x | 3.92 | 4.50 | 0.87x | 1.52x |
| resnet50 | 1 | 0.61 | 1.02 | 0.60x | 0.74 | 2.18 | 0.34x | 0.72x |
| resnet50 | 8 | 1.73 | 1.99 | 0.87x | 2.35 | 3.94 | 0.60x | 1.18x |
| raft-large | 1 | 33.66 | 32.35 | 1.04x | 40.41 | 82.95 | 0.49x | 1.25x |
| fasterrcnn-12 | 1 | 8.24 | 8.69 | 0.95x | 9.95 | 15.29 | 0.65x | 1.14x |

* In fast precision, Sharplight matches or beats ORT's TF32 default on every target model except raft (1.04x). Strict
  precision beats ORT with TF32 off on all of them except bert b8 (1.02x).
* Fast is 1.1-2.2x faster than strict on the encoders (vit b8 7.99 → 3.65 ms) and 1.2-1.7x on the convnets.
* The strict cells did not change from round 6. An interleaved A/B against a build of the round-6 commit
  (`strict-ab-round6.log`, GPU min over 4-6 rounds) gives:
  * convnext b8: 3.49 against 3.47 ms;
  * bert b8: 2.83 against 2.83 ms;
  * vit b8: 7.55 against 7.77 ms, but the rounds overlap (7.55-8.13 against 7.77-8.11).

  The strict kernels compile from the same code. The higher ratios to ORT than in §31 come from the shared GPU: in this
  session medians were often twice the minima.

### 33.3 Accuracy against an f64 reference

`bench/onnx/f64_reference.py` builds an f64 copy of each graph and runs it on onnx's ReferenceEvaluator. onnxruntime has
no f64 Conv or Erf, and the evaluator's opset-12 BatchNormalization is wrong, so it is rewritten as f64 Mul / Add.
`bench/onnx/f64_error_ort.py` scores ORT's engines against it, and OnnxBench's worst relative RMS scores Sharplight's.
The cells are the worst relative RMS over the outputs at batch 1 (`error-f64.jsonl`):

| model | ORT CPU | ORT CUDA strict | ORT CUDA TF32 | SL CPU | SL Vulkan SIMT f32 | SL Vulkan strict | SL Vulkan fast | SL Vulkan reduced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vit-base | 4.9e-7 | 3.0e-7 | 5.5e-4 | 1.3e-5 | 1.3e-6 | 1.3e-6 | 6.5e-4 | 4.4e-4 |
| whisper-tiny encoder | 9.3e-7 | 1.9e-6 | 6.7e-4 | 5.7e-6 | 1.7e-6 | 1.8e-6 | 6.5e-4 | 5.8e-4 |
| bert-base | 7.3e-7 | 9.2e-7 | 7.5e-4 | 1.8e-6 | 1.7e-6 | 1.7e-6 | 6.5e-4 | 2.7e-4 |
| convnext-tiny | 5.4e-7 | 3.2e-7 | 6.0e-4 | 4.0e-6 | 2.9e-6 | 3.2e-6 | 5.6e-4 | 1.6e-4 |
| resnet50 | 4.0e-7 | 2.0e-4 | 7.9e-4 | 1.1e-6 | 8.3e-7 | 9.3e-7 | 6.8e-4 | 8.9e-4 |

* **Fast vs TF32**: fast precision lands at 5.6-6.8e-4, ORT's TF32 default at 5.5-7.9e-4. That is the same class;
  fast is slightly more accurate on 4 of the 5 models.
* **Strict**: Sharplight Vulkan strict stays at 0.9-3.2e-6, the same as its f32 SIMT GEMMs. ORT strict is at 0.3-1.9e-6,
  except resnet50 at 2.0e-4: with TF32 off, cuDNN still picks a reduced-precision conv algorithm there.
* **Reduced precision**: `AllowReducedPrecision` (unscaled f16) is more accurate than fast on bert and convnext and less on
  resnet50 (8.9e-4). It also has no range protection for values beyond f16, and is slower on the encoders (vit b8 5.04
  against 3.65 ms).
* **Sharplight CPU** is 1.1e-6 to 1.3e-5, up to 25x ORT CPU (vit). The approximate exp / erf of the CPU kernels is the
  likely cause; not investigated this round.
* **raft-large** has no f64 row: the reference evaluator did not finish within 50 minutes.

### 33.4 All models

GPU (`table-gpu-combined.md`: the first pass, with the rows above from the rerun). Cells are min / median of the rounds in ms;
`(!)` marks a parity miss against the f32 reference (expected for fast and reduced precision; ssd-10 reorders tied
detections, as in §30-§31):

| model | batch | sl-vulkan | sl-vulkan-rp | sl-vulkan-fast | sl-vulkan-simt | ort-cuda | ort-cuda-strict | SL/ORT CUDA |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 1 | 0.74 / 1.99 | 0.69 / 2.00 (!) | 0.61 / 1.01 (!) | 0.81 / 2.08 | 1.02 / 2.57 | 2.18 / 5.07 | 0.72x |
| resnet50 | 8 | 2.35 / 5.25 | 1.94 / 4.43 (!) | 1.73 / 4.01 (!) | 3.17 / 7.14 | 1.99 / 3.45 | 3.94 / 6.64 | 1.18x |
| convnext-tiny | 1 | 1.18 / 2.52 | 1.21 / 2.63 | 1.00 / 1.68 (!) | 1.22 / 2.61 | 1.00 / 2.38 | 1.40 / 2.81 | 1.18x |
| convnext-tiny | 8 | 3.92 / 7.38 | 2.67 / 5.79 (!) | 2.25 / 5.07 (!) | 4.29 / 8.95 | 2.57 / 5.39 | 4.50 / 7.03 | 1.52x |
| efficientnet-lite4 | 1 | 0.76 / 1.94 | 0.76 / 2.20 | 0.65 / 1.91 | 0.85 / 2.19 | 0.70 / 1.65 | 10.26 / 14.51 | 1.08x |
| mobilenetv2 | 1 | 0.38 / 0.56 | 0.38 / 0.43 (!) | 0.33 / 0.38 (!) | 0.42 / 0.59 | 0.38 / 0.56 | 0.66 / 1.07 | 1.02x |
| mobilenetv2 | 8 | 0.70 / 1.17 | 0.68 / 1.04 (!) | 0.64 / 0.94 (!) | 0.74 / 1.25 | 0.76 / 1.23 | 1.42 / 2.96 | 0.92x |
| yolov8n | 1 | 1.29 / 2.57 | 1.39 / 2.69 (!) | 1.16 / 1.77 (!) | 1.41 / 2.91 | 1.41 / 2.52 | 3.53 / 5.55 | 0.92x |
| yolov10n | 1 | 1.20 / 2.51 | 1.26 / 2.77 (!) | 1.06 / 1.60 (!) | 1.28 / 2.75 | 1.24 / 3.05 | 2.70 / 4.78 | 0.97x |
| ssd-10 (NMS) | 1 | 19.67 / 27.62 (!) | 10.00 / 16.62 (!) | 15.74 / 19.07 (!) | 32.09 / 37.64 (!) | 18.72 / 24.52 | 42.55 / 47.79 | 1.05x |
| bert-base (16 tok) | 1 | 1.19 / 2.57 | 1.15 / 2.38 (!) | 1.05 / 2.56 (!) | 1.19 / 2.43 | 1.24 / 2.31 | 1.46 / 2.73 | 0.96x |
| bert-base (16 tok) | 8 | 3.48 / 6.30 | 3.44 / 6.86 (!) | 2.79 / 6.09 (!) | 3.95 / 7.00 | 2.79 / 3.92 | 3.42 / 4.59 | 1.25x |
| vit-base | 1 | 2.04 / 4.72 | 1.88 / 4.39 | 1.23 / 2.65 (!) | 2.54 / 5.32 | 1.36 / 2.66 | 2.48 / 5.24 | 1.50x |
| vit-base | 8 | 7.99 / 17.31 | 5.04 / 9.57 (!) | 3.65 / 8.10 (!) | 10.57 / 22.94 | 4.22 / 7.02 | 10.22 / 17.64 | 1.89x |
| whisper-tiny-encoder | 1 | 1.61 / 3.45 | 1.20 / 2.58 (!) | 1.04 / 2.30 (!) | 1.73 / 4.15 | 1.22 / 1.95 | 2.10 / 3.29 | 1.32x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.27 / 6.35 | 2.52 / 5.24 (!) | 3.10 / 6.40 | 3.30 / 5.91 | 4.96 / 6.60 | 5.30 / 7.43 | 0.66x |
| phi3.5-mini q4f16 (GQA) | 1 | 5.19 / 10.44 | 3.92 / 7.37 (!) | 5.90 / 10.23 | 5.37 / 9.82 | 3.95 / 6.42 | 4.11 / 6.39 | 1.31x |
| segformer-b0 | 1 | 3.09 / 6.21 | 2.82 / 5.32 | 2.82 / 5.50 (!) | 3.03 / 6.17 | 2.46 / 3.76 | 3.65 / 5.80 | 1.25x |
| segformer-b0 | 8 | 23.51 / 36.22 | 19.86 / 31.98 | 21.89 / 33.19 (!) | 22.46 / 36.71 | 23.55 / 65.86 | 30.41 / 37.69 | 1.00x |
| mobilesam-encoder | 1 | 5.50 / 11.24 | 4.64 / 9.60 (!) | 4.41 / 8.89 (!) | 6.38 / 12.79 | 6.57 / 10.88 | 10.49 / 14.49 | 0.84x |
| raft-large | 1 | 40.41 / 79.24 | 37.72 / 60.99 (!) | 33.66 / 63.03 (!) | 53.14 / 103.76 | 32.35 / 47.36 | 82.95 / 120.59 | 1.25x |
| fasterrcnn-12 (counts) | 1 | 9.95 / 17.49 | 8.28 / 15.73 | 8.24 / 12.23 | 11.42 / 22.38 | 8.69 / 21.92 | 15.29 / 26.54 | 1.14x |
| maskrcnn-12 (counts) | 1 | 15.28 / 20.51 | 12.36 / 17.30 | 9.70 / 16.51 | 14.00 / 23.51 | 16.46 / 22.23 | 22.88 / 34.24 | 0.93x |

Notes on the GPU table:
* qwen2.5 q4 now runs entirely on Vulkan (ScatterND); it is at 0.66x of ORT CUDA.
* phi-3.5 q4f16 passes parity against its new f32-copy reference.
* The q4 decoders do not gain from fast precision: their GEMMs are block-quantized and keep their own kernels.

CPU, 16 threads on both engines, 3 interleaved rounds (`cpu.jsonl`). The load average was 14-26 during the session: other
workloads, including other test suites, ran at the same time. Cells are min / median in ms:

| model | batch | sl-cpu | ort-cpu | SL/ORT CPU |
|---|---|---|---|---|
| resnet50 | 1 | 5.28 / 8.31 | 6.38 / 8.90 | 0.83x |
| resnet50 | 8 | 26.89 / 36.45 | 25.91 / 34.96 | 1.04x |
| convnext-tiny | 1 | 12.89 / 17.10 | 15.92 / 23.16 | 0.81x |
| convnext-tiny | 8 | 80.15 / 115.99 | 101.75 / 147.04 | 0.79x |
| efficientnet-lite4 | 1 | 3.62 / 6.41 | 5.53 / 11.72 | 0.65x |
| mobilenetv2 | 1 | 1.12 / 2.05 | 1.77 / 4.61 | 0.63x |
| mobilenetv2 | 8 | 9.53 / 14.33 | 7.98 / 11.45 | 1.19x |
| yolov8n | 1 | 8.34 / 11.44 | 15.49 / 20.44 | 0.54x |
| yolov10n | 1 | 10.32 / 22.67 | 16.52 / 21.14 | 0.63x |
| ssd-10 (NMS) | 1 | 162.97 / 189.53 (!) | 240.99 / 332.83 | 0.68x |
| bert-base (16 tok) | 1 | 10.48 / 13.86 | 14.30 / 24.42 | 0.73x |
| bert-base (16 tok) | 8 | 25.70 / 34.25 | 31.70 / 42.59 | 0.81x |
| vit-base | 1 | 24.90 / 64.11 | 27.50 / 35.69 | 0.91x |
| vit-base | 8 | 156.17 / 195.30 | 164.70 / 222.69 | 0.95x |
| whisper-tiny-encoder | 1 | 27.89 / 51.39 | 49.93 / 93.68 | 0.56x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 20.14 / 26.14 | 24.33 / 38.23 | 0.83x |
| phi3.5-mini q4f16 (GQA) | 1 | 51.29 / 81.18 | 61.60 / 96.68 | 0.83x |
| segformer-b0 | 1 | 27.70 / 37.62 | 52.92 / 65.04 | 0.52x |
| segformer-b0 | 8 | 258.26 / 289.88 | 576.64 / 771.84 | 0.45x |
| mobilesam-encoder | 1 | 150.47 / 184.51 | 239.30 / 324.29 | 0.63x |
| raft-large | 1 | 840.88 / 1066.00 | 1967.81 / 2320.38 | 0.43x |
| fasterrcnn-12 (counts) | 1 | 277.81 / 299.95 | 131.98 / 160.99 | 2.10x |
| maskrcnn-12 (counts) | 1 | 297.75 / 330.87 | 128.18 / 150.13 | 2.32x |

There are no CPU changes in this round. §31 did not measure ssd-10, the R-CNNs or raft on the CPU. The R-CNNs run at 2.1-2.3x
of ORT CPU, a new open gap.

### 33.5 Conformance

| | before | after |
|---|---|---|
| node tests, cpu | 1829 / 1884 | 1863 / 1884 |
| node tests, vulkan | 1829 (1778 without CPU fallback) | 1863 (1802 without CPU fallback) |
| extra tests, cpu and vulkan | 170 / 172 | 171 / 172 |

The 21 remaining node failures:
* 9 ImageDecoder tests (an unsupported op);
* 6 Bernoulli and 4 training-mode Dropout tests (random outputs);
* 2 64-bit BitCast tests (64-bit values travel as 32-bit).

The remaining extra failure is WordConvEmbedding, whose stored reference is uninitialized memory. Classification and
fixes are in `docs/onnx-support.md`.

### 33.6 Reproducing

```sh
export SHARPLIGHT_ONNX_MODELS=~/models/onnx ORT_PYTHON=~/venvs/ort/bin/python3   # plus the CUDA libs of bench/onnx/env.sh
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 3 --engines sl-vulkan,sl-vulkan-fast,sl-vulkan-simt,sl-vulkan-rp,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 3 --engines sl-cpu,ort-cpu --runs 20
SHARPLIGHT_MATRIX_PRECISION=fast dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/vit-base-patch16-224.json vulkan --batch 8
~/venvs/ort/bin/python -I bench/onnx/f64_reference.py $SHARPLIGHT_ONNX_MODELS/ref/vit-base-patch16-224.json f64 1
~/venvs/ort/bin/python -I bench/onnx/f64_error_ort.py f64/vit-base-patch16-224.json 1
dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll f64/vit-base-patch16-224.json vulkan --fast-precision   # worstRms
```

### 33.7 Open gaps

* **Strict encoders against TF32 ORT**: vit b8 1.89x, vit b1 1.50x, whisper 1.32x, bert b8 1.25x; convnext b8 1.52x.
  Fast precision closes these gaps (0.86-1.00x). Strict stays at about 1.5x TF32's cost by construction (three
  products).
  * The activation split is still a pre-pass; fusing it into the producing LayerNorm / GELU needs a row-max pass, about
    3.5 % of vit b8.
  * The large-tile occupancy of the split GEMM was not reworked.
* **raft (1.04x fast, 1.25x strict)**:
  * the correlation volume runs on the SIMT bmm (2.2 ms per launch, 3 launches); no tensor-core bmm;
  * about 290 small concats per inference (882 launches, 4.6 ms of GPU time in a fast-precision profile).
* **convnext b8**: the low-K GEMMs and split-K reductions (§31) are unchanged.
* **CPU detection**: Faster / Mask R-CNN at 2.1-2.3x of ORT CPU.
* **qwen2.5 fp16**: Vulkan's error against the f32 copy is 1.5e-3, 6x the CPU backend's in the deep layers.
* **Not done this round**:
  * CPU Winograd on the channels-last layout;
  * the CPU backend's 6-25x larger error than ORT CPU on whisper / vit;
  * a `ConcurrentWorkgroups` query (VK_NV_shader_sm_builtins / VK_AMD_shader_core_properties) instead of the
    per-device-type value.
* **Section number**: the raw data is in `bench/results/s32/`. Rename the section and the folder if another section
  takes 32 first.

## 34. New hardware baseline (RTX PRO 4500 Blackwell 32 GB, 2026-11-15)

The work continues on a new machine. Every number before this section was measured on an RTX PRO 6000 Blackwell Max-Q
(96 GB, 300 W, about 1.8 TB/s); none of them compares to the numbers here. This section re-measures the method of
sections 1-3 on one GPU of the new machine. Base: `develop` at `282dd5b` (measurement only). Raw results:
[`bench/results/s34/`](../bench/results/s34).

### 34.1 Setup

| | |
|---|---|
| GPU | one of three NVIDIA RTX PRO 4500 Blackwell (GB203, 82 SMs, 32 GB GDDR7, 256-bit at 14 Gbps = 896 GB/s, 200 W power limit, boost up to 3090 MHz), driver 595.91, CUDA 13.2. The benchmarks own GPU 1; the other two GPUs ran other agents' tests |
| CPU | 48 threads, shared with other agents (host-side numbers are noisy) |
| Sharplight | `develop` at `282dd5b`, Release, Vulkan provider, `SHARPLIGHT_VK_DEVICES=1` |
| llama.cpp | b11177, builds `runtime.llama-b11177-cuda13` (`CUDA_VISIBLE_DEVICES=1`) and `-vulkan` (`GGML_VK_VISIBLE_DEVICES=1`, NV_coopmat2), `-ngl 99 -fa 1` |
| vLLM | 0.33.0 (`CUDA_VISIBLE_DEVICES=1`, flash-attn backend, CUDA graphs, `VLLM_USE_FLASHINFER_SAMPLER=0`, prefix caching off) on bf16 HF checkpoints |

Method as in sections 2 and 16: `bench/s34_single_all.sh` runs `bench/s16_single.sh` per engine and model
(Sharplight `--bench N`, 3 process runs, best prefill of 4 and median decode step; llama-bench `-r 5`; vLLM
`bench/vllm_single.py`, median of 5). Cells are median (min..max). Sharplight prefills in 512-token chunks like llama-bench
(`-ub 512`); vLLM takes the whole prompt in one step. The model set adds Qwen3-8B Q4_K_M and gpt-oss-20b MXFP4 to the
section 1 set. vLLM runs the bf16 checkpoints (2-3.5x the weight bytes of the GGUFs on OLMoE), so its decode cells are a
reference for the batching engine, not for the kernels. Qwen3.8-27B Q8_0 (29 GB) fits for single-stream runs with a
4352-token context (Sharplight and both llama.cpp builds); vLLM has no FP8 / bf16 checkpoint of it that fits 32 GB.

### 34.2 Single stream

Tokens/s, higher is better; `SL/best` = Sharplight against the fastest other engine.

| model | metric | Sharplight | llama.cpp CUDA | llama.cpp Vulkan | vLLM (bf16) | SL/CUDA | SL/Vulkan | SL/vLLM | SL/best |
|---|---|---|---|---|---|---:|---:|---:|---:|
| gemma3-1b-q8 | pp512 | 33425 (33408..33819) | 37857 (23675..37969) | 33923 (33644..33956) | 34226 (30241..35133) | 0.88x | 0.99x | 0.98x | 0.88x |
| gemma3-1b-q8 | pp2048 | 32164 (25518..32384) | 38958 (36390..39548) | 35449 (33980..35784) | 67142 (66261..67831) | 0.83x | 0.91x | 0.48x | 0.48x |
| gemma3-1b-q8 | tg128 @d0 | 394 (386..400) | 406 (404..408) | 374 (373..375) | 308 (306..317) | 0.97x | 1.05x | 1.28x | 0.97x |
| gemma3-1b-q8 | tg128 @d4096 | 380 (380..382) | 401 (398..402) | 363 (360..366) | 277 (259..280) | 0.95x | 1.05x | 1.37x | 0.95x |
| gpt-oss-20b-mxfp4 | pp512 | 9618 (9472..9750) | 11223 (7406..11728) | 6545 (6373..6703) | - | 0.86x | 1.47x | - | 0.86x |
| gpt-oss-20b-mxfp4 | pp2048 | 9204 (9197..9223) | 11422 (11139..11526) | 6538 (6508..6713) | - | 0.81x | 1.41x | - | 0.81x |
| gpt-oss-20b-mxfp4 | tg128 @d0 | 222 (221..222) | 227 (227..228) | 220 (220..221) | - | 0.98x | 1.01x | - | 0.98x |
| gpt-oss-20b-mxfp4 | tg128 @d4096 | 210 (210..211) | 217 (208..217) | 211 (211..212) | - | 0.97x | 0.99x | - | 0.97x |
| llama3.2-1b-q8 | pp512 | 32722 (32684..32779) | 41426 (30632..42016) | 33853 (33599..34019) | 43650 (42464..44374) | 0.79x | 0.97x | 0.75x | 0.75x |
| llama3.2-1b-q8 | pp2048 | 29508 (29491..29842) | 40894 (39391..42119) | 32790 (31923..33618) | 54582 (53952..55579) | 0.72x | 0.90x | 0.54x | 0.54x |
| llama3.2-1b-q8 | tg128 @d0 | 460 (457..461) | 484 (483..486) | 460 (454..462) | 288 (287..290) | 0.95x | 1.00x | 1.60x | 0.95x |
| llama3.2-1b-q8 | tg128 @d4096 | 398 (397..400) | 443 (441..445) | 416 (408..418) | 278 (275..285) | 0.90x | 0.96x | 1.43x | 0.90x |
| mistral7b-q4km | pp512 | 6067 (6065..6069) | 7129 (6736..7355) | 6558 (6481..6944) | - | 0.85x | 0.93x | - | 0.85x |
| mistral7b-q4km | pp2048 | 5441 (5438..5441) | 6959 (6850..6980) | 6342 (6321..6422) | - | 0.78x | 0.86x | - | 0.78x |
| mistral7b-q4km | tg128 @d0 | 155 (154..155) | 158 (158..158) | 153 (152..154) | - | 0.98x | 1.01x | - | 0.98x |
| mistral7b-q4km | tg128 @d4096 | 134 (133..134) | 141 (141..142) | 137 (136..138) | - | 0.94x | 0.97x | - | 0.94x |
| nemotron30b-q4km | pp512 | 6146 (6145..6150) | 5490 (5073..5548) | 4589 (4571..4600) | - | 1.12x | 1.34x | - | 1.12x |
| nemotron30b-q4km | pp2048 | 5859 (5853..5872) | 5462 (5361..5476) | 4571 (4568..4610) | - | 1.07x | 1.28x | - | 1.07x |
| nemotron30b-q4km | tg128 @d0 | 219 (219..220) | 230 (224..231) | 201 (200..203) | - | 0.95x | 1.09x | - | 0.95x |
| nemotron30b-q4km | tg128 @d4096 | 216 (214..216) | 225 (224..226) | 185 (171..191) | - | 0.96x | 1.17x | - | 0.96x |
| olmoe-q4km | pp512 | 19531 (19515..19546) | 20236 (16961..20829) | 20577 (20158..20861) | 21046 (18995..21483) | 0.97x | 0.95x | 0.93x | 0.93x |
| olmoe-q4km | pp2048 | 17990 (17882..18004) | 18958 (18537..19184) | 18917 (18783..19205) | 39287 (38400..40014) | 0.95x | 0.95x | 0.46x | 0.46x |
| olmoe-q4km | tg128 @d0 | 620 (615..626) | 648 (648..649) | 573 (569..579) | 289 (288..290) | 0.96x | 1.08x | 2.14x | 0.96x |
| olmoe-q4km | tg128 @d4096 | 441 (435..445) | 451 (448..452) | 420 (419..422) | 218 (216..219) | 0.98x | 1.05x | 2.02x | 0.98x |
| qwen27b-q8 | pp512 | 1926 (1926..1933) | 1921 (1810..1926) | 1552 (1548..1558) | - | 1.00x | 1.24x | - | 1.00x |
| qwen27b-q8 | pp2048 | 1822 (1821..1827) | 1904 (1895..1910) | 1537 (1535..1538) | - | 0.96x | 1.19x | - | 0.96x |
| qwen27b-q8 | tg128 @d0 | 27.3 (27.2..27.4) | 27.6 (27.3..27.6) | 27.5 (27.5..27.6) | - | 0.99x | 0.99x | - | 0.99x |
| qwen27b-q8 | tg128 @d4096 | 26.3 (26.3..26.4) | 27.3 (26.9..27.4) | 27.2 (27.2..27.2) | - | 0.96x | 0.97x | - | 0.96x |
| qwen3-0.6b-q8 | pp512 | 46283 (46097..46376) | 49578 (28181..49695) | 40401 (39938..40470) | 32589 (32497..33309) | 0.93x | 1.15x | 1.42x | 0.93x |
| qwen3-0.6b-q8 | pp2048 | 33190 (33166..33197) | 46612 (42769..47347) | 37555 (15386..38216) | 82685 (77660..84008) | 0.71x | 0.88x | 0.40x | 0.40x |
| qwen3-0.6b-q8 | tg128 @d0 | 596 (435..597) | 678 (672..679) | 576 (570..579) | 461 (459..464) | 0.88x | 1.03x | 1.29x | 0.88x |
| qwen3-0.6b-q8 | tg128 @d4096 | 452 (440..453) | 479 (476..480) | 438 (435..441) | 373 (365..374) | 0.94x | 1.03x | 1.21x | 0.94x |
| qwen3-8b-q4km | pp512 | 5698 (5694..5698) | 7059 (6550..7258) | 6216 (5662..6484) | - | 0.81x | 0.92x | - | 0.81x |
| qwen3-8b-q4km | pp2048 | 5020 (5019..5029) | 6895 (6788..6917) | 5862 (5241..6051) | - | 0.73x | 0.86x | - | 0.73x |
| qwen3-8b-q4km | tg128 @d0 | 137 (137..138) | 144 (144..144) | 133 (99..137) | - | 0.95x | 1.03x | - | 0.95x |
| qwen3-8b-q4km | tg128 @d4096 | 119 (118..119) | 128 (128..128) | 111 (106..122) | - | 0.92x | 1.07x | - | 0.92x |

Notes:
* Decode is within 0.88-0.99x of llama.cpp CUDA on every model and 0.96-1.17x of llama.cpp Vulkan. The card's 896 GB/s
  bound decode: the 27B Q8_0 step reads 29 GB in 36.5 ms (0.80 TB/s, 27.3 tok/s against llama.cpp's 27.6).
* Prefill on dense models is the gap: 0.71-0.85x of llama.cpp CUDA (0.72-0.79x on Llama-1B, Qwen3-8B and Qwen3-0.6B at 2048
  tokens). The MoE and hybrid models are close or ahead: OLMoE 0.95-0.97x, the 27B 0.96-1.00x, Nemotron 1.07-1.12x.
* Prefill falls more from 512 to 2048 tokens than on llama.cpp (Mistral -10 % against -2 %): causal attention over the
  growing context costs more (34.4).
* vLLM's 2048-token prefill on the small bf16 models (55-83 k tok/s) is one 2048-row step on cuBLAS; it is 1.7-2.5x
  Sharplight's chunked prefill. At 512 tokens vLLM is at 0.70-1.4x of Sharplight.

### 34.3 Serving

`bench/s16_serve.sh` as in section 28.6 (~250-token random-word prompts, 128 output tokens, ignore_eos, temperature 0,
one server per engine with 64 slots and 512-token contexts, 2 rounds per concurrency; Sharplight has no `ignore_eos`,
but every stream ran to 128 tokens here). The small model is Llama-3.2-1B Q8_0; the largest models that fit all servers
at 64 slots are OLMoE Q4_K_M and gpt-oss-20b MXFP4 (12 GB). The 27B Q8_0 (29 GB) leaves no room for 64 sequences.
Aggregate output tok/s, median of the two rounds (min..max), and median TTFT / mean ITL in ms:

| model | conc | Sharplight | llama.cpp CUDA | llama.cpp Vulkan | SL / CUDA | TTFT p50 SL / CUDA | ITL SL / CUDA |
|---|---:|---:|---:|---:|---:|---|---|
| llama3.2-1b | 1 | 445 (443..446) | 408 (398..417) | 384 (380..389) | 1.09x | 11 / 18 | 2.16 / 2.32 |
| llama3.2-1b | 4 | 1539 (1520..1558) | 678 (665..690) | 579 (576..581) | 2.27x | 24 / 58 | 2.42 / 5.48 |
| llama3.2-1b | 16 | 4069 (4002..4136) | 1034 (1030..1037) | 707 (675..738) | 3.94x | 54 / 230 | 3.45 / 13.79 |
| llama3.2-1b | 64 | 6137 (6091..6183) | 1397 (1353..1442) | 1453 (1442..1463) | 4.39x | 139 / 317 | 8.57 / 38.19 |
| olmoe | 1 | 554 (552..556) | 479 (477..480) | 422 (420..423) | 1.16x | 18 / 49 | 1.67 / 1.71 |
| olmoe | 4 | 979 (979..980) | 614 (595..633) | 461 (439..484) | 1.59x | 34 / 164 | 3.84 / 5.27 |
| olmoe | 16 | 1415 (1403..1427) | 782 (782..782) | 373 (314..431) | 1.81x | 57 / 646 | 10.73 / 15.64 |
| olmoe | 64 | 3051 (3039..3064) | 1323 (1262..1384) | 1202 (1199..1205) | 2.31x | 99 / 824 | 18.50 / 41.79 |
| gpt-oss-20b | 1 | 216 (215..217) | 196 (192..200) | 162 (151..174) | 1.10x | 34 / 52 | 4.39 / 4.72 |
| gpt-oss-20b | 4 | 453 (437..468) | 282 (260..304) | 211 (203..219) | 1.61x | 72 / 136 | 8.38 / 13.23 |
| gpt-oss-20b | 16 | 662 (662..663) | 411 (377..446) | 247 (242..251) | 1.61x | 110 / 501 | 23.10 / 35.52 |
| gpt-oss-20b | 64 | 1850 (1848..1853) | 1032 (1001..1063) | 628 (625..630) | 1.79x | 190 / 941 | 29.76 / 53.86 |

* Sharplight serves 1.1-4.4x llama-server's throughput, with lower TTFT and ITL at every concurrency.
* The MoE models scale poorly from 4 to 16 streams: OLMoE 979 → 1415 tok/s (the 16-stream step at 45 rows takes 10.9 ms
  against 3.9 ms at 10 rows), gpt-oss 453 → 662 (23.5 against 8.6 ms), while Llama-1B goes 1539 → 4069. The batched
  expert matmuls at 16-64 rows are the next serving target.
* vLLM 0.33.0 on the bf16 checkpoints (same loadgen, `--max-num-seqs 64 --max-model-len 512`, `--gpu-memory-utilization`
  0.5 / 0.8; gpt-oss-20b has no local HF checkpoint). Its server start took 12-18 minutes here (the virtualenv sits on a
  slow network filesystem); the measured rounds are unaffected:

| model | conc | vLLM (bf16) | SL / vLLM | TTFT p50 SL / vLLM | ITL SL / vLLM |
|---|---:|---:|---:|---|---|
| llama3.2-1b | 1 | 271 (263..279) | 1.64x | 11 / 16 | 2.16 / 3.62 |
| llama3.2-1b | 4 | 1042 (1040..1045) | 1.48x | 24 / 36 | 2.42 / 3.58 |
| llama3.2-1b | 16 | 3634 (3556..3712) | 1.12x | 54 / 63 | 3.45 / 3.95 |
| llama3.2-1b | 64 | 8192 (7684..8701) | 0.75x | 139 / 190 | 8.57 / 8.15 |
| olmoe | 1 | 266 (265..266) | 2.09x | 18 / 33 | 1.67 / 3.53 |
| olmoe | 4 | 504 (495..512) | 1.94x | 34 / 66 | 3.84 / 7.49 |
| olmoe | 16 | 1116 (1106..1126) | 1.27x | 57 / 95 | 10.73 / 13.65 |
| olmoe | 64 | 2856 (2837..2875) | 1.07x | 99 / 197 | 18.50 / 20.71 |

* Sharplight is ahead of vLLM up to 16 streams and on OLMoE at 64; Llama-1B at 64 streams is the one serving cell
  behind vLLM (0.75x: vLLM's TTFT is higher, but its ITL is 5 % lower and it completes more requests per second).

### 34.4 Ranked gaps (Sharplight / best other engine)

1. **Prefill against vLLM's single-step prefill** on the small bf16 models at 2048 tokens: Qwen3-0.6B 0.40x, OLMoE 0.46x,
   Gemma-3-1B 0.48x, Llama-1B 0.54x (vLLM runs one 2048-row step; Sharplight and llama.cpp chunk at 512).
2. **Dense-model prefill against llama.cpp CUDA**: Qwen3-8B 0.73x / 0.81x (pp2048 / pp512), Mistral-7B 0.78x / 0.85x,
   Llama-1B 0.72x / 0.79x (0.75x of vLLM at 512), gpt-oss-20b 0.81x / 0.86x, Gemma 0.83x / 0.88x. Profile of Qwen3-8B
   pp2048: the Q4_K / Q6_K GEMMs are 85 % of the GPU time at about 93 TFLOPS on the [4096→12288] tiles; causal attention
   was 12 % with `fa`; split-K reductions 2-7 %.
3. **Small-model decode**: Qwen3-0.6B tg@d0 0.88x, Llama-1B tg@d4096 0.90x, Qwen3-8B tg@d4096 0.92x of llama.cpp CUDA.
   The 0.6B step is 1.5-1.6 ms of GPU time for 369 dispatches and 256 barriers: launch- and barrier-bound.
4. **Batched steps at 64 streams**: Llama-1B serving at 0.75x of vLLM (34.3); the MoE models scale poorly from 4 to 16
   streams (batched expert matmuls at 16-64 rows).
5. Everything else is at 0.94x or better.


### 34.5 First changes on the new card

Interleaved `--bench` A/B runs on GPU 1 (`best` prefill of 4 per process, median of 2-3 processes; decode is the median
GPU time per step). Greedy identity: 200-token outputs of a short and a ~600-token prompt plus an ~800-token prompt
prefilled in 128-token chunks (`--batch 128`, so attention runs against a non-empty cache), compared byte for byte.

| change | default | effect | greedy output |
|---|---|---|---|
| Two output rows per workgroup in the split-K matvec (`wip/vulkan-matvec-rows`, `SHARPLIGHT_MV_ROWS=1` restores one) | on | decode step Llama-1B Q8_0 2.10 → 2.04 ms (+4 % tok/s), gpt-oss-20b 4.37 → 4.16 ms (+5 %); bf16 (Llama-1B, Qwen3-0.6B/1.7B) and Q4_K (Qwen3-8B) unchanged | unchanged on 11 models (9 GGUF + 2 bf16) |
| K splits of the tensor-core GEMMs sized by the device's SMs (82 here) instead of 192 tiles (`SHARPLIGHT_GEMM_WAVE=fixed` restores 192) | on | pp2048 Gemma-3-1B +17 %, Llama-1B +9 %, Qwen3-0.6B +7 %, Mistral-7B +2.5 %, Qwen3-8B +2 % | changes Qwen3-8B and gpt-oss-20b (other split = other rounding of the partial sums) |
| Causal single-sequence prefill attention on the `fe` flash kernel (64 queries per workgroup, fragments in registers, blocks read only the keys their last query sees; `SHARPLIGHT_ATTN_FE_CAUSAL=0` restores `fa`) | on | pp2048 Qwen3-0.6B +38 %, Qwen3-8B +7.5 %, Llama-1B +7 %, Mistral-7B +6 %; Gemma (sliding window) keeps `fa` | changes OLMoE and Nemotron-3-Nano (other summation order); unchanged on the 7 others |

With both (pp2048): Qwen3-0.6B 50.7 k tok/s (1.09x of llama.cpp CUDA, was 0.71x), Gemma-3-1B 38.0 k (0.97x, was
0.83x), Llama-1B 35.2 k (0.86x, was 0.72x). With both on, the 27B's chunked-prefill output also changes (not bisected).
The two started as opt-ins because they change greedy output; they are the default since 2026-11-15 by decision, a
deliberate output change. With them on, the `fe` kernel also runs the prompt runs of multi-sequence steps (prompt-only and
mixed steps) with the single-sequence tiles, so the batch decoder's prefill stays bit-identical to the session's and
speculative decoding still reproduces plain decoding (`VerificationRowInvarianceTests`; without this the session and the
batch decoder differed by up to 8e-3 in the logits). No Families reference moved (`SHARPLIGHT_TEST_LARGE=1`): the llama.cpp / transformers
goldens (16-40 tokens, among them Mistral-7B Q4_K_M and gpt-oss-20b MXFP4) and the int-dot accuracy checks (OLMoE,
Mistral-7B, gpt-oss-20b) still pass, so no reference was regenerated.

## 35. ONNX on RTX PRO 4500 vs onnxruntime: concatenations computed in place (2026-11-15)

First ONNX pass on the new machine: an RTX PRO 4500 Blackwell (32 GB) used by this benchmark alone (`SHARPLIGHT_VK_DEVICES`
/ `CUDA_VISIBLE_DEVICES` pin one of the three GPUs), onnxruntime-gpu 1.32 with the CUDA EP (TF32 on, the default, and
off) and the TensorRT EP (f32 engines with TF32 tactics, engine build excluded). Same harness as sections 30-33:
`bench/onnx/parity_bench.py`, every measurement its own process, engines of one model back to back, rounds interleaved.
Raw data is in `bench/results/s35/`:

* `baseline-gpu.jsonl`: a first pass at the start of the round, before the attention-pattern and read-back commits
  below. Its Sharplight times are up to 2x the later ones on the encoders (bert b8 6.55 ms, whisper 3.87 ms) and are
  superseded.
* `gpu-before.jsonl`: one round of all engines including TensorRT, after the merge of develop and before the
  concatenation change.
* `gpu-after.jsonl`: two interleaved rounds of the Sharplight and CUDA EP engines with every change of this section.

### 35.1 What changed

* **Decomposed attention whose probabilities round-trip through Reshapes** (exports that can also return the attention
  probabilities) is now fused into the encoder flash kernel; bert-base and whisper-tiny now run `attention fe`. Between
  the baseline pass and `gpu-before.jsonl` (this commit, the read-back change and the develop merge), GPU time fell
  for bert b8 from 5.69 to 2.05 ms and for the whisper-tiny encoder from 3.60 to 1.87 ms.
* **Large outputs are downloaded on the transfer queue** and read back with several threads; OnnxBench reports the
  read-back time. SegFormer b8 returns 78 MB of logits per run.
* **Concatenation inputs are computed in their output slices** (`PlanConcatSlices`, Vulkan lowering): when every axis
  before the concatenation axis has extent 1 (a dynamic batch compiled for batch 1 counts), each input is a dense,
  contiguous slice of the output, so its producer writes there and the concatenation launches only for the inputs it
  cannot alias (views, graph inputs, segment outputs, values used twice, slices that are not 256-byte aligned). For a
  batch, channel concatenations take Resize outputs as batch-strided slices (the resize kernel got an output batch
  stride, `YS_N`) when the resize output has no other consumer. The slice values share the concatenation output's
  liveness root, and the activation copies of matmul inputs are not shared across such a root. Off with
  `SHARPLIGHT_NO_CONCAT_IN_PLACE=1`.
  * SegFormer-b0's decode head (four upsampled 256-channel maps, 1.36 ms of copy per run at b8) now has no
    concatenation launch: GPU time b8 18.16 → 16.83 ms, b1 2.52 → 2.35 ms.
  * yolov8n drops 1 of its 19 concatenation launches per run and raft 1 of 129: in their other concatenations some
    input does not qualify (a view, a graph input, a slice that is not 256-byte aligned), so the launch stays, now
    copying only those inputs.
* **ssd-12-int8 reference**: see 35.4.

### 35.2 Latency

Wall min latency in ms (minimum over the rounds of each process's min-of-N), batch as given. Sharplight Vulkan and the
CUDA EP columns come from `gpu-after.jsonl` (2 interleaved rounds); TensorRT from `gpu-before.jsonl` (1 round). The
fair pairs are fast against TF32 and strict against strict; "strict / TF32" is the gap left at f32 accuracy.

| model | batch | SL strict | SL fast | ORT TF32 | ORT strict | ORT TensorRT | fast / TF32 | strict / strict | strict / TF32 | fast / TRT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | 1 | 0.80 | 0.60 | 1.21 | 2.52 | 1.52 | 0.49x | 0.32x | 0.67x | 0.39x |
| resnet50 | 8 | 3.51 | 2.68 | 2.93 | 5.34 | 1.99 | 0.92x | 0.66x | 1.20x | 1.35x |
| convnext-tiny | 1 | 1.41 | 1.01 | 1.46 | 1.64 | 1.02 | 0.69x | 0.86x | 0.97x | 0.99x |
| convnext-tiny | 8 | 5.30 | 3.26 | 4.00 | 5.94 | 2.17 | 0.81x | 0.89x | 1.32x | 1.51x |
| efficientnet-lite4 | 1 | 0.71 | 0.61 | 1.26 | 12.17 | 0.69 | 0.48x | 0.06x | 0.56x | 0.89x |
| mobilenetv2 | 1 | 0.36 | 0.32 | 0.69 | 0.69 | 0.35 | 0.46x | 0.52x | 0.52x | 0.90x |
| mobilenetv2 | 8 | 0.94 | 0.86 | 1.05 | 1.62 | 0.55 | 0.82x | 0.58x | 0.89x | 1.56x |
| yolov8n | 1 | 1.26 | 1.09 | 1.73 | 3.58 | 0.92 | 0.63x | 0.35x | 0.73x | 1.19x |
| yolov10n | 1 | 1.12 | 0.97 | 1.72 | 2.57 | 0.73 | 0.56x | 0.44x | 0.65x | 1.33x |
| ssd-10 (NMS) | 1 | 19.46 | 13.90 | 15.30 | 31.58 | 7.65 | 0.91x | 0.62x | 1.27x | 1.82x |
| bert-base (16 tok) | 1 | 1.32 | 0.96 | 1.86 | 1.91 | 1.20 | 0.52x | 0.69x | 0.71x | 0.80x |
| bert-base (16 tok) | 8 | 3.06 | 2.05 | 3.24 | 3.99 | 2.65 | 0.63x | 0.77x | 0.94x | 0.77x |
| vit-base | 1 | 2.78 | 1.41 | 1.60 | 3.12 | 2.26 | 0.88x | 0.89x | 1.73x | 0.62x |
| vit-base | 8 | 11.41 | 5.39 | 5.97 | 12.04 | 5.07 | 0.90x | 0.95x | 1.91x | 1.06x |
| whisper-tiny-encoder | 1 | 2.10 | 1.28 | 1.92 | 3.07 | 1.56 | 0.66x | 0.68x | 1.09x | 0.82x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.56 | 3.56 | 6.06 | 5.83 | - | 0.59x | 0.61x | 0.59x | - |
| phi3.5-mini q4f16 (GQA) | 1 | 6.28 | 6.28 | 4.55 | 4.54 | - | 1.38x | 1.38x | 1.38x | - |
| segformer-b0 | 1 | 3.01 | 2.79 | 3.24 | 4.19 | 2.57 | 0.86x | 0.72x | 0.93x | 1.09x |
| segformer-b0 | 8 | 22.34 | 20.65 | 29.58 | 36.69 | 25.09 | 0.70x | 0.61x | 0.76x | 0.82x |
| mobilesam-encoder | 1 | 7.71 | 6.18 | 9.62 | 12.29 | 4.56 | 0.64x | 0.63x | 0.80x | 1.35x |
| raft-large | 1 | 55.13 | 44.74 | 60.05 | 103.17 | 44.47 | 0.75x | 0.53x | 0.92x | 1.01x |
| fasterrcnn-12 (counts) | 1 | 11.30 | 9.43 | 10.56 | 17.40 | - | 0.89x | 0.65x | 1.07x | - |
| maskrcnn-12 (counts) | 1 | 12.27 | 10.43 | 11.09 | 17.65 | - | 0.94x | 0.70x | 1.11x | - |

### 35.3 Reading the table

* **Fast against TF32**: Sharplight is faster than ORT's default on every model except the phi-3.5 decoder step
  (1.38x, a q4f16 GroupQueryAttention decoder: the LLM kernels' workstream). Range 0.46-0.94x; closest are maskrcnn
  0.94x, resnet50 b8 0.92x, ssd-10 0.91x and vit 0.88-0.90x.
* **Strict against strict**: faster on every model except phi-3.5 (0.06-0.95x; efficientnet-lite4 0.06x, where ORT's
  run with TF32 off is 10x its default). vit b8 is the closest at 0.95x.
* **Strict against TF32**: faster on 14 of 23 rows. The rest: vit 1.73x / 1.91x, phi-3.5 1.38x, convnext b8 1.32x,
  ssd-10 1.27x, resnet50 b8 1.20x, maskrcnn 1.11x, whisper 1.09x, fasterrcnn 1.07x. On ssd-10, two thirds of the GPU time are the
  256-channel 3x3 convs over 150x150 maps (1.08 ms each, about 73 TFLOPS of f16 products: three per multiply-add), so
  the gap is the strict split's cost, not a scheduling loss; fast precision is at 0.91x.
* **TensorRT** stays ahead on the convnets and detectors at batch 8 or high resolution (fast / TRT up to 1.82x on
  ssd-10, 1.51x convnext b8, 1.56x mobilenetv2 b8, 1.35x mobilesam). Sharplight fast beats it on resnet50 b1, bert, vit
  b1, whisper and segformer b8, and ties on convnext b1 and raft. TensorRT could not build Faster / Mask R-CNN (its ONNX
  parser needs `libnvinfer_vc_plugin`, not installed) and fails on the com.microsoft MatMulNBits / GroupQueryAttention
  graphs.
* **Parity**: every strict row passes against the ORT CPU reference except ssd-10 (worst relative RMS 2.1e-2 on the
  NMS outputs, already in the first pass): near-tied boxes switch order after NMS. The fast rows marked as parity
  failures in the raw data are within TF32's error class (section 33.3).
* Changed rows between `gpu-before` (1 round) and `gpu-after` (2 rounds) other than SegFormer (yolov8n strict 1.65 →
  1.26 ms, bert b1 1.78 → 1.32 ms) are run-to-run variation: yolov8n's profiled GPU time is 1.00 ms with and without
  the concatenation change, and bert has no concatenation.

**CPU (indicative)**: one round, min of 10 runs, 8 threads pinned to cores 0-7 of the AMD EPYC 7443P, which other
workloads shared (`cpu.jsonl`; medians up to 1.5x the minima):

| model | batch | SL CPU | ORT CPU | SL / ORT |
|---|---:|---:|---:|---:|
| resnet50 | 1 | 12.7 | 13.1 | 0.97x |
| resnet50 | 8 | 92.7 | 77.1 | 1.20x |
| convnext-tiny | 1 | 30.1 | 24.5 | 1.23x |
| convnext-tiny | 8 | 177.1 | 137.2 | 1.29x |
| mobilenetv2 | 1 | 2.3 | 1.9 | 1.17x |
| mobilenetv2 | 8 | 13.1 | 10.8 | 1.22x |
| yolov8n | 1 | 20.0 | 25.4 | 0.79x |
| bert-base (16 tok) | 1 | 16.3 | 12.8 | 1.28x |
| bert-base (16 tok) | 8 | 49.7 | 54.8 | 0.91x |
| vit-base | 1 | 56.2 | 59.8 | 0.94x |
| vit-base | 8 | 448.1 | 392.5 | 1.14x |
| whisper-tiny-encoder | 1 | 84.5 | 70.6 | 1.20x |
| segformer-b0 | 1 | 47.9 | 54.8 | 0.87x |
| segformer-b0 | 8 | 378.4 | 549.9 | 0.69x |
| fasterrcnn-12 (counts) | 1 | 548.2 | 306.4 | 1.79x |

Sharplight's CPU backend is ahead or even on resnet50 b1, yolov8n, bert b8, vit b1 and segformer, and 1.1-1.3x behind on
the other convnet and encoder rows; Faster R-CNN stays at 1.8x (section 33's CPU detection gap).

### 35.4 ssd-12-int8: ORT's u8 x s8 kernels saturate, Sharplight follows the spec

`SurveyModelTests.Survey("ssd-12-int8", cpu|vulkan)` failed against references generated with ORT 1.32 on this
machine's AMD EPYC 7443P, while Sharplight's CPU and Vulkan backends agreed to 1e-7.

* ORT 1.31.1 and 1.32 give identical outputs, with and without graph optimizations: the requantization did not change.
* The ONNX spec's QLinearConv accumulates exactly in int32, applies the f32 multiplier and rounds half to even before
  saturating. Sharplight computes exactly that on both backends.
* On x86 CPUs without VNNI, ORT's u8 x s8 kernels sum adjacent products with int16 saturation (VPMADDUBSW). An emulation
  of that saturation over channels-last K pairs reproduces ORT's output of the first QLinearConv (input zero point 114,
  7x7x3 int8 weights) exactly; 0.5 % of its outputs (105,961 of 23 M) differ from the spec by up to 5 LSB, which changes
  the selected boxes.
* `bench/onnx/make_survey_refs.py` now builds this reference (`u8weights`) from a copy whose int8 QLinearConv weights
  are stored as uint8 (w + 128, zero point + 128, the same values by the spec). ORT's u8 x u8 kernels widen to 16 bits
  first and match the spec on any x86 CPU. Both backends pass against it (max abs 1.2e-7). Documented in
  `docs/onnx-support.md`.

### 35.5 Tests

ONNX importer tests 181 (0 failed, 28 skipped), Vulkan 1445 (0 failed), CPU 1286 (0 failed), conformance 4197 with 49
failures. The failures are the known unsupported or nondeterministic cases (Bernoulli, ImageDecoder, f64 BitCast,
training Dropout, the two dynamic-int8 extra tests) on both providers, plus 3 Vulkan `test_unique_sorted_with_axis*`
cases (output shape [3, 3] instead of [2, 3]) that fail identically with `SHARPLIGHT_NO_CONCAT_IN_PLACE=1`, so they
predate this change (the Vulkan Unique kernels landed in develop).

### 35.6 Open gaps

* **Strict encoders against TF32**: vit 1.73x (b1) / 1.91x (b8), whisper 1.09x. The three products of the split cost
  about 1.5x by construction; fast precision is at 0.66-0.90x of TF32. The activation split is still a pre-pass that
  the producing LayerNorm / GELU could write.
* **Strict convnets against TF32**: convnext b8 1.32x, resnet50 b8 1.20x, ssd-10 1.27x: the strict split's cost on large
  3x3 / 1x1 convs (73 TFLOPS of f16 products on ssd-10's 256-channel convs).
* **TensorRT**: 1.2-1.8x faster than Sharplight fast on convnets / detectors at batch 8 or high resolution.
* **phi-3.5 q4f16 decoder step**: 1.38x of ORT CUDA (LLM kernels).
* **Concatenations**: batch-strided slices are only taken from Resize; convs, GEMMs and elementwise producers would need
  an output batch stride too. Concatenations whose inputs are views still copy them (raft keeps 117 launches per run).
* **ssd-10 strict parity**: NMS order of near-tied boxes (relative RMS 2.1e-2).
* **Conformance**: 3 Vulkan `test_unique_sorted_with_axis*` failures from develop (35.5).

### 35.7 Reproduce

```bash
. bench/onnx/env.sh   # ORT_PYTHON with onnxruntime-gpu 1.32, CUDA and TensorRT libraries
export SHARPLIGHT_ONNX_MODELS=~/models/onnx SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2
python3 bench/onnx/parity_bench.py run gpu.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-fast,ort-cuda,ort-cuda-strict,ort-trt
BENCH_THREADS=8 python3 bench/onnx/parity_bench.py run cpu.jsonl --rounds 1 --runs 10 --engines sl-cpu,ort-cpu
SHARPLIGHT_NO_CONCAT_IN_PLACE=1 dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/segformer-b0.json vulkan --batch 8 --profile
```

## 36. LLM round 2 on the RTX PRO 4500: 4-warp GEMM tiles, cheaper K-quant fetches, decode combine in the last workgroup (2026-11-15)

Base: `develop` at `9a858ae8`; branch `perf/llm-r2`. Raw results: [`bench/results/s36/`](../bench/results/s36). Same machine, GPU and method as section 34 (`bench/s34_single_all.sh`
with `REPS=2`, `SHARPLIGHT_VK_DEVICES=1`; best prefill of 4 per process, median decode step; "tg @d16" is the decode step
after a 16-token prompt, section 34's "tg128 @d0"). Before and after were measured back to back; the llama.cpp CUDA column
is section 34's. Every change keeps greedy output bit-identical (36.3).

### 36.1 What changed

| change | effect (RTX PRO 4500) |
|---|---|
| 128-token `gemm_cm2` tiles always use the 4-warp layout (4 subgroups of 64x64, 255 registers, 40 KB shared memory: two workgroups per SM). The 8-warp layout (64x32 subgroup tiles), which took grids of 384+ workgroups, stays for the split-f16 form only. Same tile, same K order, same K splits | Q4_K [4096→14336] at 512 rows 632 → 487 us, [4096→12288] 553 → 472 us; bf16 [2048→6144] at 2048 rows 575 → 429 us; Q8_0 [2048→8192] 678 → 572 us. The 4-warp tile was faster at every grid size measured (128-1536 workgroups) |
| `gemm_cm2` on f16 activations: one base address for a thread's activation rows (constant offsets per item instead of eight 64-bit addresses) | [4096→12288] Q4_K 472 → 464 us |
| K-quant weight fetch: the Q4_K / Q5_K block header (d, dmin, 12 scale bytes) as one aligned 16-byte load instead of a word and 2-3 byte loads; 2-byte aligned 16-byte loads (Q6_K codes) as four word loads, or five plus shifts at 2 mod 4, instead of eight halfword loads | Q6_K [12288→4096] at 512 rows 613 → 584 us |
| Split-K decode attention merges its partials in the last workgroup of each row (`decode` with `DEC_FUSED_COMBINE`): each key range publishes its partials (device-scope release) and counts its arrival in a zeroed counter; the last one merges the row with `decode_combine`'s code and resets the counter. Smaller load rounds in the merge keep decode at 99 registers. Used where the last workgroup's share is narrow (query heads per workgroup x DV <= 256); `SHARPLIGHT_ATTN_DECODE_FUSED=0` restores the combine launch | one launch and one barrier less per layer: Qwen3-0.6B 369 → 341 dispatches and 256 → 228 barriers per decode step, 1.48 → 1.42 ms GPU at 16 tokens of context (645 → 662 tok/s), unchanged at 4096; OLMoE +2.7 % at 4096 |

Qwen3-8B pp2048 profile (GPU time per 2048 tokens, serialized launches): 354.9 → 327.5 ms; the Q4_K / Q6_K GEMMs are now at
104-111 TFLOPS on the [4096→12288] and [12288→4096] shapes (93 before).

### 36.2 Single stream, before / after

Tokens/s; `SL/CUDA` against llama.cpp CUDA (section 34).

| model | metric | before | after | after / before | llama.cpp CUDA | SL/CUDA before | SL/CUDA after |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b-q8 | pp512 | 60294 | 60643 | 1.006x | 49578 | 1.22x | 1.22x |
| qwen3-0.6b-q8 | pp2048 | 53086 | 53208 | 1.002x | 46612 | 1.14x | 1.14x |
| qwen3-0.6b-q8 | tg @d16 | 523 | 540 | 1.033x | 678 | 0.77x | 0.80x |
| qwen3-0.6b-q8 | tg @d4096 | 464 | 460 | 0.991x | 479 | 0.97x | 0.96x |
| llama3.2-1b-q8 | pp512 | 40237 | 41092 | 1.021x | 41426 | 0.97x | 0.99x |
| llama3.2-1b-q8 | pp2048 | 37473 | 38221 | 1.020x | 40894 | 0.92x | 0.93x |
| llama3.2-1b-q8 | tg @d16 | 475 | 478 | 1.007x | 484 | 0.98x | 0.99x |
| llama3.2-1b-q8 | tg @d4096 | 421 | 415 | 0.987x | 443 | 0.95x | 0.94x |
| gemma3-1b-q8 | pp512 | 42372 | 42946 | 1.014x | 37857 | 1.12x | 1.13x |
| gemma3-1b-q8 | pp2048 | 39711 | 40266 | 1.014x | 38958 | 1.02x | 1.03x |
| mistral7b-q4km | pp512 | 6592 | 7393 | **1.122x** | 7129 | 0.92x | **1.04x** |
| mistral7b-q4km | pp2048 | 6140 | 6837 | **1.113x** | 6959 | 0.88x | **0.98x** |
| qwen3-8b-q4km | pp512 | 6168 | 7023 | **1.139x** | 7059 | 0.87x | **0.99x** |
| qwen3-8b-q4km | pp2048 | 5674 | 6385 | **1.125x** | 6895 | 0.82x | **0.93x** |
| olmoe-q4km | pp512 | 20898 | 21422 | 1.025x | 20236 | 1.03x | 1.06x |
| olmoe-q4km | pp2048 | 20558 | 21070 | 1.025x | 18958 | 1.08x | 1.11x |
| olmoe-q4km | tg @d4096 | 441 | 453 | 1.027x | 451 | 0.98x | 1.00x |
| gpt-oss-20b-mxfp4 | pp512 | 9908 | 9708 | 0.980x | 11223 | 0.88x | 0.86x |
| gpt-oss-20b-mxfp4 | pp2048 | 9403 | 9449 | 1.005x | 11422 | 0.82x | 0.83x |

Notes:
* The "before" column is today's `develop`, which already includes section 34.5 (SM-sized K splits, the causal `fe` kernel):
  its dense prefill is above section 34.2's table (Qwen3-8B pp2048 5674 against 5020 there).
* Decode of the models whose decode attention keeps the combine launch (Gemma, Mistral, Qwen3-8B, gpt-oss: wider shares
  than 256) runs the same kernels as before; their decode cells are not repeated. The run above was taken before the width
  limit and measured the fused merge on them too: at 4096 tokens Gemma -6.4 %, gpt-oss -2.8 %, Qwen3-8B / Mistral -0.5 %,
  which is why the limit exists.
* Small-model decode cells are noisy on this shared machine (the 0.6B at 16 tokens: 523-640 tok/s across runs of the same
  build); interleaved A/B pairs of the fused merge gave 1.48 → 1.42 ms GPU per step (Qwen3-0.6B, 16 tokens) in every pair.
* gpt-oss prefill (MXFP4 expert GEMMs, `GW_RAW` path) is unchanged within noise.

### 36.3 Greedy output

200-token greedy outputs of four prompts (a short one, a ~600-token one, an ~800-token one prefilled in 128-token chunks
and a ~3000-token one, whose decode runs split-K attention with several key ranges) on 8 models (Qwen3-0.6B, Llama-3.2-1B,
Gemma-3-1B, Qwen3-1.7B bf16, Mistral-7B and Qwen3-8B Q4_K_M, OLMoE, gpt-oss-20b): all 32 byte-identical to `develop`.
Suites (Release, niced, `SHARPLIGHT_VK_DEVICES=1`): Vulkan 1472 passed / 22 skipped, Cpu 1289 / 36 skipped, Families
970 / 368 skipped (large models off), no failures.

### 36.4 Tried and dropped

Measured with `SHARPLIGHT_VK_PROFILE=1` on Qwen3-8B (512-row chunks) and end to end:

* **Where the Q4_K GEMM loses time**: without global fetches the main loop runs [4096→12288] in 283 us (MMA, shared memory
  and barriers: ~180 TFLOPS at this wave fill); with the activation fetch only 314 us; with the weight loads but no decode
  math 426 us; everything 464 us. The weight loads (latency at two workgroups per SM) cost far more than the decode math.
* **Raw Q4_K units decoded at staging** (two 16-byte loads in flight across the MMAs, as MXFP4R does): 496-516 us against
  464 (more live registers at the 255-register cap; decoding between the barriers serializes it).
* **256x64 tiles** (each weight decoded once per 256 rows): 498-508 us.
* **K blocks of 32 with double buffering** on the 4-warp tile: 644 us.
* **Dual gate/up `gemm_cm2`** with the SwiGLU product in the store (B fragments alternate gate / up, twice the column tiles):
  bit-identical, but ±1-2 % end to end (Llama-1B +2 %, Qwen3-8B / Mistral -1.3 / -1.8 %): the separate gate and up GEMMs
  already run concurrently and fill the same waves; the SwiGLU launch it removes is 2 % of the step.
* **Merging two heads side by side** in the fused decode combine changed the rounding on some models (the driver contracts
  the interleaved sums differently), so the merge runs one head at a time.

### 36.5 Open gaps

* Dense prefill against llama.cpp CUDA: Qwen3-8B 0.93x at 2048 tokens, Llama-1B 0.93x, Mistral 0.98x. The GEMM is
  fetch-latency bound at two workgroups per SM within Vulkan's 48 KB of shared memory; the next step is a deeper fetch
  pipeline (or int8 MMQ, which changes output and stays opt-in).
* Small-model decode: Qwen3-0.6B at 16 tokens 0.80x of llama.cpp CUDA; barriers cost about 30 % of its step
  (`SHARPLIGHT_VK_NO_BARRIERS=1`: 1.47 → 1.02 ms). Next: rope / q-k norm / cache writes inside the decode attention kernel
  (one barrier and three launches per layer), the residual add and norm in the producing matvec.
* gpt-oss prefill 0.83-0.86x (MXFP4 expert GEMMs), batched steps at 16-64 streams (not worked on in this round).

## 37. ONNX round 2 on the RTX PRO 4500: convs against TensorRT, phi-3.5 at ORT's level, Unique along an axis (2026-11-15)

Same machine, harness and engines as section 35 (`bench/onnx/parity_bench.py`, one process per measurement, GPU 2
pinned with `SHARPLIGHT_VK_DEVICES` / `CUDA_VISIBLE_DEVICES`, onnxruntime-gpu 1.32). Raw data in `bench/results/s37/`:

* `gpu-after.jsonl`: two interleaved rounds of the Sharplight (strict, fast) and CUDA EP (TF32, strict) engines with
  every change of this section and develop merged (the public API pass).
* `trt.jsonl`: one round of the TensorRT EP (f32 engines, TF32 tactics, engine build excluded) on the convnet and detector
  rows; the other TensorRT cells are section 35's.

### 37.1 What changed

* **Tap-major conv reduction** (`conv_cm`, split-f16 forms: strict and fast precision). The implicit GEMM now reduces over
  k = (kh * KW + kw) * CG + c instead of c * KH * KW + kh * KW + kw; the constant split weights are reordered on the host when
  they are split (cached per tap count). With CG % 8 == 0 a staged item of 8 k is 8 channels of one input pixel: one address
  and one bounds check per item instead of a k / (KH * KW) decomposition and a check per value. Stems with few channels
  (3 -> 64 7x7) decompose once per item and step the position per value. The B gather was the bound of the fast form:
  resnet50 b8's 256-channel 3x3 convs 74 -> 43 us. `SHARPLIGHT_CONV_NO_TAP_MAJOR=1` restores the old order.
* **Scaling after the products**: the per-block scale and split of the im2col values moved from the global-load step to
  just before the shared-memory store, after the previous step's products, so the loads stay in flight during the MMAs.
* **Fast precision keeps block scales across K steps**: a 16-pixel block keeps its power-of-two scale while a step's largest
  magnitude scales into [2^3, 2^15) (no overflow; values down to 2^-18 of the step's largest stay normal halves); otherwise
  the scale moves and the block's accumulators are multiplied by the power-of-two factor. The products then accumulate on
  the tensor cores across steps, as TF32's do, instead of a per-step partial sum and a rescaled add. Strict precision keeps
  its per-step partial sums (they bound the tensor cores' truncating accumulation).
* **Narrow convs on the tensor cores**: conv_cm now takes reduction lengths and output channels per group from 16 (were 64
  and 32). Expansions from 16-24 channels over large maps and 16-channel detector stems are memory bound and the padded
  tile beats the SIMT conv_gemm (mobilenetv2 b8 fast 0.87 -> 0.58 ms).
* **Depthwise convs of width 5 and more** compute up to 4 outputs of a row per thread, loading each input row segment and
  weight once (convnext-tiny b8's 96-channel 7x7 depthwise 72 -> 35 us). `SHARPLIGHT_CONV_DW_SINGLE=1` keeps one output.
* **Q4_0 decode** reads a group's 8 code bytes as four 16-bit loads instead of eight byte loads (bit-identical). This is the
  f32-activation matvec of ONNX MatMulNBits weights (accuracy_level 0): phi-3.5-mini q4f16's [3072 -> 3072] matvec
  12.3 -> 10.2 us. The helper (`load8_q4_0`) is shared with the LLM kernels; the change is local to it and exact.
* **Vulkan Unique along an axis**: the lowering ignored the IR's `slices` attribute and ran the flattened form, so the three
  `test_unique_sorted_with_axis*` conformance cases returned [3, 3] instead of [2, 3]. Slices are now compared element by
  element and sorted lexicographically.

Tried and dropped (measured, no gain): a 64-deep K step in conv_cm (TBK = 64; no change, the step count is not the
bound), Q4_0 matvec units of 16 values (one load of 8 code bytes for both nibbles: 1.4x slower, fewer loads in flight),
4 rows per workgroup for Q4 matvecs (phi -3 %, qwen2.5 wall +4 %), fusing a decoder's Q / K / V MatMulNBits into one
matvec (no GPU-time change: the three matvecs already overlap), several rows per workgroup in `split_rows` (convnext b8
fast +5 %).

### 37.2 Latency

Wall min latency in ms (minimum over rounds of each process's min-of-N). "Before" is section 35's `gpu-after.jsonl`;
ORT columns from this round; TensorRT from `trt.jsonl` where measured (convnets, detectors), else section 35.

| model | batch | SL strict | SL fast | ORT TF32 | ORT strict | ORT TensorRT | fast / TF32 | strict / strict | strict / TF32 | fast / TRT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | 1 | 0.80 → 0.71 | 0.60 → 0.53 | 1.18 | 2.52 | 1.06 | 0.45x | 0.28x | 0.60x | 0.50x |
| resnet50 | 8 | 3.51 → 3.06 | 2.68 → 2.09 | 2.94 | 5.36 | 2.00 | 0.71x | 0.57x | 1.04x | 1.04x |
| convnext-tiny | 1 | 1.41 → 1.39 | 1.01 → 0.99 | 1.47 | 1.65 | 0.72 | 0.67x | 0.84x | 0.95x | 1.38x |
| convnext-tiny | 8 | 5.30 → 4.77 | 3.26 → 3.03 | 4.01 | 5.97 | 2.20 | 0.76x | 0.80x | 1.19x | 1.38x |
| efficientnet-lite4 | 1 | 0.71 → 0.62 | 0.61 → 0.53 | 1.24 | 12.20 | 0.69 | 0.43x | 0.05x | 0.50x | 0.77x |
| mobilenetv2 | 1 | 0.36 → 0.31 | 0.32 → 0.27 | 0.71 | 0.69 | 0.33 | 0.38x | 0.45x | 0.44x | 0.83x |
| mobilenetv2 | 8 | 0.94 → 0.70 | 0.86 → 0.60 | 1.05 | 1.63 | 0.56 | 0.57x | 0.43x | 0.67x | 1.08x |
| yolov8n | 1 | 1.26 → 1.07 | 1.09 → 0.86 | 1.71 | 3.58 | 1.00 | 0.50x | 0.30x | 0.63x | 0.87x |
| yolov10n | 1 | 1.12 → 0.98 | 0.97 → 0.82 | 1.72 | 2.56 | 0.72 | 0.48x | 0.39x | 0.57x | 1.14x |
| ssd-10 (NMS) | 1 | 19.46 → 14.90 | 13.90 → 9.19 | 15.30 | 31.72 | 7.63 | 0.60x | 0.47x | 0.97x | 1.20x |
| bert-base (16 tok) | 1 | 1.32 → 1.33 | 0.96 → 0.96 | 1.87 | 1.85 | 1.20 | 0.51x | 0.72x | 0.71x | 0.80x |
| bert-base (16 tok) | 8 | 3.06 → 3.01 | 2.05 → 2.06 | 3.19 | 3.91 | 2.65 | 0.65x | 0.77x | 0.94x | 0.78x |
| vit-base | 1 | 2.78 → 2.76 | 1.41 → 1.42 | 1.61 | 3.12 | 2.26 | 0.88x | 0.88x | 1.71x | 0.63x |
| vit-base | 8 | 11.41 → 10.72 | 5.39 → 4.72 | 5.97 | 12.05 | 5.07 | 0.79x | 0.89x | 1.80x | 0.93x |
| whisper-tiny-encoder | 1 | 2.10 → 1.95 | 1.28 → 1.25 | 1.89 | 3.08 | 1.56 | 0.66x | 0.63x | 1.03x | 0.80x |
| qwen2.5-0.5b q4 (MatMulNBits) | 1 | 3.56 → 3.49 | 3.56 → 3.48 | 6.03 | 5.81 | - | 0.58x | 0.60x | 0.58x | - |
| phi3.5-mini q4f16 (GQA) | 1 | 6.28 → 4.70 | 6.28 → 4.71 | 4.54 | 4.57 | - | 1.04x | 1.03x | 1.04x | - |
| segformer-b0 | 1 | 3.01 → 2.86 | 2.79 → 2.69 | 3.28 | 4.23 | 2.18 | 0.82x | 0.68x | 0.87x | 1.23x |
| segformer-b0 | 8 | 22.34 → 21.53 | 20.65 → 19.84 | 29.83 | 37.04 | 25.64 | 0.67x | 0.58x | 0.72x | 0.77x |
| mobilesam-encoder | 1 | 7.71 → 7.55 | 6.18 → 5.99 | 9.64 | 12.41 | 4.55 | 0.62x | 0.61x | 0.78x | 1.32x |
| raft-large | 1 | 55.13 → 46.49 | 44.74 → 36.15 | 59.69 | 102.59 | 44.47 | 0.61x | 0.45x | 0.78x | 0.81x |
| fasterrcnn-12 (counts) | 1 | 11.30 → 10.53 | 9.43 → 7.91 | 10.55 | 17.43 | - | 0.75x | 0.60x | 1.00x | - |
| maskrcnn-12 (counts) | 1 | 12.27 → 10.98 | 10.43 → 8.48 | 10.76 | 17.62 | - | 0.79x | 0.62x | 1.02x | - |

### 37.3 Reading the table

* **Convs**: fast precision went down 6-34 % on every conv model (resnet50 b8 2.68 → 2.09 ms, mobilenetv2 b8 0.86 →
  0.60, ssd-10 13.90 → 9.19, raft 44.7 → 36.2, Faster / Mask R-CNN 9.4 / 10.4 → 7.9 / 8.5); strict went down with them
  (ssd-10 19.5 → 14.9, resnet50 b8 3.51 → 3.06), so strict now beats ORT's TF32 default on ssd-10 (0.97x) and ties it on
  resnet50 b8 (1.04x) and Faster R-CNN (1.00x).
* **TensorRT**: fast is now ahead of or level with TensorRT on resnet50 (0.50x b1, 1.04x b8), efficientnet-lite4 0.77x,
  mobilenetv2 b1 0.77x, yolov8n 0.94x, raft 0.81x, segformer b8 0.79x and the encoders. Still behind: convnext 1.38x
  (b1 and b8: LayerNorm, the activation split pre-pass of its GEMMs and the residual add of a transposed operand, not the
  convs), mobilesam 1.31x, ssd-10 1.20x, yolov10n 1.13x, mobilenetv2 b8 1.09x, segformer b1 1.05x. TensorRT's own
  numbers move between runs by tactic choice (convnext b1 1.02 → 0.72 ms, resnet50 b1 1.52 → 1.06 ms against
  section 35).
* **phi-3.5-mini q4f16 decode step**: 6.28 → 4.70 ms, 1.03-1.04x of ORT CUDA (was 1.38x). Most of it is the Q4_0 group
  decode (GPU time 4.55 → 4.03 ms), the rest develop's changes since section 35. About 0.35 ms of the wall time is the
  read-back of the 65 outputs (64 present K / V tensors). With int8 activations (`--reduced-precision`, which the model
  does not allow: accuracy_level 0) the int-dot matvecs reach 4.26 ms.
* **Fast against TF32**: faster than ORT's default on every row but phi-3.5 (1.04x); range 0.38-0.89x.
* **Strict against strict**: 0.05-0.89x on every row but phi-3.5 (1.03x).
* **Parity**: every strict row passes except ssd-10, as before. Its relative RMS of 2.1e-2 is tie order: all 200
  reference detections are in Sharplight's output (same label, score within 2e-4, box within 2e-3), 198 at the same
  position; detections 138 and 139 (label 1, scores 0.06941171 and 0.069411665 in the reference) swap. Not a bug.

### 37.4 Tests

After the develop merge: ONNX importer tests 181 (0 failed, 28 skipped), conformance 4197 with 46 failures, Vulkan 1472 (0 failed), CPU 1289 (0 failed). The conformance failures
are the known unsupported or nondeterministic cases (Bernoulli, ImageDecoder, f64 BitCast, training Dropout, the two
dynamic-int8 extra tests) on both providers; the three Vulkan `test_unique_sorted_with_axis*` failures of section 35 are
fixed.

### 37.5 Open gaps

* **convnext against TensorRT** (1.38x): the conv share is small now; the GEMMs' activation split is a separate pass
  (`split_rows`, about 0.45 ms of 3.0 at b8) that the producing channel-strided LayerNorm could write, and the residual add
  reads the block output through a transpose in its own launch.
* **ssd-10 / mobilesam / yolov10n against TensorRT** (1.1-1.3x): the fast conv runs at about 45-55 TFLOPS on 3x3 convs
  against the GEMM's 96; with the im2col loads removed it reaches about 60, so the shared staging (single buffer, two
  barriers per step) and the f32 NCHW gather are the next bounds (NHWC activations, a double-buffered conv pipeline).
* **phi-3.5**: the f32-activation Q4_0 matvec reads about 520 GB/s ([3072 → 3072]); decomposed attention and rotary run
  as several small launches (MultiHeadAttention on f16 operands does not take the fused attention, which needs f32).
* **Strict encoders against TF32**: unchanged (vit 1.7-1.8x), by construction.

### 37.6 Reproduce

```bash
. bench/onnx/env.sh   # TRT_LIB pointing at the TensorRT 10 libraries for ort-trt
export SHARPLIGHT_ONNX_MODELS=~/models/onnx SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2
python3 bench/onnx/parity_bench.py run gpu-after.jsonl --rounds 2 --engines sl-vulkan,sl-vulkan-fast,ort-cuda,ort-cuda-strict
python3 bench/onnx/parity_bench.py run trt.jsonl --rounds 1 --engines ort-trt \
  --models "resnet50,convnext-tiny,efficientnet-lite4,mobilenetv2,yolov8n,yolov10n,ssd-10 (NMS),segformer-b0,mobilesam-encoder"
SHARPLIGHT_CONV_NO_TAP_MAJOR=1 dotnet bench/onnx/OnnxBench/bin/Release/net11.0/OnnxBench.dll $SHARPLIGHT_ONNX_MODELS/ref/resnet50-v1-12.json vulkan --batch 8 --fast-precision --profile
```

## 38. Prefill against vLLM on identical weights: whole-pass prefill up to 2048 rows (2026-11-16)

Base: `refactor/sharplight-package` (= `develop` with `Sharplight.Defaults` renamed to `Sharplight`) plus `develop` up to
`05b1c9cd`; branch `perf/llm-r3`. Raw results: [`bench/results/s38/`](../bench/results/s38), script
[`bench/s38_all.sh`](../bench/s38_all.sh), tables [`bench/s38_summarize.py`](../bench/s38_summarize.py).

### 38.1 Setup

Earlier vLLM comparisons (sections 2, 34) ran vLLM on bf16 checkpoints and Sharplight on Q8_0 / Q4_K_M GGUFs. Here both
engines load **the same Hugging Face checkpoints**: Sharplight through its HF importer (bf16 weights, FP8 E4M3 with 128x128
block scales for the FP8 checkpoint), vLLM natively.

| | |
|---|---|
| GPU | one RTX PRO 4500 Blackwell (32 GB, 896 GB/s, 200 W), driver 595.91, CUDA 13.2; GPU 2 owned by this run (`SHARPLIGHT_VK_DEVICES=2`, `CUDA_VISIBLE_DEVICES=2`) |
| models | `Qwen/Qwen3-0.6B`, `meta-llama/Llama-3.2-1B-Instruct`, `Qwen/Qwen3-8B` (bf16) and `Qwen/Qwen3-8B-FP8` (E4M3, 128x128 block scales) |
| Sharplight | Release, Vulkan provider. "before" = the fixed 512-row passes of earlier sections (`--batch 512`, server `--prefill-chunk 512`); "after" = this section's defaults (38.2) |
| vLLM | 0.33.0, flash-attn backend, CUDA graphs, prefix caching off, `--gpu-memory-utilization 0.85`; offline runs take up to 8192 tokens per step (its default for the `LLM` class), so pp8192 is one step |
| single stream | Sharplight `--bench N --ctx N+256`, 2 process runs (best prefill of 4 per run, best of the runs; median decode step). vLLM `bench/vllm_single.py --depth 8192` (median of 5). "tg @d16 / @d8192" = decode after 16 / 8192 tokens |
| serving | one server per engine and model (16 slots, 4096-token context), `bench/loadgen.py`: 2000-token random-word prompts, 16 output tokens, ignore_eos, 3 requests per worker, 2 rounds; TTFT p50 (median of the rounds) and output tok/s (which, with 16 output tokens per 2000-token prompt, measures prefill throughput) |

### 38.2 What changed

| change | default |
|---|---|
| **Rows per forward pass chosen per model** (`LlmModel.BatchTokens`): `LlmLoadOptions.MaxBatchTokens` defaults to 0, which takes 2048 rows (at most the context length, at least 512) and halves them down to 512 while the estimated single-sequence footprint (`PlacementPlanner.Estimate`: weights, KV cache, activations bound) exceeds the devices' available memory; CPU-only placements keep 512. The CLI / server `--batch` and `PrecompileOptions.BatchTokens` default to 0 too (precompile resolves it on the compiling device; other graphs keep 512). An explicit value is used as given | on |
| **Serving: a prompt takes whole passes** (`EngineOptions.PrefillChunk` 0 = the step's token budget, was 512 rows per sequence and step; server `--prefill-chunk`). Decode rows are still scheduled first; the budget is the decoder's rows (2048) | on |

Why it helps: four 512-row passes cost more than one 2048-row pass at every layer: four times the launches and barriers, GEMM
grids of a quarter of the size (K splits at 512 rows, one at 2048), and the attention of the later chunks re-reads the
cached keys in separate passes. Measured on the CLI with explicit `--batch` (Qwen3-0.6B, Llama-3.2-1B bf16):

| model | prompt | 512 rows | 1024 | 2048 | 4096 / 8192 |
|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | 2048 | 53.2 k tok/s | | 75.7 k | 75.5 k (8192) |
| Qwen3-0.6B | 8192 | 35.0 k | | 45.4 k | 44.4 k (8192) |
| Llama-3.2-1B | 2048 | 37.6 k | 44.4 k | 44.3 k | 44.2 k (4096) |
| Llama-3.2-1B | 8192 | 29.3 k | 34.3 k | 34.9 k | 35.1 k (4096) |

The gain saturates at 1024-2048 rows, so 2048 is the cap (larger passes only cost activation memory: the session's logits and
FFN buffers grow with the rows).

### 38.3 Single stream, before / after

Tokens/s, higher is better.

| model | metric | Sharplight before | Sharplight after | vLLM | SL/vLLM before | SL/vLLM after |
|---|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | pp512 | 59922 | 59707 | 32756 | 1.83x | 1.82x |
| qwen3-0.6b | pp2048 | 53100 | 75421 | 81080 | 0.65x | 0.93x |
| qwen3-0.6b | pp8192 | 34777 | 45305 | 60494 | 0.57x | 0.75x |
| qwen3-0.6b | tg@d16 | 452 | 460 | 463 | 0.98x | 0.99x |
| qwen3-0.6b | tg@d8192 | 292 | 292 | 310 | 0.94x | 0.94x |
| llama3.2-1b | pp512 | 40269 | 40419 | 40243 | 1.00x | 1.00x |
| llama3.2-1b | pp2048 | 37638 | 44430 | 54887 | 0.69x | 0.81x |
| llama3.2-1b | pp8192 | 29222 | 34850 | 49507 | 0.59x | 0.70x |
| llama3.2-1b | tg@d16 | 292 | 295 | 288 | 1.01x | 1.02x |
| llama3.2-1b | tg@d8192 | 257 | 258 | 266 | 0.96x | 0.97x |
| qwen3-8b | pp512 | 6695 | 6703 | 8440 | 0.79x | 0.79x |
| qwen3-8b | pp2048 | 6141 | 6464 | 8977 | 0.68x | 0.72x |
| qwen3-8b | pp8192 | 5215 | 5555 | 7859 | 0.66x | 0.71x |
| qwen3-8b | tg@d16 | 53 | 53 | 52 | 1.03x | 1.03x |
| qwen3-8b | tg@d8192 | 49 | 49 | 48 | 1.02x | 1.02x |
| qwen3-8b-fp8 | pp512 | 10597 | 11231 | 15080 | 0.70x | 0.74x |
| qwen3-8b-fp8 | pp2048 | 10135 | 11136 | 15378 | 0.66x | 0.72x |
| qwen3-8b-fp8 | pp8192 | 7889 | 8791 | 12642 | 0.62x | 0.70x |
| qwen3-8b-fp8 | tg@d16 | 90 | 90 | 87 | 1.04x | 1.04x |
| qwen3-8b-fp8 | tg@d8192 | 78 | 78 | 78 | 1.01x | 1.00x |

Notes:
* Prefill at 2048 tokens goes from 0.65x to 0.93x of vLLM on Qwen3-0.6B (+42 %) and from 0.69x to 0.81x on Llama-1B
  (+18 %); the 8B models gain 5-10 % (their GEMMs fill the GPU at 512 rows already). At 512 tokens nothing changes:
  Sharplight is 1.8x vLLM on the 0.6B and level on Llama-1B.
* The 8B models are GEMM bound: 0.72x of vLLM in bf16 and FP8 at 2048 tokens. Counting 2 x 7.6 G matmul FLOPs per token, vLLM runs the
  bf16 8B at about 140 TFLOPS and the FP8 one at about 235; Sharplight at about 100 and 170.
* Long prompts lose more than vLLM: at 8192 tokens 0.69-0.75x (0.6B 0.75x, Llama 0.70x, 8B 0.71x, FP8 0.70x). The causal
  attention over 2048-row passes against up to 8192 cached keys is the next lever.
* Decode on identical weights is level with vLLM: 0.94-1.04x (bf16 8B 53 against 52 tok/s, FP8 90 against 87).

### 38.4 Serving: time to first token

2000-token prompts, 16 output tokens; TTFT p50 in ms (lower is better) and output tok/s (higher is better).

| model | streams | TTFT p50 SL before | TTFT p50 SL after | TTFT p50 vLLM | tok/s SL before | tok/s SL after | tok/s vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 1 | 42 | 30 | 37 | 201 | 244 | 218 |
| qwen3-0.6b | 16 | 595 | 129 | 130 | 320 | 423 | 565 |
| llama3.2-1b | 1 | 60 | 53 | 48 | 140 | 151 | 157 |
| llama3.2-1b | 16 | 738 | 207 | 246 | 258 | 295 | 414 |
| qwen3-8b | 1 | 343 | 324 | 232 | 25 | 26 | 30 |
| qwen3-8b | 16 | 4360 | 859 | 808 | 44 | 46 | 66 |
| qwen3-8b-fp8 | 1 | 211 | 191 | 133 | 42 | 44 | 51 |
| qwen3-8b-fp8 | 16 | 2732 | 511 | 556 | 71 | 77 | 111 |

* At 16 streams TTFT p50 falls 3.6-5.3x (Qwen3-0.6B 595 → 129 ms, Llama-1B 738 → 207, Qwen3-8B 4360 → 859, FP8 2732 →
  511) and is now at vLLM's level (0.84-1.06x of vLLM's TTFT). Before, every admitted prompt advanced 512 rows per step, so
  all 16 finished their prefill together at the end; now each step prefills one prompt in a whole 2048-row pass and its
  first token comes out right away, as vLLM's scheduler does.
* Single-stream TTFT: 0.6B 30 ms against vLLM's 37; Llama-1B 53 against 48; the 8B models 1.4x vLLM's (GEMM bound, 38.3).
* Prefill throughput at 16 streams (output tok/s) is 0.69-0.75x of vLLM's: the same prefill kernel gap as in 38.3.

### 38.5 Greedy output

200-token greedy outputs of a short, a ~600-token and a ~3000-token prompt, `--batch 512` against the new default on the same
build: byte-identical on Qwen3-0.6B, Llama-3.2-1B and Qwen3-8B (bf16 HF), Gemma-3-1B Q8_0 (sliding window) and OLMoE Q4_K_M.
Qwen3-8B-FP8 differs on the ~600-token prompt after about 40 generated words ("read through the notes" against "read
through the provided text"; the other two prompts are identical). Qwen3-8B Q4_K_M differs on the two long prompts after 120-150 generated words (near-tie flips, both continuations coherent): its
[4096 → N] GEMMs take K splits at 512 rows and none at 2048, a summation-order change of the kind accepted for section 34.5.
`--batch 512` restores the previous output. The Families references (llama.cpp / transformers goldens) are unchanged (38.6).

### 38.6 Tests

Release, niced, `SHARPLIGHT_VK_DEVICES=2`, after merging `develop` at `05b1c9cd`: Vulkan 1487 passed / 22 skipped, Cpu 1305 / 36
skipped, Serving 294 / 13 skipped, Families 975 / 364 skipped (large models off) with 2 failures: `tiny-random-mistral-3 HF`
on both CPU and Vulkan (generates other tokens from the first one on). That case loads with an explicit `MaxBatchTokens =
256` and fails on the CPU provider as well, so it does not involve this change; it passed in section 36's run and came in
with the `develop` merge (not bisected here).

### 38.7 Next levers

* **bf16 / FP8 GEMM at large M** (38.3: about 100 / 170 TFLOPS against vLLM's 140 / 235 on the 8B): the 8B's prefill and its
  TTFT are GEMM bound.
* **Causal attention at long context**: pp8192 at 0.69-0.75x of vLLM against 0.72-0.93x at 2048.
* **Per-layer launches and barriers at prefill** on the small models (Llama-1B pp2048 0.81x).

### 38.8 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2 VLLM_VENV=~/venvs/vllm
bench/s38_all.sh vllm
TAG=before SL_ARGS="--batch 512" SRV_ARGS="--prefill-chunk 512" bench/s38_all.sh sharplight
bench/s38_all.sh sharplight after
python3 bench/s38_summarize.py bench/results/s38 before,after
```

## 39. LLM round 4 against vLLM: grouped GEMM tile order, dual bf16 gate / up GEMM, register softmax in prefill attention (2026-11-17)

Base: `develop` at `31717994` ("before"), branch `perf/llm-r4` ("after"). Same method, checkpoints and vLLM 0.33.0 as section 38,
on GPU 1 (`SHARPLIGHT_VK_DEVICES=1`, `CUDA_VISIBLE_DEVICES=1`); checkpoints copied to local disk. Raw results:
[`bench/results/s39/`](../bench/results/s39) (`bench/s38_all.sh` with `OUT_DIR=bench/results/s39`, tables from
`bench/s38_summarize.py`). Before and after ran back to back on the same GPU; no other process used it during the runs.

### 39.1 What changed

| change | effect (RTX PRO 4500) |
|---|---|
| **Grouped tile order** (`grouped_tile` in common.slang; gemm_cm2 and the int8 GEMM, launches without K splits, weight matrices of 48 MB or more): consecutive workgroups walk 16 token tiles of one column tile before the next. Launched column tiles first, the ~164 resident 128x128 workgroups covered one or two token tiles and every weight tile, so a weight matrix larger than L2 was streamed from DRAM once per token tile (bf16 [4096 → 12288] at 2048 rows: 16 x 100 MB, about 1.8 ms of DRAM time for a 1.9 ms GEMM). Same per-output arithmetic, bit-identical. `SHARPLIGHT_GEMM_GROUP=0` restores the launch order | bf16 [4096 → 12288] at 2048 rows 1.91 → 1.68 ms (108 → 123 TFLOPS), at 8192 rows 98 → 125 TFLOPS; [12288 → 4096] 97 → 100 |
| **Dual gemm_cm2 for f16 / bf16 gate / up pairs** (`GraphLowerer.DualGemm.cs`): one launch reads the f16 activations once, alternates the 16-row fragments of both matrices in the B tile and stores `silu(gate) * up` (the fused elementwise group as `epi2`), for 128-token tiles where the separate GEMMs would not split K. Bit-identical to the separate launches (`DualGemmTests`). `SHARPLIGHT_GEMM2_DUAL=0` disables it | gate / up / SiLU·mul of Qwen3-8B at 2048 rows 3.70 → 3.27 ms per layer |
| **Register softmax in the prefill / encoder attention kernel `fe`** (FE_REG, devices with the MMA fragment layout, verified by the kernel): the online softmax runs on the score accumulators (row maxima and sums over the four lanes of a row) and the probabilities become the A operands of P·V directly, so the key loop has no score / probability staging in shared memory and no subgroup barriers; 64-key tiles, 128 queries (8 subgroups) per workgroup, f16 K/V staged without the f32 round trip. `SHARPLIGHT_ATTN_FE_REG=0` restores the staged softmax | Qwen3-0.6B pp8192 attention 92.7 → 70.1 ms of 175 ms GPU time |
| **Split-f16 flash attention safe for any magnitude** (`fa` and `fe` with FA_SPLIT, strict-precision f32 K/V): each query row, each key row of a tile and each V tile is scaled by a power of two into [2^14, 2^15) before the hi / lo split, and the scores and tile products are unscaled exactly; within f16's range the results are unchanged (power-of-two scaling commutes with the rounding). Queries / keys / values beyond f16's range used to give NaN. XTTS's Perceiver (latents of 1e10) drops its `f32_products` opt-out, which is removed (`FullAttentionTests.SplitHalfLargeMagnitudes`). Qwen-Image's / FLUX's per-block choice of f16 products (`F16Safe`, bound 50) stays: it guards the rounding of single f16 products (scores 2 % off in block 0), which no scaling fixes, and its fallback is this split kernel | strict-precision attention only |

What did not help (measured, not kept): the workgroup-scope coopmat2 GEMM with grouped order (87-109 TFLOPS, below gemm_cm2), 32-value K
blocks double buffered with two register prefetch sets (register spills, 55-69 TFLOPS), 8-warp 128x128 tiles (120 vs 128), double-buffered
32-key tiles in `fe` (slower than single-buffered 64-key tiles). Diagnostic: with the global loads and stores removed, gemm_cm2 runs at
about 196 TFLOPS, so the remaining gap is the single-buffered staging that the 48 KB of shared memory per workgroup forces on
128x128x64 tiles.

### 39.2 Single stream

Tokens/s, higher is better (Sharplight: best of 2 process runs x 4 prefills; vLLM: median of 5).

| model | metric | Sharplight before | Sharplight after | vLLM | SL/vLLM before | SL/vLLM after |
|---|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | pp512 | 58880 | 58426 | 32688 | 1.80x | 1.79x |
| qwen3-0.6b | pp2048 | 74358 | 82870 | 82156 | 0.91x | 1.01x |
| qwen3-0.6b | pp8192 | 44476 | 53016 | 58706 | 0.76x | 0.90x |
| qwen3-0.6b | tg@d16 | 459 | 460 | 461 | 0.99x | 1.00x |
| qwen3-0.6b | tg@d8192 | 285 | 294 | 308 | 0.92x | 0.95x |
| llama3.2-1b | pp512 | 39895 | 41995 | 44578 | 0.89x | 0.94x |
| llama3.2-1b | pp2048 | 43700 | 48626 | 53097 | 0.82x | 0.92x |
| llama3.2-1b | pp8192 | 34146 | 40151 | 47421 | 0.72x | 0.85x |
| llama3.2-1b | tg@d16 | 293 | 294 | 286 | 1.03x | 1.03x |
| llama3.2-1b | tg@d8192 | 258 | 259 | 267 | 0.96x | 0.97x |
| qwen3-8b | pp512 | 6560 | 6619 | 8016 | 0.82x | 0.83x |
| qwen3-8b | pp2048 | 6237 | 7101 | 8729 | 0.71x | 0.81x |
| qwen3-8b | pp8192 | 5356 | 6228 | 7650 | 0.70x | 0.81x |
| qwen3-8b | tg@d16 | 53 | 53 | 52 | 1.03x | 1.02x |
| qwen3-8b | tg@d8192 | 49 | 49 | 48 | 1.02x | 1.02x |
| qwen3-8b-fp8 | pp512 | 10448 | 10669 | 14277 | 0.73x | 0.75x |
| qwen3-8b-fp8 | pp2048 | 10713 | 10958 | 14967 | 0.72x | 0.73x |
| qwen3-8b-fp8 | pp8192 | 8435 | 9060 | 12333 | 0.68x | 0.73x |
| qwen3-8b-fp8 | tg@d16 | 90 | 89 | 87 | 1.03x | 1.03x |
| qwen3-8b-fp8 | tg@d8192 | 78 | 78 | 72 | 1.08x | 1.08x |

* Prefill at 2048 tokens: Qwen3-0.6B 0.91x → 1.01x of vLLM, Llama-3.2-1B 0.82x → 0.92x, Qwen3-8B bf16 0.71x → 0.81x
  (+14 %, grouped order and dual gate / up), Qwen3-8B-FP8 0.72x → 0.73x (its GEMMs already ran token tiles first).
* At 8192 tokens the attention change adds to it: Qwen3-0.6B 0.76x → 0.90x (+19 %), Llama-1B 0.72x → 0.85x, Qwen3-8B
  0.70x → 0.81x, FP8 0.68x → 0.73x.
* Decode unchanged (0.95-1.08x of vLLM).
* GEMM throughput of the 8B at 2048 rows (profile, `SHARPLIGHT_VK_PROFILE=1`): bf16 gate / up 126 TFLOPS, down 110, q / o
  115 (was 104-108 / 106 / 115); FP8 gate / up 205, down 193, q / o 190. Targets of 135 / 220 TFLOPS are not reached: the
  bf16 kernel is bound by its single-buffered staging (39.1), the FP8 kernel by its register count (222-250 registers for
  the 8-subgroup 128x128 tile: one workgroup per SM).

### 39.3 Serving: time to first token

2000-token prompts, 16 output tokens; TTFT p50 in ms (lower is better), output tok/s (higher is better).

| model | streams | TTFT p50 SL before | TTFT p50 SL after | TTFT p50 vLLM | tok/s SL before | tok/s SL after | tok/s vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 1 | 32 | 31 | 33 | 225 | 220 | 233 |
| qwen3-0.6b | 16 | 88 | 116 | 128 | 415 | 433 | 558 |
| llama3.2-1b | 1 | 52 | 49 | 44 | 143 | 158 | 166 |
| llama3.2-1b | 16 | 174 | 158 | 204 | 287 | 313 | 412 |
| qwen3-8b | 1 | 342 | 300 | 239 | 25 | 27 | 30 |
| qwen3-8b | 16 | 887 | 795 | 726 | 45 | 50 | 64 |
| qwen3-8b-fp8 | 1 | 201 | 197 | 139 | 43 | 43 | 51 |
| qwen3-8b-fp8 | 16 | 535 | 534 | 493 | 74 | 74 | 109 |

* Single stream: Llama-1B 52 → 49 ms (vLLM 44), Qwen3-8B 342 → 300 ms (vLLM 239); FP8 unchanged (201 / 197, vLLM 139).
* 16 streams: Llama-1B 174 → 158 ms (vLLM 204), Qwen3-8B 887 → 795 (vLLM 726); throughput +9-11 % on Llama-1B and Qwen3-8B.
  Qwen3-0.6B's TTFT at 16 streams moved 88 → 116 ms with higher throughput (433 vs 415 tok/s); one pair of rounds, not
  re-measured.

### 39.4 Greedy output

200-token greedy outputs of a short, a ~600-word and a ~1500-word prompt, before against after: byte-identical on
Qwen3-0.6B, Llama-3.2-1B and Qwen3-8B (all three prompts). Qwen3-8B-FP8 differs on the short prompt (a different story title
after ~70 words) and the ~600-word one ("including quantized formats" against "with quantized weights" after ~140 words), both
coherent: `SHARPLIGHT_ATTN_FE_REG=0` restores both, so the register softmax's sums (64-key tiles, four-lane row sums instead
of 32-key tiles and two-lane sums) flip these near-ties. The grouped order and the dual GEMM are bit-identical by
construction (`DualGemmTests`).

### 39.5 Tests

Release, niced, GPU 1, after merging `develop` at `649cd0fd`: Vulkan 1533 passed / 23 skipped, Cpu 1313 / 36 skipped, Serving
294 / 13 skipped, Families 1003 / 364 skipped (large models off), TextToSpeech (XTTS cloning without the opt-out) and
ImageGeneration (tiny Qwen-Image / Qwen-Image 2.1) green. The first Families run caught an out-of-bounds store of the dual
GEMM for output widths that are multiples of 16 but not of 64 (tiny GeGLU encoders: ModernBERT, T5 v1.1, N = 96); fixed and
covered by `DualGemmTests` (N = 96 with GELU). The benchmark models' widths are multiples of 64, so 39.2-39.4 are unaffected.

### 39.6 Next levers

* bf16 GEMM staging: 48 KB of shared memory per workgroup leaves 128x128 tiles single buffered; compute alone runs at ~196
  TFLOPS. Candidates: B fragments loaded from L1 without staging, smaller-padding layouts that fit two stages.
* FP8 GEMM: fewer live accumulators (the per-block scaling keeps a second set of fragments live).
* Down projection at 2048 rows: 512 tiles on 164 resident workgroups (3.1 waves); stream-K would remove the tail.
* Attention: with the K/V staging removed `fe` runs ~25 % faster (53.8 vs 70.1 ms on Qwen3-0.6B pp8192).
* Launch and barrier overhead is not where the small models lose: Llama-1B pp2048 runs 197 dispatches / 132 barriers per pass,
  41.6 ms of GPU time against 42.1 ms of wall time per pass, and the GEMMs are 83 % of it (gate / up 124 TFLOPS, down 109
  with 256 tiles on 164 resident workgroups). The down projection's f16 input copy (3 % of the pass) could come from the dual
  GEMM's store instead of a conversion launch.
* FP8 tile layouts tried: 4-subgroup 128x64 (two workgroups per SM) 163-177 TFLOPS, 8-subgroup 64x32 subtiles 145-156,
  against 173-195 for the default.

### 39.7 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=1 CUDA_VISIBLE_DEVICES=1 VLLM_VENV=<venv> HF_DIR=<checkpoints> OUT_DIR=bench/results/s39
bench/s38_all.sh vllm
bench/s38_all.sh sharplight before   # on the base build
bench/s38_all.sh sharplight after
python3 bench/s38_summarize.py bench/results/s39 before,after
```

## 40. LLM round 5 against vLLM: bf16 weights converted after the MMAs, dual GEMM storing the down projection's f16 input (2026-11-17)

Base: `develop` at `af8f2608` ("before"), branch `perf/llm-r5` merged with it ("after"). Same method, checkpoints and GPU as
section 39 (GPU 1, checkpoints on local disk); before and after ran back to back. The vLLM figures are section 39's (vLLM
0.33.0, same GPU, measured the same day), copied into [`bench/results/s40/`](../bench/results/s40) with the Sharplight results.

### 40.1 What changed

| change | effect (RTX PRO 4500; GEMM figures from `BenchTests.MatMulPrefill`, which includes the activations' f16 conversion) |
|---|---|
| **bf16 weights converted when staged** (gemm_cm2, f16 / bf16 weights): the 16 fetched bf16 values stay raw in the prefetch registers (as many as the converted halves) and are converted to f16 when they are written to shared memory, after the step's MMAs. Converted at the fetch, the conversion made each warp wait for its global loads before it started the MMAs, so loads and MMAs did not overlap. Same conversion, bit-identical | bf16 [4096 → 12288] at 2048 rows 123 → 136 TFLOPS, [12288 → 4096] 100 → 108, [4096 → 4096] 112 → 122; at 512 and 8192 rows +10-14 % alike |
| **Dual gate / up GEMM: one weight base address per thread** (DUAL with 128 threads and 64-value K blocks): every item of a thread reads the same matrix (gate or up) and rows 16 apart, so the per-item pointer selection and row address arithmetic (the kernel was at 254 registers) became one base address plus constant offsets | dual bf16 [4096 → 12288] x2 at 2048 rows 126 → 143 TFLOPS (with the change above, which alone gives 130) |
| **The dual GEMM stores the f16 copy its consumer reads** (`DUAL_Y16`): the down projection reads the f16 copy of SiLU(gate)·up; the dual GEMM now stores it next to the f32 output (one f16 accumulator store per fragment, round to nearest even like the conversion kernel) for the row counts it serves, and the conversion launch only runs for the others (the norm-written copies of section 28 with a per-row-count condition). Bit-identical (`DownProjectionOfDualMatchesSeparateBitwise`); `SHARPLIGHT_NORM_HALF_COPY=0` turns it off with the norm copies | Qwen3-8B bf16 at 2048 rows: GPU time per pass 264.1 → 260.5 ms (conversion 5.6 ms gone, dual GEMM +1.3 ms) |
| **FP8 GEMM: block products per output row fragment** (pair layout): each weight block's partial products are accumulated for one row fragment at a time (A fragments loaded once, B fragments per row fragment) instead of one column fragment at a time. Same products and scaling order per element, bit-identical | FP8 [4096 → 12288] at 2048 rows 194 → 197 TFLOPS, dual gate / up 201 → 203 |

### 40.2 Single stream

Tokens/s, higher is better (Sharplight: best of 2 process runs x 4 prefills; vLLM: median of 5, section 39). For the same
statistic on both sides see 40.9.

| model | metric | Sharplight before | Sharplight after | vLLM | SL/vLLM before | SL/vLLM after |
|---|---|---:|---:|---:|---:|---:|
| qwen3-0.6b | pp512 | 58896 | 60769 | 32688 | 1.80x | 1.86x |
| qwen3-0.6b | pp2048 | 83119 | 83480 | 82156 | 1.01x | 1.02x |
| qwen3-0.6b | pp8192 | 53287 | 53660 | 58706 | 0.91x | 0.91x |
| qwen3-0.6b | tg@d16 | 460 | 462 | 461 | 1.00x | 1.00x |
| qwen3-0.6b | tg@d8192 | 288 | 286 | 308 | 0.94x | 0.93x |
| llama3.2-1b | pp512 | 42121 | 45502 | 44578 | 0.94x | 1.02x |
| llama3.2-1b | pp2048 | 48710 | 53989 | 53097 | 0.92x | 1.02x |
| llama3.2-1b | pp8192 | 40215 | 43865 | 47421 | 0.85x | 0.93x |
| llama3.2-1b | tg@d16 | 296 | 295 | 286 | 1.03x | 1.03x |
| llama3.2-1b | tg@d8192 | 259 | 257 | 267 | 0.97x | 0.96x |
| qwen3-8b | pp512 | 6836 | 7653 | 8016 | 0.85x | 0.95x |
| qwen3-8b | pp2048 | 7114 | 7832 | 8729 | 0.81x | 0.90x |
| qwen3-8b | pp8192 | 6240 | 6784 | 7650 | 0.82x | 0.89x |
| qwen3-8b | tg@d16 | 53 | 53 | 52 | 1.03x | 1.03x |
| qwen3-8b | tg@d8192 | 49 | 49 | 48 | 1.01x | 1.02x |
| qwen3-8b-fp8 | pp512 | 10518 | 10738 | 14277 | 0.74x | 0.75x |
| qwen3-8b-fp8 | pp2048 | 10974 | 11053 | 14967 | 0.73x | 0.74x |
| qwen3-8b-fp8 | pp8192 | 9067 | 9147 | 12333 | 0.74x | 0.74x |
| qwen3-8b-fp8 | tg@d16 | 89 | 90 | 87 | 1.03x | 1.03x |
| qwen3-8b-fp8 | tg@d8192 | 78 | 78 | 72 | 1.07x | 1.07x |

* Prefill of the bf16 models: Qwen3-8B +10-12 % (pp2048 0.81x → 0.90x of vLLM, pp512 0.95x, pp8192 0.89x), Llama-3.2-1B
  +8-11 % (pp2048 0.92x → 1.02x, pp512 1.02x, pp8192 0.93x). Qwen3-0.6B gains 0.5-3 %: its GEMMs (K = 1024, 16 K steps per
  tile) and its attention, not the weight fetch, set its pass time. An interleaved repetition (base and new alternating per
  model and length, 2 runs each) gave the same figures within 1.5 %.
* Qwen3-8B-FP8 +1 % (the FP8 GEMM change); it stays at 0.74x of vLLM, the largest remaining gap (40.4).
* Decode unchanged (0.93-1.07x of vLLM).
* GEMM throughput of Qwen3-8B at 2048 rows (profile, `SHARPLIGHT_VK_PROFILE=1`): bf16 gate / up 140 TFLOPS (section 39: 126),
  down 117 (110), q / o 125 (115), k / v 124; FP8 gate / up 205, down 195, q / o 192 (205 / 193 / 190). The bf16 target of 135
  is reached by the gate / up GEMM only (the down projection's 512 tiles take 3.1 waves); FP8 stays below 220 (40.7).

### 40.3 Serving: time to first token

2000-token prompts, 16 output tokens; TTFT p50 in ms (lower is better), output tok/s (higher is better).

| model | streams | TTFT p50 SL before | TTFT p50 SL after | TTFT p50 vLLM | tok/s SL before | tok/s SL after | tok/s vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b | 1 | 29 | 29 | 33 | 220 | 241 | 233 |
| qwen3-0.6b | 16 | 83 | 84 | 128 | 445 | 443 | 558 |
| llama3.2-1b | 1 | 46 | 42 | 44 | 151 | 166 | 166 |
| llama3.2-1b | 16 | 239 | 133 | 204 | 312 | 339 | 412 |
| qwen3-8b | 1 | 300 | 272 | 239 | 27 | 29 | 30 |
| qwen3-8b | 16 | 792 | 725 | 726 | 50 | 54 | 64 |
| qwen3-8b-fp8 | 1 | 196 | 195 | 139 | 43 | 44 | 51 |
| qwen3-8b-fp8 | 16 | 428 | 528 | 493 | 73 | 75 | 109 |

* Single stream: Qwen3-8B 300 → 272 ms (vLLM 239), Llama-1B 46 → 42 ms (vLLM 44).
* 16 streams: Qwen3-8B 792 → 725 ms, level with vLLM's 726; throughput +8 % (Qwen3-8B, Llama-1B). The p50 TTFT at 16 streams
  depends on where the 16 prompts fall in the steps and moves by up to ±30 % between rounds: a repetition of Llama-1B and
  Qwen3-8B-FP8 (both builds again) gave Llama-1B 184 → 154 ms and FP8 530 → 524 ms with the same throughputs (341 / 75 tok/s
  after), so FP8's 428 before is a lucky round, not a regression.

### 40.4 Tried and not kept

| tried | result |
|---|---|
| bf16: 32-value K blocks, double-buffered shared tiles (the existing GBK 32 path) | 88 / 72 TFLOPS ([4096 → 12288] / [12288 → 4096] at 2048 rows) against 123 / 100 single-buffered 64-value blocks (94 / 77 with the deferred conversion) |
| bf16 down projection: two K splits (forced through `SHARPLIGHT_GEMM_WAVE`) for the 3.1 waves of 512 tiles | 1.94 ms against 1.91 (the reduction costs what the fuller last wave saves); [8192 → 2048] (1.6 waves) 0.70 against 0.66 ms |
| bf16 down projection: grouped tile order with 4 / 8 / 32 token tiles per group instead of 16, or none | 1-4 % slower at 2048 and 8192 rows |
| FP8: 64-value K steps in two shared buffers (one barrier per step), 128-value weight blocks scaled per half | 146 TFLOPS against 195: twice the steps and block scalings |
| FP8: a second register set of prefetched operands (loads issued two steps ahead) | 128-146 TFLOPS: register spills (the kernel already uses 222-250 registers) |
| FP8: 16 subgroups per workgroup (4x4 x 32x32, 1 workgroup with 16 warps per SM), 2x8 x 64x16, 4x1 x 32x64, 4-subgroup 128x64 tiles with the row-fragment loop | 165 / 182 / 154 / 172 TFLOPS against 197 |
| FP8: compile only the pair-layout tile loop (no generic fallback in the same shader) | no change (the driver gives the kernel the same registers) |
| FP8: one weight base address per thread (as for the dual bf16 GEMM) | +0.5-1 %, within the spread of the end-to-end runs; not kept |
| dual GEMM's f16 copy staged through shared memory with 16-byte stores per lane | the dual GEMM 6.4 ms slower per pass (more than the conversion it saves); the direct f16 accumulator store costs 1.3 ms |

Diagnostics (with `SHARPLIGHT_GEMM_MACROS`): bf16 [4096 → 12288] at 2048 rows runs 1.04 ms with the global loads removed after the
first step (`GEMM_EXP=1`, 199 TFLOPS) and 1.06 ms with the MMAs removed (`GEMM_EXP=2`), against 1.52 ms with both; for
[12288 → 4096] 1.26 / 1.49 / 1.91 ms. The fetch alone of the down projection moves 3.2 GB from L2 in 1.49 ms (2.1 TB/s), so
it is now about as expensive as the MMAs, and the two overlap only partly at two workgroups per SM. The FP8 GEMM's figures (an experiment
build, 1 workgroup per SM): 0.73 ms without loads (284 TFLOPS), 0.67 ms without MMAs, 1.06 ms with both.

### 40.5 Greedy output

200-token greedy outputs of a short, a ~1100-token and a ~4500-token prompt, before against after on the same `develop` base:
byte-identical on Qwen3-0.6B, Llama-3.2-1B, Qwen3-8B and Qwen3-8B-FP8 (all 12). The long prompt runs with `--ctx 8192`: the
round-4 script gave it `--ctx 4096`, which it exceeds, so section 39.4's long-prompt comparison compared two empty outputs
(the short and mid ones are valid). Every change keeps each output's arithmetic:
the bf16 conversion is the same operation at another point of the loop, the dual GEMM's f16 copy rounds like the conversion
kernel (`DownProjectionOfDualMatchesSeparateBitwise`), the FP8 row-fragment loop computes the same products in the same order.

### 40.6 Tests

Release, niced, GPU 1, after merging `develop` at `af8f2608` (the Metal provider merge): Vulkan 1540 passed / 23 skipped,
Cpu 1313 / 36 skipped, Serving 294 / 13 skipped, Families 1003 / 364 skipped (large models off), Metal code generation
(`Sharplight.Metal.Tests`: the shared kernels of a transformer block compiled to MSL, with and without 8x8 simdgroup matrices)
4 / 1 skipped. The Metal merge left `DualGemmTests` on the removed `Gpu.Vulkan`, so `develop`'s Vulkan tests did not compile;
fixed here (`Gpu.Device`). New: `DualGemmTests.DownProjectionOfDualMatchesSeparateBitwise` (bf16 / f16, partial row and column
tiles).

### 40.7 Next levers

* FP8 GEMM (Qwen3-8B-FP8 pp2048 0.74x of vLLM, GEMMs 195-205 TFLOPS against ~235 for vLLM): one workgroup per SM (222-250
  registers for the 8-subgroup 128x128 tile; the 48 KB of shared memory per workgroup forbid a second stage), so the
  fetch and the MMAs overlap only partly (0.67 + 0.73 ms alone, 1.06 ms together for [4096 → 12288]). A 2-workgroup layout
  needs ≤ 128 registers per thread, which the driver does not give the 128x64 4-subgroup tile either (229-255).
* bf16 GEMM fetch: the down projection's fetch alone takes about as long as its MMAs (1.49 against 1.26 ms, 2.1 TB/s from L2).
* Stream-K without a reduction launch (tiles split across workgroups, partial sums in a fixed order) for grids of 1-3 waves:
  the two-way K split's reduction costs what the fuller last wave saves.
* Norms writing the f16 / E4M3 activations at prefill sizes instead of f32 plus a conversion (bf16: 3.3 ms of 260 on Qwen3-8B
  in the [4096 → 4096] / gate-up conversions; FP8: 7.4 ms of 189 in the activation quantization).
* Attention K/V staging for long context (Qwen3-0.6B pp8192 0.91x; section 39.6).

### 40.8 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=1 CUDA_VISIBLE_DEVICES=1 VLLM_VENV=<venv> HF_DIR=<checkpoints> OUT_DIR=bench/results/s40
bench/s38_all.sh sharplight before   # on the base build
bench/s38_all.sh sharplight after
python3 bench/s38_summarize.py bench/results/s40 before,after   # vLLM files copied from bench/results/s39
# GEMM micro-benchmarks (bf16 / FP8, Qwen3-8B shapes at 2048 rows; SHARPLIGHT_BENCH_DUAL=1 for gate / up pairs):
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_ROWS=2048 SHARPLIGHT_BENCH_TYPES=bf16 SHARPLIGHT_BENCH_SHAPES="4096,12288;12288,4096;4096,4096" \
  dotnet run --project tests/Sharplight.Vulkan.Tests -c Release -- -method "*MatMulPrefill"
```

### 40.9 Prefill with an identical procedure (Sharplight, vLLM, llama.cpp)

For charts that compare engines, this table uses one procedure and one statistic on every side, measured back to back in one
session on GPU 1 with no other process on it (2026-11-17, after all of the above; `bench/s40_identical.sh`, table from
`bench/s40_identical_summarize.py`, raw output in [`bench/results/s40/identical/`](../bench/results/s40/identical)):

* the same weights on both sides: the Hugging Face bf16 / FP8 checkpoints for Sharplight and vLLM 0.33.0, the same GGUF files
  for Sharplight and llama.cpp (b11177, CUDA build, `llama-bench -ngl 99 -fa 1`, its default batch sizes);
* the same prompts: 512, 2048 and 8192 random token ids, no prefix caching;
* the same repetitions and statistic: one process run per engine and model (Sharplight: one per prompt length), per length one
  warm-up prefill and 5 timed prefills, reported as the **median of the 5** (Sharplight `--bench`'s "prefill median of 5";
  vLLM `bench/vllm_single.py --points pp --reps 5`, a generate call with one output token; llama-bench `-r 5`, the median of
  its `samples_ns`).

Tokens/s of prompt processing, higher is better.

| model | weights | prompt | Sharplight | other engine | other | Sharplight / other |
|---|---|---:|---:|---|---:|---:|
| qwen3-0.6b | HF bf16 | 512 | 61062 | vLLM | 30203 | 2.02x |
| qwen3-0.6b | HF bf16 | 2048 | 84458 | vLLM | 80743 | 1.05x |
| qwen3-0.6b | HF bf16 | 8192 | 51887 | vLLM | 59802 | 0.87x |
| llama3.2-1b | HF bf16 | 512 | 45544 | vLLM | 44107 | 1.03x |
| llama3.2-1b | HF bf16 | 2048 | 53369 | vLLM | 54322 | 0.98x |
| llama3.2-1b | HF bf16 | 8192 | 42176 | vLLM | 48653 | 0.87x |
| qwen3-8b | HF bf16 | 512 | 7285 | vLLM | 8114 | 0.90x |
| qwen3-8b | HF bf16 | 2048 | 7583 | vLLM | 8755 | 0.87x |
| qwen3-8b | HF bf16 | 8192 | 6716 | vLLM | 7653 | 0.88x |
| qwen3-8b-fp8 | HF FP8 | 512 | 10309 | vLLM | 14770 | 0.70x |
| qwen3-8b-fp8 | HF FP8 | 2048 | 10722 | vLLM | 14956 | 0.72x |
| qwen3-8b-fp8 | HF FP8 | 8192 | 9057 | vLLM | 12317 | 0.74x |
| qwen3-0.6b | GGUF Q8_0 | 512 | 60628 | llama.cpp | 50393 | 1.20x |
| qwen3-0.6b | GGUF Q8_0 | 2048 | 78302 | llama.cpp | 47411 | 1.65x |
| qwen3-0.6b | GGUF Q8_0 | 8192 | 49540 | llama.cpp | 36731 | 1.35x |
| llama3.2-1b | GGUF Q8_0 | 512 | 35905 | llama.cpp | 42839 | 0.84x |
| llama3.2-1b | GGUF Q8_0 | 2048 | 43077 | llama.cpp | 42196 | 1.02x |
| llama3.2-1b | GGUF Q8_0 | 8192 | 35972 | llama.cpp | 36735 | 0.98x |
| qwen3-8b | GGUF Q4_K_M | 512 | 6645 | llama.cpp | 7007 | 0.95x |
| qwen3-8b | GGUF Q4_K_M | 2048 | 6378 | llama.cpp | 6874 | 0.93x |
| qwen3-8b | GGUF Q4_K_M | 8192 | 5768 | llama.cpp | 6233 | 0.93x |

Against the best-of figures of 40.2 the medians are 0-3 % lower on Sharplight (Qwen3-8B pp2048 7583 against 7832) and within
the run-to-run spread on vLLM (section 39's medians: 8729 against 8755 here). Decode (median of 64 steps against vLLM's median of
5 128-token runs) and TTFT (median of the loadgen rounds on both sides) in 40.2 / 40.3 already use the same statistic.

## 41. Apple silicon: native Metal provider against MoltenVK and llama.cpp Metal (macOS CI, 2026-11-17)

GitHub's `macos-15` runner (Apple M1 virtual machine, paravirtual Apple GPU: Mac2 family, argument buffers tier 2, 3 CPU cores,
7 GB), branch `macos/metal-r1`, jobs `metal` and `metal-bench` of `.github/workflows/macos.yml`, run 44018997843. Numbers are
indicative only (shared virtualized hardware, one run each). Sharplight: `sharplight MODEL --provider metal|vulkan --bench N`
(best warm prefill of N tokens; decode: best of 64 steps). llama.cpp b11177 built in CI with `GGML_METAL=ON`:
`llama-bench -ngl 99 -fa 1 -p 512 -n 128 -r 3` (mean).

| Model | Metal pp512 | MoltenVK pp512 | llama.cpp Metal pp512 | Metal decode | MoltenVK decode | llama.cpp Metal tg128 |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q4_K_M | **811.9** | 430.0 | 218.7 | **62.9** | 46.3 | 7.5 |
| Qwen3-0.6B Q8_0 | **855.4** | 427.6 | 329.2 | 62.1 | **65.7** | 6.6 |
| Qwen3-1.7B Q4_K_M | **395.9** | 161.7 | 126.6 | **29.8** | 24.4 | 6.1 |
| Qwen3-1.7B Q8_0 | **415.6** | 159.8 | 79.8 | 25.0 | **29.9** | 3.1 |
| Llama-3.2-1B Q4_K_M | **559.7** | 225.9 | 144.9 | **40.1** | 33.0 | 9.6 |
| Llama-3.2-1B Q8_0 | **589.1** | 221.6 | 243.0 | 36.7 | **39.5** | 10.8 |
| Qwen2.5-0.5B Q4_K_M (pp128 / pp512) | **1561.4 / 1281.4** | 622.1 / 568.2 | — | **72.2** | 58.4 | — |

Metal prefill is 2-2.6x MoltenVK: the prefill matmuls run on `simdgroup_matrix` (`gemm_sg`; MoltenVK exposes no cooperative
matrices, so the Vulkan kernels take the SIMT GEMM), and launches are encoded without MoltenVK's translation layer. Decode is
memory bound: Metal is 20-30 % faster for Q4_K_M, but 6-17 % slower than MoltenVK for Q8_0, where MoltenVK takes the
integer-dot matvec (`int_dot`, emulated by MoltenVK) that the Metal provider does not enable yet. llama.cpp's numbers on this
paravirtual GPU are far below its usual Apple silicon results (decode 3-11 tok/s); its Metal backend evidently runs a reduced
path on this device (the VM's GPU does not report the Apple7+ family), so the comparison needs real Apple hardware before
any conclusion.

### 41.1 Round 2 (split-K `gemm_sg`, 8x8 flash attention, `int_dot` on Metal; run 44031312716)

Same runner and commands; "Metal, int_dot off" runs `--disable-features int_dot`.

| Model | Metal pp512 | MoltenVK pp512 | llama.cpp Metal pp512 | Metal decode | Metal decode, int_dot off | MoltenVK decode | llama.cpp tg128 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B Q4_K_M | **1331.5** | 431.6 | 341.4 | **65.8** | 62.6 | 46.2 | 10.3 |
| Qwen3-0.6B Q8_0 | **1455.1** | 428.5 | 335.6 | 64.8 | **66.0** | 62.7 | 6.8 |
| Qwen3-1.7B Q4_K_M | **483.1** | 163.2 | 108.6 | **36.5** | 30.6 | 24.3 | 5.7 |
| Qwen3-1.7B Q8_0 | **522.5** | 155.3 | 165.2 | **28.8** | 28.6 | 28.2 | 5.3 |
| Llama-3.2-1B Q4_K_M | **698.6** | 226.0 | 133.1 | **48.2** | 38.5 | 32.5 | 8.0 |
| Llama-3.2-1B Q8_0 | 691.0 | 210.0 | 196.4 | **37.4** | 34.8 | 36.0 | 8.3 |
| Qwen2.5-0.5B Q4_K_M (pp128 / pp512) | **1581.9 / 1709.9** | 618.2 / 553.7 | — | 64.1 | — | 58.2 | — |

Split-K raises prefill by 20-64 % over round 1 (the 64 x 128 tile left most of the GPU idle at 512 rows of these narrow models);
Metal prefill is now 3-3.4x MoltenVK. `int_dot` stays on: +5 to +25 % decode for Q4_K_M, Q8_0 within the runner's noise. The
llama.cpp numbers keep the caveat above (reduced path on this paravirtual GPU).

## 42. LLM round 6: per-model gap to vLLM and llama.cpp, SD3.5 attention regression fixed, dual GEMM for quantized weights (2026-11-17)

Base: `develop` at `b4a5e60d` (v0.2.0 merged back), branch `perf/llm-r6`. GPU 1 (RTX PRO 4500 Blackwell, 32 GB, 82 SMs), vLLM
0.33.0, llama.cpp b11177 (CUDA build). Raw output in [`bench/results/s42/`](../bench/results/s42).

### 42.1 Gap table (identical procedure)

`bench/s42_gap.sh` per engine and model, table from `bench/s42_summarize.py`: one process run per engine and model (Sharplight:
one per prompt length), per length one warm-up and 5 timed prefills of 512 / 2048 / 8192 random token ids, the **median of the
5** on every side, no prefix caching. Decode: Sharplight's median of 64 steps after the prompt (512 / 8192 tokens); vLLM's median
of 5 128-token runs at depth 16 / 8192 (`bench/vllm_single.py`); llama-bench's tg128 (median of 5) after an empty context. Each
row compares the same weights (the Hugging Face checkpoint for Sharplight and vLLM, the same GGUF file for Sharplight and
llama.cpp, `-ngl 99 -fa 1`). Engines ran back to back per model, the GPU otherwise idle. Tokens/s, higher is better; measured
before the changes of 42.4 (the Llama-3.1-8B Q4_K_M row after them is in 42.4).

| model | weights | other engine | metric | Sharplight | other | Sharplight / other |
|---|---|---|---|---:|---:|---:|
| qwen3-8b | HF bf16 | vLLM | pp512 | 7276 | 8203 | 0.89x |
| qwen3-8b | HF bf16 | vLLM | pp2048 | 7611 | 8782 | 0.87x |
| qwen3-8b | HF bf16 | vLLM | pp8192 | 6733 | 7661 | 0.88x |
| qwen3-8b | HF bf16 | vLLM | tg@d16 (SL: after 512) | 52 | 52 | 1.01x |
| qwen3-8b | HF bf16 | vLLM | tg@d8192 | 49 | 48 | 1.02x |
| qwen3-8b-fp8 | HF FP8 | vLLM | pp512 | 10260 | 14446 | 0.71x |
| qwen3-8b-fp8 | HF FP8 | vLLM | pp2048 | 10673 | 15024 | 0.71x |
| qwen3-8b-fp8 | HF FP8 | vLLM | pp8192 | 9086 | 12346 | 0.74x |
| qwen3-8b-fp8 | HF FP8 | vLLM | tg@d16 (SL: after 512) | 87 | 87 | 1.01x |
| qwen3-8b-fp8 | HF FP8 | vLLM | tg@d8192 | 77 | 77 | 1.00x |
| llama3.1-8b | HF bf16 | vLLM | pp512 | 7372 | 8532 | 0.86x |
| llama3.1-8b | HF bf16 | vLLM | pp2048 | 7722 | 8836 | 0.87x |
| llama3.1-8b | HF bf16 | vLLM | pp8192 | 6938 | 7806 | 0.89x |
| llama3.1-8b | HF bf16 | vLLM | tg@d16 (SL: after 512) | 53 | 53 | 1.01x |
| llama3.1-8b | HF bf16 | vLLM | tg@d8192 | 50 | 49 | 1.02x |
| llama3.1-8b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp512 | 6496 | 7121 | 0.91x |
| llama3.1-8b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp2048 | 6402 | 6993 | 0.92x |
| llama3.1-8b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp8192 | 5841 | 6388 | 0.91x |
| llama3.1-8b-q4_k_m | GGUF Q4_K_M | llama.cpp | tg128 (SL: after 512) | 138 | 149 | 0.93x |
| qwen3.8-27b-q8_0 | GGUF Q8_0 | llama.cpp | pp512 | 1827 | 1885 | 0.97x |
| qwen3.8-27b-q8_0 | GGUF Q8_0 | llama.cpp | pp2048 | 1875 | 1869 | 1.00x |
| qwen3.8-27b-q8_0 | GGUF Q8_0 | llama.cpp | pp8192 | 1699 | 1815 | 0.94x |
| qwen3.8-27b-q8_0 | GGUF Q8_0 | llama.cpp | tg128 (SL: after 512) | 29 | 28 | 1.05x |
| qwen3-30b-a3b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp512 | 6984 | 5976 | 1.17x |
| qwen3-30b-a3b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp2048 | 8691 | 5896 | 1.47x |
| qwen3-30b-a3b-q4_k_m | GGUF Q4_K_M | llama.cpp | pp8192 | 7340 | 5315 | 1.38x |
| qwen3-30b-a3b-q4_k_m | GGUF Q4_K_M | llama.cpp | tg128 (SL: after 512) | 215 | 257 | 0.84x |
| qwen3.6-35b-a3b-q4_k_m | GGUF UD-Q4_K_M | llama.cpp | pp512 | 6859 | 4987 | 1.38x |
| qwen3.6-35b-a3b-q4_k_m | GGUF UD-Q4_K_M | llama.cpp | pp2048 | 7962 | 4892 | 1.63x |
| qwen3.6-35b-a3b-q4_k_m | GGUF UD-Q4_K_M | llama.cpp | pp8192 | 6835 | 4749 | 1.44x |
| qwen3.6-35b-a3b-q4_k_m | GGUF UD-Q4_K_M | llama.cpp | tg128 (SL: after 512) | 191 | 189 | 1.01x |
| gpt-oss-20b-mxfp4 | GGUF MXFP4 | llama.cpp | pp512 | 9819 | 11399 | 0.86x |
| gpt-oss-20b-mxfp4 | GGUF MXFP4 | llama.cpp | pp2048 | 10594 | 11416 | 0.93x |
| gpt-oss-20b-mxfp4 | GGUF MXFP4 | llama.cpp | pp8192 | 8888 | 10926 | 0.81x |
| gpt-oss-20b-mxfp4 | GGUF MXFP4 | llama.cpp | tg128 (SL: after 512) | 229 | 227 | 1.01x |

Where the gap is (per-op profiles, `SHARPLIGHT_VK_PROFILE=1`, 2048-token prefill; raw in `bench/results/s42/profiles/`):

* **Dense bf16 (Qwen3-8B, Llama-3.1-8B: 0.86-0.89x of vLLM)**: GEMMs are 85 % of the GPU time (Qwen3-8B: dual gate / up 140
  TFLOPS, down 118, q / o 125, k / v 112), attention 4.6 %, norms 3.6 %. vLLM's GEMMs run at ~160 TFLOPS on these shapes; the
  gap is the GEMM fetch (42.3 / section 40.4).
* **FP8 (Qwen3-8B-FP8: 0.71-0.74x)**: GEMMs 81 % (gate / up 199 TFLOPS, down 189, q / o 184, k / v 159), activation
  quantization 4 %, attention 6.4 %, norms 5.6 %. Even at vLLM's ~235 TFLOPS the GEMMs alone would leave 0.83x: the rest
  (quantization launches, norms, attention) is a third of the gap. See 42.3.
* **GGUF dense (Llama-3.1-8B Q4_K_M 0.91x, Qwen3.8-27B Q8_0 0.94-1.00x of llama.cpp)**: Llama's gate / up ran as two GEMMs and an
  elementwise launch (4.3 %); fixed in 42.4. The 27B (32 GB GPU, 29 GB of Q8_0 weights) runs 512-row passes; its time is the
  int8 GEMM (MMQ) and the delta-rule chunks of its linear-attention layers.
* **MoE (now round 6b's)**: Qwen3-30B-A3B and Qwen3.6-35B-A3B prefill 1.17-1.63x of llama.cpp, but Qwen3-30B-A3B decode 0.84x;
  gpt-oss-20b prefill 0.81-0.93x (its MXFP4 expert GEMM is 52 % of the pass at 107 TFLOPS), decode level.
* **Did not run**: Gemma-3-12B from Hugging Face: Sharplight's HF importer does not take `Gemma3ForConditionalGeneration`
  (the multimodal wrapper the 12B ships as; Gemma-3 GGUFs load), and vLLM stalled on the network file system while loading it;
  Qwen3.8-27B-FP8 (29 GB of weights): Sharplight needs 28.4 GiB plus 2.4 GiB for the output layer, which do not fit one 32 GB
  GPU without `--cpu-offload`; vLLM did not start within the session either (it needs the same weights plus its KV cache).
  Qwen3-30B-A3B-FP8 (31 GB) fits neither.

### 42.2 SD3.5 Medium at 1024 px: root cause of the split-f16 attention regression

Section 39's power-of-two scaling of the split-f16 flash kernels (`fa` / `fe` with FA_SPLIT, every query row, key row and V
tile scaled into [2^14, 2^15) before the hi / lo split) dropped SD3.5 Medium's 1024 px text-to-image parity against diffusers
from 38.4 to 27.0 dB (GGUF 30.2 to 26.5 dB); it was reverted for v0.2.0. Bisected here:

* The only split-f16 attention in the SD3.5 pipeline is the CLIP-L / CLIP-G text encoders' (causal `fa`, 77 tokens). The
  transformer runs single f16 products (its RMS-normed queries and keys, `F16ProductsAttribute`), the VAE's single-head
  attention (head size 512) another kernel. `VaeDecodesDiffusersLatents` (new) decodes diffusers' final latents at 512 and 1024:
  78.1 / 76.5 dB with either kernel.
* The scaling is exact in f16's normal range, but it moves the rounding of the lo parts of small values (subnormal f16 when
  unscaled): the CLIP hidden states changed by up to 9e-3 absolute (relative RMS 4e-5 to 7e-4 between the two kernels, about as
  large as either one's distance to a CPU run of the same graph, so neither is more accurate). The 28-step 1024 px trajectory
  follows the text conditioning that closely: 38.4 → 27.0 dB. Scaling only Q, only K or only V each moved it as well; with all
  three scales at 1 the outputs were bit-identical to the old kernel.
* Fix (`split_scale_outside`, common.slang): scale 1 while the row's or tile's largest magnitude is in [2^-6, 2^15), the power
  of two outside, where the hi parts would overflow (XTTS's 1e10 latents, `SplitHalfLargeMagnitudes`) or the largest values
  lose bits to f16's subnormals. In range the kernels round exactly as before 622e6bd9 (the CLIP outputs are bit-identical).

| test | before the fix | after |
|---|---|---|
| Sd35MediumTests.TextToImageMatchesDiffusers diffusers 512 / 1024 | 51.6 / 27.0 dB | 48.2 / 38.4 dB |
| (gguf) 512 / 1024 | 41.8 / 26.5 dB | 43.1 / 30.2 dB |
| FullAttentionTests (incl. SplitHalfLargeMagnitudes) | 30 / 30 | 30 / 30 |
| XttsCloningTests (no f32_products opt-out) | 5 / 5 | 5 / 5 |

The diffusers 1024 threshold is now 35 dB (was 30), so a rounding change in the text encoders shows up as a failure.

### 42.3 FP8 GEMM: what limits it (lever 1, not solved)

Experiments on [4096 → 12288] at 2048 rows (`BenchTests.MatMulPrefill`, the quantize launch included; 197 TFLOPS as shipped):

| experiment (`F8_EXP` macro, experiment build) | ms | TFLOPS |
|---|---:|---:|
| as shipped (8 subgroups, 128x128x128, 222 registers, 1 workgroup per SM) | 1.044 | 197 |
| no global loads, no staging, no barriers (MMAs, shared-memory fragment loads and block scaling only) | 0.722 | 286 |
| ... and the fragments loaded once (MMAs + scaling) | 0.700 | 294 |
| ... without the block scaling (MMAs + shared loads) | 0.636 | 324 |
| MMAs only | 0.620 | 333 |
| loads, staging and barriers only (no MMAs) | 0.675 | 305 |
| 128x64 tiles, 3 workgroups per SM: loads / staging only | 0.948 | 218 |

* The FP8 MMAs run at ~335 TFLOPS on this card (not the ~520 an earlier note assumed); the block scaling costs 8 %, the
  shared-memory fragment loads 3 %. The fetch alone (0.675 ms) moves 1.6 GB from L2 at 2.4 TB/s; the 128x64 tile, with three
  workgroups per SM, moves 1.5x the bytes at 2.5 TB/s: L2 bandwidth, not latency, bounds the fetch. Fetch and MMAs take
  about the same time and overlap to 1.04 ms (perfect overlap: ~0.72).
* Two workgroups per SM are out of reach for a 128x128 tile: the per-block scaling needs a second accumulator set, and
  the driver gives a 4-subgroup 128x128 tile (64x64 per subgroup) 255 registers with spills (94-122 TFLOPS), with or
  without register prefetch; preventing the B-fragment hoisting did not lower the count (238 vs 222).
* Activations read straight into the MMA fragments (new `F8_AFRAG` layout written by the quantization: each lane's
  operands of a 128-value block as two 16-byte loads; only the weights staged, double buffered, one barrier per step; layout
  probed and NaN on mismatch): 178 TFLOPS, because building the A fragments from bytes costs ~15 %; with the fragment
  construction taken out (an invalid experiment) it reaches 207. Not kept; the 207 shows the ceiling of single-workgroup
  double buffering at this tile size is ~+5 %.
* Other layouts (4,2,2,4 / 1,8,8,1): 190 TFLOPS. 16-byte shared stores: no change (the compiler already merges them).

So a FP8 GEMM at vLLM's ~235 TFLOPS needs the fetch and the MMAs to overlap almost fully at 1.6 GB of L2 traffic, which the
48 KB of shared memory per workgroup (one 36 KB stage) and the register budget do not allow with register-staged loads. The
remaining options are bigger tiles with less L2 traffic per FLOP (256-wide tiles need more than 48 KB per stage or K steps of
64 with two accumulator sets) or asynchronous copies the Vulkan driver does not expose today.

### 42.4 Dual gate / up GEMM for quantized weights (kept)

The dual gemm_cm2 (gate and up in one launch storing `silu(gate) * up`, section 39) already decoded generic formats 16 values
at a time; the lowering offered it only to f16 / bf16. It now serves every format gemm_cm2 decodes that way where the separate
matmuls also run gemm_cm2 (not the int8 GEMM, not MXFP4R's 32-value raw units). Bit-identical to the separate launches
(`DualGemmTests` with q4_k, q6_k, q5_k, q4_0 and the down projection after a q4_k dual).

| measurement | before | after |
|---|---:|---:|
| Q4_K [4096 → 14336] x2 + SiLU·mul at 2048 rows | 4.28 ms (112 TFLOPS) | 3.77 ms (128) |
| Q6_K [4096 → 12288] x2 + SiLU·mul at 2048 rows | 4.42 ms | 3.87 ms |
| Llama-3.1-8B Q4_K_M prefill 2048 (median of 5, two runs each) | 6356 / 6347 tok/s | 6814 / 6813 tok/s |

Llama-3.1-8B Q4_K_M pp2048 goes from 0.92x to 0.97x of llama.cpp (6993 tok/s in 42.1).

### 42.5 Stream-K without a reduction launch (lever 2, tried, not kept)

Both the FP8 GEMM and gemm_cm2 got a stream-K form in one launch: G workgroups (one per SM for the FP8 tile, two for the bf16
tile) run all but the last two waves of tiles whole in launch order, then share the remaining G to 2G tiles' K steps evenly.
A tile cut in two is started by the workgroup whose range ends in it (its first work, accumulators to a scratch slot, a flag
raised with a device-scope release) and finished by the next workgroup as its last work (it polls the flag with its first
subgroup, loads the accumulators and continues in the same K order), so every output sums its K steps exactly as the plain
launch: bit-identical, no reduction launch, no deadlock (a workgroup only waits for its predecessor's first work).

| GEMM at 2048 rows | plain | stream-K |
|---|---:|---:|
| FP8 [12288 → 4096] (512 tiles, 6.2 waves of 82) | 1.178 ms | 1.163 ms |
| FP8 [4096 → 4096] | 0.384 ms | 0.372 ms |
| FP8 [4096 → 1024] (128 tiles) | 0.122 ms | 0.120 ms |
| bf16 [12288 → 4096] (512 tiles, 3.1 waves of 164) | 1.921 ms | 1.948 ms |
| bf16 [4096 → 4096] | 0.573 ms | 0.583 ms |
| Qwen3-8B-FP8 prefill 2048, end to end (two runs each) | 190.5 / 190.5 ms | 190.9 / 190.8 ms |

Greedy outputs with the FP8 stream-K were byte-identical (three prompts, `--ctx 8192`). The GEMM gains are 1-3 % for FP8 and
negative for bf16, and nothing end to end: with two workgroups per SM a partial last wave is not idle time (the lone workgroup
on an SM runs at nearly the SM's full rate), and with one per SM the fetch, not the wave quantization, sets the time. The first
version that walked the K ranges in tile order lost L2 locality on the down projection (1.215 ms); the data-parallel waves
first fixed that. A pointer placed after the 32-bit row count in the parameter block hung the GPU (the host packs the block
without padding): pointers go before the count. Code kept out of the branch (diff in the round's notes).

### 42.6 Greedy output

Every change kept is bit-identical by construction and test: the attention fix restores the pre-622e6bd9 rounding for in-range
data (the CLIP outputs of the SD3.5 pipeline are bit-identical to the old kernel's); the dual quantized GEMM equals the separate
launches bit for bit (`DualGemmTests`, gate / up and the down projection reading its output). The FP8 stream-K experiment
(not kept) gave byte-identical 200-token greedy outputs on Qwen3-8B-FP8 for a short, a ~1100-token and a ~4500-token prompt
(`--ctx 8192`). A greedy rerun of Llama-3.1-8B Q4_K_M with and without the dual GEMM did not finish: the network file system
holding the GGUF stalled for the rest of the session.

### 42.7 Tests

TESTS

### 42.8 Next levers

* FP8 GEMM (the largest dense gap, 0.71-0.74x): see 42.3; the activation quantization (4 % of the pass) and the norms (5.6 %)
  can be fused (norms writing E4M3 rows directly), worth ~0.03x.
* bf16 GEMM fetch (0.86-0.89x): the down projection's fetch alone takes as long as its MMAs (section 40.4).
* Attention at 8192 tokens is 6-9 % of the dense passes; K/V double buffering in `fe` (lever 3) is still open.
* Gemma-3 from Hugging Face (`Gemma3ForConditionalGeneration`) in the HF importer, so the dense table can include it.
* MoE, serving and Qwen3.8-Flash-Next: round 6b.

## 43. Serving and MoE against vLLM and llama.cpp: expert products of 16-64 sequences on the grouped GEMM (round 6b, 2026-11-17)

Base: `develop` at `4c6c6e7a` ("before"), branch `perf/llm-r6b` ("after"). One RTX PRO 4500 (GPU 2, nothing else running on it;
checkpoints on local disk). Section 42 is the parallel dense-model round.

### 43.1 Procedure (identical on every engine)

`bench/s43_serve.sh` (on top of `bench/s16_serve.sh`): per model one server per engine, 64 slots of 512 tokens (Sharplight
`serve --slots 64 --ctx 512`, llama-server `-np 64 -c 32768 -ngl 99 -fa on`, vLLM `--max-num-seqs 64 --max-model-len 512`,
prefix caching off), then `bench/loadgen.py` at 1 / 4 / 16 / 64 concurrent streams: closed loop, ~250-token random-word prompts,
128 output tokens with `ignore_eos`, temperature 0, one concurrent warm-up request per worker, 8 / 6 / 4 / 3 requests per
worker, **3 rounds per concurrency, median** (min..max) of the aggregate output tok/s; TTFT p50 and the per-stream decode rate
(1000 / mean inter-token latency) are the medians over the rounds. Engines of one model ran back to back. Same weights per pair:
the GGUF files for Sharplight and llama.cpp (b11177 CUDA), the Hugging Face checkpoint (`openai/gpt-oss-20b`, native MXFP4
experts, bf16 rest) for Sharplight and vLLM 0.33.0 (Marlin MXFP4 MoE backend, FlashInfer attention, CUDA graphs). Tables:
`bench/s43_serve_summarize.py bench/results/s43/serve-before bench/results/s43/serve-after`; raw JSON (with the servers'
`/metrics` deltas) in [`bench/results/s43/`](../bench/results/s43).

Models: Qwen3-30B-A3B Q4_K_M (128 experts, top 8), Qwen3.6-35B-A3B UD-Q4_K_M (256 experts, top 8), gpt-oss-20b (32
experts, top 4) as MXFP4 GGUF and as the HF checkpoint. Qwen3-30B-A3B-FP8 (31 GB) does not leave vLLM room for a KV cache on
the 32 GB card, so Qwen3-30B is compared with llama.cpp only.

### 43.2 Gap table

Output tok/s, higher is better; "SL / other" uses the after column.

| model | weights | streams | SL before | SL after | other engine | other | SL / other | TTFT p50 SL / other (ms) | decode tok/s per stream SL / other |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| Qwen3-30B-A3B | GGUF Q4_K_M | 1 | 211 | 211 | llama.cpp | 203 | 1.04x | 41 / 78 | 225 / 231 |
| | | 4 | 421 | 427 | llama.cpp | 274 | 1.56x | 67 / 238 | 113 / 79 |
| | | 16 | 646 | **958** | llama.cpp | 428 | 2.24x | 135 / 822 | 65 / 32 |
| | | 64 | 1548 | 1568 | llama.cpp | 849 | 1.85x | 254 / 1029 | 28 / 15 |
| gpt-oss-20b | GGUF MXFP4 | 1 | 231 | 234 | llama.cpp | 184 | 1.27x | 33 / 53 | 248 / 199 |
| | | 4 | 478 | 483 | llama.cpp | 299 | 1.61x | 69 / 146 | 127 / 82 |
| | | 16 | 675 | **1148** | llama.cpp | 439 | 2.61x | 106 / 608 | 78 / 31 |
| | | 64 | 1953 | 2012 | llama.cpp | 792 | 2.54x | 180 / 891 | 36 / 14 |
| gpt-oss-20b | HF MXFP4 | 1 | 178 | 179 | vLLM | 169 | 1.06x | 33 / 29 | 186 / 175 |
| | | 4 | 383 | 384 | vLLM | 503 | 0.76x | 63 / 72 | 100 / 133 |
| | | 16 | 602 | **1078** | vLLM | 1440 | 0.75x | 108 / 166 | 73 / 102 |
| | | 64 | 1916 | 1996 | vLLM | 3243 | 0.62x | 181 / 372 | 36 / 61 |
| Qwen3.6-35B-A3B | GGUF UD-Q4_K_M | 1 | | 193 | llama.cpp | 149 | 1.30x | 41 / 157 | 205 / 181 |
| | | 4 | | 435 | llama.cpp | 148 | 2.95x | 109 / 549 | 118 / 59 |
| | | 16 | | 778 | llama.cpp | 113 | 6.89x | 145 / 12035 | 52 / 24 |
| | | 64 | | 1188 | llama.cpp | 303 | 3.93x | 270 / 4896 | 21 / 8 |

Qwen3.6-35B-A3B (256 experts: the per-item bound stays 128, so before = after) was measured with the after build only.
llama-server serves its hybrid delta-rule / attention layers poorly under concurrency (TTFT 12 s at 16 streams).

* Against llama.cpp on the same GGUF Sharplight serves 1.04-2.6x the throughput (Qwen3.6: 1.3-6.9x) with a lower TTFT everywhere (llama-server's
  64-stream rounds of gpt-oss spread 533..1183 tok/s). The single-stream decode of Qwen3-30B is level (225 against 231 tok/s).
* Against vLLM on the same HF checkpoint Sharplight leads at one stream (1.06x, lower ITL) and is behind from 4 streams on
  (0.62-0.76x after this round, 0.42-0.76x before). TTFT is lower than vLLM's from 4 streams on.
* The HF checkpoint decodes slower than the GGUF on Sharplight (179 against 234 tok/s single stream) because its attention
  projections and LM head are bf16 where the GGUF has Q8_0: the bf16 LM head [2880 → 201088] alone takes 1.34 ms of the
  5.3 ms step. vLLM reads the same bf16 weights.

### 43.3 Lever 1: expert products of 16-64 sequences on the grouped GEMM (per-item matvec up to half the expert count)

The batched decode step of a MoE model ran its expert products on the per-item int-dot matvec up to 128 (token, slot) items and on the
tensor-core grouped GEMM above. The per-item matvec reads an expert's weights once for every item routed to it; the grouped GEMM
reads them once per expert tile. With 128 items and 32 experts (gpt-oss-20b at 32 sequences) every expert is read about 4 times.
`BatchBench` (64 slots, 256-token contexts, greedy on the device, minimum GPU ms per step of 20) with the per-item bound forced to
128 / 64 / 32 / 16 items (`SHARPLIGHT_MOE_MMVQ_MAX`):

| model (experts x slots) | sequences | items | per-item ≤ 128 (before) | ≤ 64 | ≤ 32 | ≤ 16 | after (≤ experts / 2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B Q4_K_M (128 x 8) | 4 | 32 | 8.68 | 8.69 | 8.70 | 10.75 | 8.67 |
| | 8 | 64 | 13.71 | 13.69 | 13.53 | 13.51 | 13.68 |
| | 12 | 96 | 16.26 | 13.51 | 13.57 | 13.55 | 13.48 |
| | 16 | 128 | 20.83 | 15.46 | 15.49 | 15.44 | 15.40 |
| gpt-oss-20b MXFP4 GGUF (32 x 4) | 4 | 16 | 7.30 | 7.30 | 7.28 | 7.28 | 7.28 |
| | 8 | 32 | 11.68 | 11.66 | 11.68 | 9.02 | 9.01 |
| | 16 | 64 | 20.43 | 20.38 | 10.68 | 10.62 | 10.61 |
| | 24 | 96 | 29.65 | 12.26 | 12.25 | 12.25 | 12.20 |
| | 32 | 128 | 38.49 | 13.75 | 13.77 | 13.77 | 13.72 |
| gpt-oss-20b HF MXFP4 (32 x 4) | 8 | 32 | 14.17 | 14.07 | 14.17 | 10.44 | 10.41 |
| | 16 | 64 | 23.49 | 23.52 | 11.49 | 11.49 | 11.44 |
| | 32 | 128 | 43.39 | 14.23 | 14.22 | 14.24 | 14.20 |

The crossover sits at about one item per two experts on both models (Qwen3-30B: 32 items per-item 8.7 ms against the GEMM's 10.8, 64
items level; gpt-oss: 16 items level, 32 items 11.7 against 9.0). The per-item bound is now half the expert count, clamped to
16-128, when the tensor-core grouped GEMM serves the larger steps; without tensor cores (scalar `moe_gemm`) it stays 128. Models with
256 experts (Qwen3.6-35B-A3B) keep 128. The bound still covers the row-invariant speculative verification rows
(`InvariantMatvecItems`), so MTP / n-gram verification stays bitwise equal to single-row decoding.

Tried first and not kept: a **grouped int-dot matvec** (`moe_mmvq` reading the expert-sorted routing, one launch row per expert tile
of up to 8 items, each weight chunk loaded once for the tile's items, the fused gate / up epilogue kept; bitwise equal to the per-item
matvec). On Qwen3-30B-A3B it took 39.8 / 62.0 ms at 32 / 64 sequences against the GEMM's 18.8 / 25.4: once an expert sees several items the
int8 dot products and the per-chunk K-quant unpacking (repeated for every item), not the weight traffic, appear to bound it (not
profiled further).

Serving effect (43.2): 16 streams Qwen3-30B-A3B 646 → 958 tok/s (+48 %), gpt-oss-20b GGUF 675 → 1148 (+70 %), HF 602 → 1078
(+79 %; against vLLM 0.42x → 0.75x). 1, 4 and 64 streams are unchanged within 1-4 % (4 streams stay on the per-item matvec,
64 streams were already on the GEMM).

Greedy output: single-stream decoding and speculative verification rows run exactly the kernels they ran before (at most 8 / 4
items per token, below every bound, and the row-invariant rows stay on the matvec), so their outputs are unchanged by
construction. Batched steps of 4-16 sequences now take the f16 tensor-core GEMM where they took the int8 matvec, as steps of more
than 16 sequences already did: those rows round differently (within the tolerances of `MoeTests`, new
`MatvecRangeFollowsExpertCount`).

### 43.4 Where the time goes against vLLM (gpt-oss-20b HF)

`BatchBench` profile of the decode step (16 sequences, after; `SHARPLIGHT_VK_PROFILE=1`), 11.1 GPU ms per step:

| op | ms per step |
|---|---:|
| expert GEMMs (fused gate / up [2880 → 5760] and down, MXFP4, `gemm_cm2` grouped) | 6.8 |
| LM head [2880 → 201088] bf16 | 1.38 |
| q / k / v / o projections bf16 | 2.2 |
| attention (decode + combine) | 0.62 |
| routing, gathers, norms, elementwise, top-k, sampling | ~1.0 (2400 small `fused[Add]` launches alone 0.33) |

Host overhead is not the gap: the engine's step metric covers 99.4-99.6 % of the wall time at 64 streams (`metrics_delta` in the
raw files). In serving, a 64-stream round is 192 requests x (250 prompt + 128 output) tokens: 422 steps of ~171 rows (64 decode
rows plus chunked prefill of ~3 prompts), 29 ms each, 12.3 s per round against vLLM's 7.6 s. The decode-only step at 64 sequences
is 18.5 ms (`BatchBench`), so about a third of the gap is the mixed prefill / decode steps and the rest the expert GEMMs and the bf16
projections at 64-171 rows. vLLM's Marlin MXFP4 MoE reads the experts once per step for all rows with gate and up fused; ours
fuses gate / up only in the checkpoint's own layout (HF) and runs the routing, the f16 gather of x and the SiLU epilogue as
separate launches.

### 43.5 Not measured: Qwen3.8-Flash-Next with CPU offload

The UD-IQ1_S shards (66 GB) exceed the 60 GB local-disk budget of this round, and the shared network filesystem delivered
1-2 MB/s for them during the session (a pre-read reached 16 GB in three hours; a 22 GB GGUF load through it hung in
`fuse_get_req` and was killed). The Flash-Next comparison against llama.cpp (same split, MTP) is left for a session with the
files on local disk; section 17.5 has the last single-GPU + CPU figures.

### 43.6 Tests

QWEN36_TESTS

### 43.7 Next levers

* **Mixed prefill / decode steps of MoE models** (gpt-oss at 64 streams: 29 ms per 171-row step against 18.5 ms for 64 decode
  rows): the prompt chunks make the expert GEMMs taller; a token budget per step that keeps the GEMM tiles full, or decode-priority
  scheduling with larger, rarer prefill chunks.
* **Expert GEMM against Marlin**: gate / up as one dual GEMM with the SiLU-clamp epilogue (as the int-dot matvec already does),
  the routing and the f16 gather in one launch, MXFP4 fetch at the bandwidth limit for 2-8 rows per expert.
* **bf16 LM head / projections at decode** (HF checkpoints): 1.38 ms of 11.1 for the LM head at 16 sequences; FP8 or Q8 weight
  copies are an accuracy trade, so first the bf16 skinny GEMM's own bandwidth.
* Qwen3.8-Flash-Next with local files (43.5).

### 43.8 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2 GGUF_DIR=<gguf files> HF_DIR=<hf checkpoints> LLAMA_DIR=<llama.cpp runtimes>
export VLLM_VENV=<venv with vLLM 0.33.0> LOADGEN_PY=python3 READY_TIMEOUT=10800   # vLLM's first start JIT-compiles FlashInfer
bench/s43_serve.sh bench/results/s43/serve-before qwen3-30b-a3b gpt-oss-20b gpt-oss-20b-hf qwen3.6-35b-a3b   # base build
ENGINES=sharplight SHARPLIGHT_SERVER=<after build>/sharplight bench/s43_serve.sh bench/results/s43/serve-after qwen3-30b-a3b gpt-oss-20b gpt-oss-20b-hf
python3 bench/s43_serve_summarize.py bench/results/s43/serve-before bench/results/s43/serve-after
# per-item bound sweep (43.3):
SHARPLIGHT_MOE_MMVQ_MAX=64 bench/BatchBench/bin/Release/net11.0/BatchBench <model> --slots 64 --ctx 512 --depth 256 \
  --sizes 1,2,4,8,12,16,24,32,64 --modes greedy --reps 20
```

## 44. LLM round 7: norms writing the GEMMs' f16 / E4M3 activations, Gemma 3 from Hugging Face, K/V staging and GEMM tiles examined (2026-11-18)

Base: `develop` at `138aaee8` (rounds 6 and 6b merged), branch `perf/llm-r7`. GPU 2 (RTX PRO 4500 Blackwell, 32 GB, nothing else
on it), checkpoints on local disk. Raw output in [`bench/results/s44/`](../bench/results/s44).

### 44.1 Gap table (procedure of section 42)

`bench/s42_gap.sh` per model and prompt length (one warm-up, median of 5 prefills; decode median of 64 steps). "Before" is the
same build with `SHARPLIGHT_NORM_PREFILL_COPY=0` (44.2 off, everything else identical), run back to back with "after". The
vLLM column is section 42's (vLLM 0.33.0, same checkpoints, GPU 1 of the same model); vLLM was not rerun here: the shared network
file system that holds its virtualenv stalled for most of the session (directory listings of 10-30 s). Session-to-session
spread on Sharplight is a few percent (section 42's Qwen3-8B-FP8 pp2048 was 10673 tok/s, this session's "before" 11045), so the
"after / vLLM" ratios carry that uncertainty; the before / after gain is measured within one session.

| model | weights | metric | before | after | after / before | vLLM (§42) | after / vLLM |
|---|---|---|---:|---:|---:|---:|---:|
| qwen3-8b | HF bf16 | pp512 | 7532 | 7542 | 1.00x | 8203 | 0.92x |
| | | pp2048 | 7820 | 7953 | 1.02x | 8782 | 0.91x |
| | | pp8192 | 6940 | 7051 | 1.02x | 7661 | 0.92x |
| | | tg (after 512) | 52.3 | 52.7 | | 52 | 1.01x |
| qwen3-8b-fp8 | HF FP8 | pp512 | 10989 | 11207 | 1.02x | 14446 | 0.78x |
| | | pp2048 | 11045 | 11344 | 1.03x | 15024 | 0.76x |
| | | pp8192 | 9352 | 9557 | 1.02x | 12346 | 0.77x |
| | | tg (after 512) | 87.6 | 88.4 | | 87 | 1.02x |
| llama3.1-8b | HF bf16 | pp512 | 7687 | 7752 | 1.01x | 8532 | 0.91x |
| | | pp2048 | 7918 | 8047 | 1.02x | 8836 | 0.91x |
| | | pp8192 | 7113 | 7218 | 1.01x | 7806 | 0.92x |
| | | tg (after 512) | 53.3 | 53.6 | | 53 | 1.01x |
| gemma-3-12b | HF bf16 (new, 44.4) | pp512 | 4356 | 4370 | 1.00x | not run | |
| | | pp2048 | 4425 | 4447 | 1.00x | not run | |
| | | pp8192 | 3646 | 3656 | 1.00x | not run | |
| | | tg (after 512) | 33.4 | 33.5 | | not run | |

A/B runs alternating the switch (2048 tokens, median of 5 per run): Qwen3-8B-FP8 186.4 → 179.7 ms (4 runs each, -3.6 %), Qwen3-8B
bf16 262.3 → 258.0 ms (3 each, -1.6 %), Llama-3.1-8B Q4_K_M (GGUF, f16 activations into gemm_cm2) 290.0 → 287.6 ms (2 each, -0.8 %).
Gemma 3 gains least: its norms feed the GEMMs only after its extra sandwich norms and the GELU path, and its pass is dominated by
the 48 layers' GEMMs.

### 44.2 Row norms write the GEMMs' activations at prefill sizes (kept)

At prefill sizes the residual add + RMS norm wrote f32 rows, and a separate launch converted them for the tensor-core GEMMs
(`to_f16`, or `quantize_fp8` for the fp8 GEMM: E4M3 codes plus one scale per 128 values). Profile of Qwen3-8B-FP8 at 2048 rows
(`SHARPLIGHT_VK_PROFILE=1`): norms 10.25 ms (72 launches), quantization of the norm outputs 2.05 ms, GPU time 179.7 ms.

* From 128 rows (`SHARPLIGHT_NORM_COPY_MIN_ROWS`), a norm whose every reader is a GEMM that takes one activation form writes that
  form itself: the f16 copy (existing small-step mechanism, extended), or the E4M3 rows of the fp8 GEMM. For E4M3 the normalized
  row passes through shared memory (4 bytes per value), and each 32-lane subgroup then quantizes 128 consecutive values with
  quantize_fp8's arithmetic (now shared in `e4m3.slang`), so the codes and scales are bit-identical to the separate launch.
* The f32 rows are skipped from 128 rows when, once the whole segment is lowered, every reader of the value was found to take the
  copy for all steps of 128 rows or more (recorded where the GEMMs ask for their activations; any other reader, a segment output or
  a view the GEMM cannot read keeps the f32 rows). The norm launch has two variants, chosen after lowering.
* Profile after: norms 7.44 ms, the two quantizations gone (the o-projection's input, written by attention, is still quantized
  separately: 0.63 ms), GPU time 174.3 ms. bf16 Qwen3-8B: conversions 2.6 → 0.78 ms, norms 9.27 → 7.27 ms, GPU 254.4 → 250.6 ms.
* Writing the f16 copy next to the f32 rows (without skipping them) was measured first: the conversions disappear but the norms
  grow by the same 1.9 ms (GPU time 254.41 → 254.49 ms); the gain needs the f32 rows gone.

Greedy output: 200-token greedy outputs of the short, ~1100-token and ~4500-token prompts (`--ctx 8192`) on Qwen3-8B and
Qwen3-8B-FP8 with the switch off and on: byte-identical (all 6). Tests: `NormHalfCopyTests.NormThenPrefillGemms` (new: f8e4m3_b128,
bf16, f16, q4_k at 100 / 130 / 300 rows, with and without the norm output also read as a graph output).

### 44.3 Tried and not kept

| tried | result |
|---|---|
| Prefill attention `fe` (register softmax): K and V tiles staged separately, the K tile of tile j + 1 after the scores of tile j, the V tile after its P·V, loads issued as soon as registers free (two barriers per tile as before, no store sandwiched between them) | attention at 8192 tokens (Qwen3-8B-FP8, 144 launches) 170.8 → 192.3 ms: slower |
| `fe` with 32-key tiles | 182.8 ms against 170.8 |
| `fe` diagnostic: no K/V fetch or staging at all (`FE_EXP=1`) | 132.0 ms: staging and fetch cost 39 ms of 171 (23 %), the bound for any K/V pipelining |

Why `fe` resists: it runs at 255 registers per thread (`SHARPLIGHT_VK_PIPELINE_STATS=1`), so one 256-thread workgroup fills an
SM's register file (8 warps per SM). The output accumulators (64 registers), Q fragments (32), scores (32) and the K/V prefetch (32)
leave no room for a second workgroup or for a second set of tiles, and the 48 KB shared-memory limit (`maxComputeSharedMemorySize`
on this driver) holds one 64-key K/V tile pair (34.8 KB). Attention is 6.4 % of the pass at 2048 tokens and 20 % at 8192 for
Qwen3-8B-FP8, so the 8192-token gap to vLLM now has a large attention share.

GEMM tiles (lever 1 of the brief, examined, not built): section 42.3 found the FP8 GEMM bounded by L2 traffic against MMAs of the same
duration. Less L2 traffic per FLOP needs a tile with more than 128 x 128 outputs; with 128-value K steps a 256 x 128 FP8 tile needs
55 KB of shared memory per stage (limit 48 KB), and with 64-value steps the per-block scaling needs the block accumulators across
two steps (a second accumulator set: 128 + 64 registers of accumulators per thread at 256 threads, past the 255-register limit that
already spilled the 4-subgroup 128 x 128 variant). The bf16 GEMM's 256 x 128 tile fits only with 32-value single-buffered steps,
the configuration that section 40.4 measured 28 % slower than 64-value steps. Neither was pursued; asynchronous copies (not exposed
by the Vulkan driver) or larger shared memory would change this.

### 44.4 Gemma 3 from Hugging Face

The HF importer now takes Gemma 3's vision-language checkpoints (`Gemma3ForConditionalGeneration`, model type `gemma3`, e.g.
gemma-3-12b-it) as their text decoder: `language_model.model.*` (and the newer `model.language_model.*`) map like
`gemma3_text`, the vision tower and projector are skipped. Their `text_config` lists only non-default values (the 12B's has no
`head_dim`, `rope_theta`, `query_pre_attn_scalar` ...), so `HfConfig` fills transformers' `Gemma3TextConfig` defaults for keys a
nested `gemma3_text` config leaves out.

* gemma-3-12b-it, "The capital of France is", 12 greedy tokens: " Paris.\nThe capital of Germany is Berlin.\nThe", the same ids as
  transformers 5.23 (fp32 on CPU, `tools/hf_greedy_reference.py`).
* Families case `tiny-random-gemma-3` (the same wrapper layout) against transformers' greedy ids.
* Prefill 4447 tok/s at 2048 tokens, decode 33.5 tok/s (44.1). vLLM was not run on it (44.1).

### 44.5 Next levers

* Attention at 8192 tokens (20 % of the FP8 pass): a register budget that allows two `fe` workgroups per SM (Q fragments reloaded from
  shared memory, fewer prefetch registers), or the o-projection's input written by attention as E4M3 / f16 directly (0.63 / 0.78 ms
  per 2048-token pass).
* The down projection's E4M3 input (`quantize_fp8` of the SwiGLU output, 2.5 % of the FP8 pass): the dual GEMM's tile covers 64 of
  the 128 values of a scale group.
* GEMM: see 44.3; vLLM serving and prefill on the same session once the network file system allows (dense serving keys
  `qwen3-8b-fp8-hf` / `qwen3-8b-hf` added to `bench/s43_serve.sh`).

### 44.6 Tests

After merging `develop` (at `b0fa1808`) into the branch, on GPU 2:

| suite | total | failed | skipped |
|---|---:|---:|---:|
| Sharplight.Vulkan.Tests (full) | 1558 | 0 | 23 |
| Sharplight.Families.Tests (full) | 1005 | 8 (see below) | 364 |
| Sharplight.Import.HuggingFace.Tests (full) | 315 | 0 | 101 |

The 8 Families failures (7 `Batched_GreedyOutput_Matches_SingleSequence` Vulkan cases, `EmbeddingFamilyTests` modernbert on Vulkan)
were device-memory exhaustion (`InsufficientMemoryException` / `ErrorOutOfDeviceMemory`) while the importer suite ran on the same
GPU; rerun alone, those classes pass (63 tests, 0 failed). Raw profiles (`profiles/`) and the greedy outputs of 44.2 (`greedy/`,
`nc0` = switch off, `nc1` = on) are in `bench/results/s44/`.

### 44.7 Reproducing

```bash
# gap rows (44.1): before = SHARPLIGHT_NORM_PREFILL_COPY=0
SHARPLIGHT_NORM_PREFILL_COPY=0 bench/s42_gap.sh bench/results/s44/gap qwen3-8b-fp8_before sharplight ~/models/hf/Qwen3-8B-FP8
bench/s42_gap.sh bench/results/s44/gap qwen3-8b-fp8 sharplight ~/models/hf/Qwen3-8B-FP8
# attention experiments (44.3)
SHARPLIGHT_FE_MACROS=FE_EXP=1 SHARPLIGHT_VK_PROFILE=1 sharplight ~/models/hf/Qwen3-8B-FP8 --bench 8192 --ctx 8448
```

## 45. Range guard of the f16 activation copies (2026-11-18)

Bug found in the Qwen3-TTS bring-up: with bf16 weights, its code predictor (a 5-layer Qwen3 decoder fed the talker's hidden
state, outliers near 95) gave wrong logits on Vulkan (490 of 705 argmax disagreements with the reference, worst top-logit error
37) while the CPU matched. Bisect: only `SHARPLIGHT_SKINNY_GEMM=0` (or `SHARPLIGHT_ROW_INVARIANT=8`, which moves the 2-row steps
off the skinny GEMM) helped; the norm copies, the bf16 → f16 weight decode and the accumulators were not involved. A replay of the
predictor in f32 showed layer 2's gated FFN reaching gate 343 and up 427, so silu(g)·u = 1.46e5 at the down projection's input:
beyond f16's 65504. Every tensor-core GEMM reads f16 copies of its activations (`to_f16`, the dual gemm_cm2's `DUAL_Y16` copy,
the skinny GEMM's staging), so those rows became infinities and NaN. The talker never reached 2^15 (and its single-row decode
steps run on the f32 matvec), so only the predictor's 2-row steps showed it.

Fix (generic, all devices with the f16 tensor-core path): copies of values that no norm bounds are row-ranged. A row whose largest
magnitude reaches 2^15 is stored multiplied by a power of two (into [2^14, 2^15)), with a scale word per row after the rows, and
the GEMM outputs of that row are multiplied by the inverse (exact). Rows below 2^15 are stored as before, so outputs are
bit-identical for every model that stays in range. Producers: `to_f16_ranged` (one workgroup per row, rows up to 8192 values held
in registers: one read), and the dual gemm_cm2's f16 copy, which raises per-row flags (atomic max of its subgroup tile's largest
magnitude) that `f16_range_fixup` turns into rescaled rows and scale words. Consumers: the skinny GEMM in its store (the
f32-staging variant finds each row's range itself), gemm_cm2 and gemm_nv2 through `f16_range_unscale` after the launch (one word
read per row when nothing is scaled), split-K in `splitk_reduce`. Norm-written copies stay plain (a norm's output is at most √K
times its largest weight), and a dual gemm_cm2 whose x would be a row-ranged copy is not used. `SHARPLIGHT_F16_RANGE=0` restores
plain copies.

Cost, Qwen3-8B bf16 at 2048 rows (GPU 2, `SHARPLIGHT_VK_PROFILE=1`): GPU time 249.6 → 251.3 ms (dual gemm_cm2 +0.5 ms, the
three small launches 0.5 ms, `to_f16_ranged` as fast as `to_f16`); wall clock, 3 alternating runs of `--bench 2048` each:
prefill median of 5 258.6 → 259.2 ms, decode 52.3 → 52.2 tok/s (single rows use the f32 matvec: unchanged). A first version that
scaled the accumulators inside gemm_cm2 cost 5 % (more shared memory and registers in the 128-row tiles) and was replaced.

Qwen3-TTS now keeps bf16 code-predictor weights (the importer's f32 workaround is gone): 0 disagreements, worst top-logit error
0.015 on Vulkan (0.016 on CPU). Test: `F16RangeTests` (Vulkan.Tests): matmuls with rows up to 1.5e5 on the skinny GEMM, split-K
and gemm_cm2, and a tiny FFN block fed hidden states with outliers of 95 / -60 whose product reaches 2^17, on the skinny GEMMs,
split-K and the dual gemm_cm2 with its flagged copy; all 13 cases gave NaN before.

Not covered: the MoE grouped GEMM's sorted f16 rows (`moe_gather_x16`) and the int8 / FP8 paths (their per-block scales keep the
range, at their usual precision). Seen on the way: with `SHARPLIGHT_SKINNY_X16=0` (diagnostic), matmuls of fewer than 4096 outputs
return zeros for 9-16 rows (no f16 copy is made for that range); unrelated to this change.

## 46. LLM round 8: fresh vLLM baseline, expert-grouped skinny GEMM for MoE decode, MoE range guard (2026-11-18)

Base: `develop` at `6420f2ea` ("before", built in a separate worktree), branch `perf/llm-r8` ("after"). GPU 2 (RTX PRO 4500
Blackwell, 32 GB, nothing else on it), vLLM 0.33.0 in a fresh virtualenv on local disk, checkpoints on local disk. Raw output in
[`bench/results/s46/`](../bench/results/s46).

### 46.1 Fresh vLLM baseline (procedures of sections 42 and 43)

Round 7 could not rerun vLLM (its virtualenv sat on the stalling network file system). Rerun here with the identical procedures
(`bench/s42_gap.sh`, `bench/s43_serve.sh`); the Sharplight columns are this branch (its changes do not touch prefill).

| model | weights | metric | Sharplight | vLLM (§46) | SL / vLLM | vLLM (§42) |
|---|---|---|---:|---:|---:|---:|
| qwen3-8b | HF bf16 | pp512 | 7292 | 8478 | 0.86x | 8203 |
| | | pp2048 | 7860 | 9059 | 0.87x | 8782 |
| | | pp8192 | 6961 | 7887 | 0.88x | 7661 |
| | | tg (SL after 512 / vLLM depth 16) | 53 | 52 | 1.02x | 52 |
| | | tg@d8192 | 49 | 48 | 1.02x | 48 |
| qwen3-8b-fp8 | HF FP8 | pp512 | 10630 | 15281 | 0.70x | 14446 |
| | | pp2048 | 11154 | 15308 | 0.73x | 15024 |
| | | pp8192 | 9436 | 12577 | 0.75x | 12346 |
| | | tg (after 512 / depth 16) | 89 | 87 | 1.02x | 87 |
| | | tg@d8192 | 79 | 78 | 1.02x | 77 |
| llama3.1-8b | HF bf16 | pp512 | 7630 | 8699 | 0.88x | 8532 |
| | | pp2048 | 7918 | 8992 | 0.88x | 8836 |
| | | pp8192 | 7113 | 7959 | 0.89x | 7806 |
| | | tg (after 512 / depth 16) | 54 | 53 | 1.02x | 53 |
| | | tg@d8192 | 50 | 49 | 1.02x | 49 |

vLLM measures 1.5-6 % faster prefill than in section 42 (other GPU of the same model, older driver state), so section 44's
"after / vLLM" ratios, taken against the section 42 figures, were 0.01-0.04x optimistic: dense bf16 is at 0.86-0.89x, FP8 at
0.70-0.75x, decode level (1.02x).

Serving, gpt-oss-20b HF (median of 3 rounds; vLLM 0.33.0 Marlin MXFP4, same procedure as 43.1):

| streams | SL before | SL after | vLLM (§46) | after / vLLM | vLLM (§43) | TTFT p50 after / vLLM (ms) | per-stream decode after / vLLM |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 178 | 177 | 169 | 1.05x | 169 | 33 / 31 | 184 / 175 |
| 4 | 382 | **470** | 501 | 0.94x | 503 | 71 / 77 | 124 / 134 |
| 16 | 1075 | **1201** | 1428 | 0.84x | 1440 | 109 / 175 | 82 / 102 |
| 64 | 1977 | 1979 | 3221 | 0.61x | 3243 | 184 / 381 | 35 / 61 |

Serving, Qwen3-8B-FP8 HF (dense, first measured against vLLM with the keys added in section 44; this branch, median of 3):

| streams | Sharplight | vLLM | SL / vLLM | TTFT p50 SL / vLLM (ms) | per-stream decode SL / vLLM |
|---:|---:|---:|---:|---|---|
| 1 | 88 | 86 | 1.02x | 34 / 32 | 90 / 87 |
| 4 | 305 | 326 | 0.94x | 70 / 72 | 79 / 85 |
| 16 | 934 | 1123 | 0.83x | 104 / 157 | 62 / 78 |
| 64 | 2036 | 2717 | 0.75x | 193 / 391 | 36 / 50 |

### 46.2 Where the time goes (gpt-oss-20b HF, decode steps)

`BatchBench` (64 slots, 256-token contexts, greedy on the device), per-op profile of the step before this round
(`SHARPLIGHT_VK_PROFILE=1`, `bench/profile_categories.py`):

| | 16 sequences | 64 sequences |
|---|---:|---:|
| GPU ms per step | 13.7 | 20.2 |
| MoE (expert GEMMs, routing, gather) | 58 % | 57 % |
| dense GEMM / matvec (bf16 q / k / v / o, router, LM head) | 27 % | 21 % |
| attention | 4.6 % | 10.7 % |
| norms / elementwise / rope / cache | 7.0 % | 7.0 % |

The expert GEMMs, measured alone (`BenchTests.MoeDecode`, new: gate / up [2880 → 5760] + down [2880 → 2880] MXFP4 on uniform
routing, 6 weight copies so the experts come from memory), read the touched experts at 600 GB/s at 16-64 sequences (the card
streams bf16 at ~860 GB/s, the 1-row MXFP4 matvec at ~800): gemm_cm2's 32-row tiles with 2-8 rows per expert, one barrier per
K step. Host overhead: the step metric covers the wall time as before; recording a new step shape costs 2-20 ms and a 64-stream
loadgen round met ~75 new shapes with the first-in first-out cache of 64 recordings (46.5).

### 46.3 Expert-grouped skinny GEMM (kept)

The skinny GEMM (`gemm_sk`, 2-16 rows, K split across the subgroups of a workgroup, no barrier per K step) got an expert-grouped
form (`MOE` with `X_SORTED`): each workgroup row of the grid is a 16-row expert tile of the routing scratch, its rows the
expert-sorted f16 rows of `moe_gather_x16`, its output row the routed item. On the direct path (operands loaded straight into
the MMA fragments, `matrix_mma_layout`) MXFP4R weights are now fetched raw (codes and scale byte) and decoded after the MMAs,
like FP8; workgroups take two 16-row weight fragments and 4 subgroups.

| `MoeDecode`, µs per expert block (gate / up + down) | 16 seq. | 64 seq. | 96 seq. | 128 seq. |
|---|---:|---:|---:|---:|
| gemm_cm2 (before) | 642 | 720 | 731 | 749 |
| skinny, staged | 616 | 748 | | |
| skinny, direct, 1 fragment x 8 subgroups | 557 | 709 | | |
| skinny, direct, 2 x 4 (kept) | **495** | **562** | 615 | 759 |
| skinny, direct, 4 x 4 / 2 x 8 / 1 x 4 | 508 / 512 / 487 | 579 / 601 / 582 | | |

Selection (`GraphLowerer.Moe.cs`): the skinny form serves up to 12 items per expert on the direct path (MXFP4, Q8_0, FP8) and up
to one item per expert on the staged path (other formats: Qwen3-30B-A3B Q4_K gains at 8-16 sequences but loses 5 % at 32); the
per-item int-dot matvec now hands over at a quarter of the expert count (8-128 items) when the skinny form follows (gpt-oss at
3 / 4 sequences: 8.04 / 9.19 → 7.48 / 7.67 ms). Also: 16-bit weights with 256-1023 outputs (gpt-oss's k / v projections) take the
skinny GEMM up to 8 rows instead of the f32 split-K matvec, which re-reads the weights per row (bf16 [2880 → 512] at 8 rows 23.3 →
7.9 µs).

Decode step, GPU ms (BatchBench, greedy, minimum of 20; "before" is the base build for gpt-oss and this branch with
`SHARPLIGHT_MOE_SKINNY=0` and the old per-item bound for gpt-oss at 8 sequences and for Qwen3-30B-A3B):

| sequences | gpt-oss-20b HF before | after | Qwen3-30B-A3B Q4_K_M before | after |
|---:|---:|---:|---:|---:|
| 1 | 5.30 | 5.29 | 4.27 | 4.26 |
| 2 | | 6.79 | 5.61 | 5.65 |
| 4 | 9.32 | **7.73** | 8.57 | 8.66 |
| 8 | 10.97 | **8.58** | 13.71 | **12.45** |
| 16 | 12.03 | **9.90** | 15.12 | **14.30** |
| 32 | 13.87 | **12.83** | 18.89 | 18.55 |
| 64 | 18.65 | 18.70 | | 25.36 |

At 64 sequences BatchBench's routing (random tokens) concentrates on few experts, which then take several 16-row tiles; with the
uniform routing of `MoeDecode` the same step size gains 22 % (720 → 562 µs per block). Serving at 64 streams does not move (46.1):
its steps are 64 decode rows plus ~107 prompt rows (422 steps of 171 rows per round), and the prompt rows put 20-30 items on each
expert, the range of gemm_cm2.

Greedy output: single-stream decoding and speculative verification rows run the kernels they ran before (the row-invariant
items stay on the per-item matvec), so their outputs are unchanged. Batched steps of 3-32 sequences of models whose expert
products run the skinny form now round like the skinny GEMM (K split across subgroups) instead of the int8 matvec or gemm_cm2;
`MoeTests.ExpertGroupedSkinny` (new) checks every format against the CPU, and the Families batched greedy tests pass (46.7).

### 46.4 Tried and not kept

| tried | result |
|---|---|
| Two token fragments (32-row expert tiles) on the direct skinny path for 12-24 items per expert | 128 / 171 sequences 749 / 769 → 754 / 793 µs per block |
| 128-row routing tiles launched per height class (≤16 skinny, 17-32 / 33-64 / 65-128 gemm_cm2), so no expert is read by two launches | 171 / 256 / 512 / 1024 sequences 767 / 987 / 1573 / 2514 → 794 / 1010 / 1796 / 2683 µs: the split-tile launches' second read hits L2; at 20-128 items per expert the grouped MXFP4 GEMM is bound by its MMA / decode work (52-81 TFLOPS), not by weight traffic |
| MXFP4 decode removed from gemm_cm2's raw path (diagnostic, wrong results) | 256 sequences 989 → 904 µs: the decode is ~9 % of the mid-size expert GEMM |
| Forced 64 / 128-row expert tiles on mixed steps (60 decode + 128 / 256 prompt rows) | 30.6 / 39.4 ms → 29.7 / 39.6 (64) and 32.7 / 42.6 (128): noise or slower |

### 46.5 Recording cache (kept)

Recorded steps are now evicted least recently used and 256 are kept per segment (was 64, first in first out): a 64-stream
gpt-oss round records ~75 shapes (2-20 ms each) while it drains, which later rounds now find recorded. Small: 3 rounds at 16 / 64
streams 1186 / 1220 / 1227 and 1983 / 1991 / 2020 tok/s (the first round pays the recordings).

### 46.6 Range guard of the MoE grouped GEMM's sorted f16 rows; f16 copy with `SHARPLIGHT_SKINNY_X16=0`

* `moe_gather_x16` (the expert-sorted f16 rows the grouped gemm_cm2 and the skinny form read) is row-ranged like the dense copies
  (section 45): one workgroup per sorted row finds its largest magnitude (`f16_range_row`, shared with `to_f16_ranged`), a row
  reaching 2^15 is stored scaled with a scale word per sorted position, and the GEMMs multiply each row's outputs by the inverse
  in their store (exact; rows below 2^15 are stored and computed exactly as before, the multiply by 1 is exact).
  `SHARPLIGHT_F16_RANGE=0` restores the plain gather. Test: `F16RangeTests.Expert_gemm_reads_rows_beyond_f16_range` (bf16,
  f16, MXFP4; per-slot and shared rows; 32-, 64- and 128-row tiles): NaN with the guard off, within tolerance with it.
  Not covered: `moe_gemm_cm` (devices with 8x8 fragments) and `SHARPLIGHT_MOE_X16=0`, which convert f32 rows in the tile.
* With `SHARPLIGHT_SKINNY_X16=0`, the shared f16 copy started above the skinny range of 4096-wide matrices (17 rows) for every
  width, while narrower matrices hand 9-16 rows to gemm_cm2, which read the unwritten copy (zeros). The copy now starts above
  the skinny range of the matmul's own width. Test: `SkinnyGemmTests.NarrowAboveSkinnyRange` (512-4096 outputs at 9-16 rows)
  and `NarrowAboveSkinnyRangeWithF32Staging`, which reruns it in a child process with `SHARPLIGHT_SKINNY_X16=0` (the switch is
  read once per process).

### 46.7 Tests

After merging `develop` (at `823c72e7`) into the branch, on GPU 2:

| suite | total | failed | skipped |
|---|---:|---:|---:|
| Sharplight.Vulkan.Tests (full) | 1594 | 0 | 25 |
| Sharplight.Families.Tests (full, batched greedy parity included) | 1005 | 0 | 364 |

New or changed: `MoeTests.ExpertGroupedSkinny` (8 cases), `MoeTests.MatvecRangeFollowsExpertCount` (new bounds),
`F16RangeTests.Expert_gemm_reads_rows_beyond_f16_range` (5), `SkinnyGemmTests.NarrowAboveSkinnyRange` (5) and
`NarrowAboveSkinnyRangeWithF32Staging`, `BenchTests.MoeDecode` (benchmark). A first full run found `QuantFormatTests.MatMulIdFewItems`
(12 items, 8 experts, staged formats) on gemm_cm2: the per-item bound had dropped to a quarter of the experts although the staged
skinny form serves nothing above it there; the bound now drops only where the skinny form reaches past it.

### 46.8 Next levers

* **Mixed decode / prefill steps of MoE models** (64 streams, 0.61x of vLLM): the prompt rows cost ~0.094 ms each (128 rows add
  12 ms to a 19 ms step, 70 % of it in the expert GEMMs at 20-30 items per expert, 50-80 TFLOPS). A grouped MXFP4 GEMM with
  more MMA work per decoded weight (wider tiles per expert, or the int8 / FP8 MMA path for MXFP4 weights) is the lever; the
  weight traffic is not (46.4).
* The bf16 attention projections at 64 rows (split-K gemm_cm2 + reductions, ~3.1 ms per step against ~1.5 ms of weight traffic).
* FP8 prefill (0.70-0.75x against the fresh baseline): section 44.3.

### 46.9 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2 VLLM_VENV=<venv with vLLM 0.33.0> HF_DIR=<hf checkpoints>
bench/s42_gap.sh bench/results/s46/gap qwen3-8b vllm ~/models/hf/Qwen3-8B          # and sharplight, per model
ENGINES=vllm bench/s43_serve.sh bench/results/s46/serve-vllm gpt-oss-20b-hf
ENGINES=sharplight SHARPLIGHT_SERVER=<base build>/sharplight bench/s43_serve.sh bench/results/s46/serve-before gpt-oss-20b-hf
ENGINES=sharplight bench/s43_serve.sh bench/results/s46/serve-after gpt-oss-20b-hf
# expert block at serving step sizes (46.3); SHARPLIGHT_MOE_SKINNY=0 for gemm_cm2
SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_COPIES=6 SHARPLIGHT_BENCH_TYPES=mxfp4 SHARPLIGHT_BENCH_ROWS=16 \
  dotnet run --project tests/Sharplight.Vulkan.Tests -c Release -- -method "*MoeDecode"
bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/hf/gpt-oss-20b --slots 64 --ctx 512 --depth 256 \
  --sizes 1,2,4,8,16,32,64 --modes greedy --reps 20
```

## 47. LLM round 9: mixed serving steps examined, MXFP4 decoded per MMA, bias epilogues in the split-K reduction (2026-11-18)

Base: `develop` at `93b68949` (round 8), branch `perf/llm-r9`. GPU 2 (RTX PRO 4500 Blackwell, 32 GB, nothing else on it),
checkpoints on local disk; other agents built and tested on the same host (load average 9-28), which moved host-bound
measurements by up to 10 % (47.6). "Before" is this branch with the switches that restore round 8's behaviour
(`SHARPLIGHT_GEMM_MACROS=SK_PAIR_DECODE=0 SHARPLIGHT_SPLITK_EPILOGUE=0`), run in the same session. vLLM figures are section 46's
(vLLM 0.33.0, same checkpoints, same procedure, measured the same day). Raw output in [`bench/results/s47/`](../bench/results/s47).

### 47.1 Where the mixed steps go

`BatchBench --mixed` (64 decode rows at depth 256 plus one fresh prompt, greedy, minimum GPU ms of 20; profiles over 10 steps,
`SHARPLIGHT_VK_PROFILE=1`, in `profiles/`):

| step | gpt-oss-20b HF | Qwen3-8B-FP8 |
|---|---:|---:|
| 64 decode rows | 19.3 ms | 17.0 ms |
| 64 + 107 prompt rows (the 64-stream serving average of 171 rows) | 29.2 ms | 30.5 ms |
| 64 + 256 prompt rows | 40.1 ms | 40.3 ms |

171-row step, ms per step: gpt-oss: expert GEMMs 19.3 (gate / up [2880 → 5760] 12.6, down 6.7, both gemm_cm2 with 32-row tiles),
bf16 projections + LM head 5.5, attention 2.6, norms / elementwise 2.2 (of which 120 bias `fused[Add]` launches 0.84), split-K
reductions 0.5. Qwen3-8B-FP8: FP8 GEMMs 22.0 (gate / up as two split-K launches, 10.5; down 5.8; q / o 4.1; k / v 1.6), split-K
reductions 1.5 (7 per layer), attention 5.4 (decode rows 4.6), bf16 LM head 1.5.

At 64 streams all 32 experts of every gpt-oss layer are touched, so a step reads at least the 10.1 GB of MXFP4 experts plus 2.4 GB
of bf16 projections and LM head: ~14.6 ms at the card's ~860 GB/s. vLLM's mean inter-token latency at 64 streams is 16.4 ms
(section 46) including its mixed steps, i.e. close to that bound; ours is 28.6 ms, and the decode-only step alone is 19.3.

### 47.2 The grouped MXFP4 GEMM at 20-30 items per expert: what limits it

`BenchTests.MoeDecode` at 171 rows (684 items, uniform routing, 21 per expert; gate / up + down, µs per expert block):

| experiment | gate / up | down | block |
|---|---:|---:|---:|
| gemm_cm2, 32-row tiles (as shipped; 92-95 registers, 31.5 KB shared, 2 workgroups per SM) | 489 | 268 | 766 |
| `GPF=1` (one prefetch register set) | 543 | 279 | 849 |
| `GEMM_EXP=2`: no MMAs (loads, decode, staging, barriers) | 479 | 252 | 759 |
| `GEMM_EXP=1`: no loads after the first step (MMAs, decode, staging) | 186 | 96 | 312 |
| `GEMM_EXP=4`: no activation loads | 467 | 258 | 753 |
| `GPF=3` / `GPF=4` (new: loads issued 2 / 3 steps ahead; 96 / 99 registers) | 475 / 492 | 268 / 297 | 771 / 816 |
| K-loop barriers without memory semantics (diagnostic) | 490 | 266 | 783 |

The weight fetch (281 MB for gate / up in 489 µs: 575 GB/s) bounds the kernel; the MMAs, the decode and the shared-memory staging
take ~40 % of its time and hide under it. Deeper register prefetch and barrier semantics do not change it, so it is not the
number of loads a thread keeps in flight alone. Other forms (all slower, µs per block at 171 rows):

| tried | block |
|---|---:|
| gemm_cm2 32-row layouts [1,4,2,2] / [1,8,2,1] / [1,4,2,4] / [2,4,1,4] / [1,8,2,2] / [2,2,1,4] (`SHARPLIGHT_GEMM2_CFG32`) | 820 / 766 / 941 / 894 / 880 / 826 |
| 16-row skinny expert tiles at 21 items per expert (`SHARPLIGHT_MOE_SKINNY=24`: two tiles per expert) | 1004 |
| new: direct skinny GEMM with 2 token fragments (32-row expert tiles, each decoded weight serving both), subgroups along N (`SHARPLIGHT_MOE_WIDE=fm,fn,sg,ks,n`): 2,2,8,1 / 2,2,4,4 / 2,4,4,1 / 2,2,4,2 / 3,2,8,2 | 879 / 864 / 827 / 783 / 1015 |
| ... the same without the MXFP4 decode / without activation loads (diagnostics, 2,4,4,1) | 781 / 877 |

The wide direct tiles run at 168-255 registers. The opt-in switch (and the kernel's `SK_KS` / several-fragment direct path) is kept
for further work; the default is unchanged. The real lever stays a fetch at the bandwidth limit for 16-48 rows per expert, which
neither the staged gemm_cm2 (single-buffered 64-value K steps, 2 workgroups per SM) nor the register-direct skinny form reach on
this driver.

### 47.3 MXFP4 weights decoded per MMA in the direct skinny GEMM (kept)

The direct path decoded a lane's 16 weights of a 64-value block per row before the block's four MMAs (32 registers per weight
fragment). It now decodes the 4 values each MMA needs right before it (`SK_PAIR_DECODE`, MXFP4R only), so no decoded block is held.
`MoeDecode` (6 weight copies) at 16 / 64 sequences 516 / 603 → 496 / 567 µs per block; BatchBench decode steps of gpt-oss-20b HF:

| sequences | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| before, GPU ms | 7.89 | 9.13 | 10.29 | 13.37 | 19.58 |
| after | 7.76 | 8.94 | 10.04 | 12.94 | 18.99 |

The decoded values are identical (same arithmetic), so outputs do not change. FP8 weights lose 2-6 % with the per-MMA decode
(Qwen3-8B-FP8 decode steps at 2-16 sequences 11.9-13.7 → 12.7-14.1 ms) and keep the block decode.

### 47.4 Bias and residual epilogues in the split-K reduction (kept)

A weight matmul's bias add (`BiasAdd` units: q / k / v / o biases) and the folded elementwise groups were applied in the store
only by the split-f16 GEMM (f32 weights); for f16 / bf16 / quantized weights on the f16 tensor-core GEMM they ran as their own
launch. Where that GEMM splits K (9-~256 rows for projections of 512-4096 outputs), `splitk_reduce` now applies the group (its
existing `MM_EPI` path); other row counts run the group's launch as before. Each output is the same expression of the same sums,
so results are bit-identical. gpt-oss-20b (biased q / k / v / o and router): 96 fewer launches per step; 64-row decode step 18.72
→ 18.36 ms, 171-row mixed step 29.32 → 29.22 ms (BatchBench, two alternating runs each). `SHARPLIGHT_SPLITK_EPILOGUE=0` restores
the separate launch. Test: `FusionTests.MatMulBias` (new bf16 / f16 / MXFP4 cases at 20-300 rows).

### 47.5 FP8 GEMM with 64-row tiles at 129-192 and 257-320 rows (opt-in, not kept as default)

The fp8 GEMM takes 64-row tiles up to 64 rows and 128-row tiles above. For 150-320 rows 128-row tiles pad up to 41 % of the rows,
and the small grids split K (two launches plus a reduction per matmul, and the dual gate / up GEMM is skipped where the separate
GEMMs split). Qwen3-8B-FP8, BatchBench mixed steps (64 decode rows + prompt), GPU ms, 128-row tiles → 64-row tiles:

| rows | 104 | 128 | 150 | 171 | 214 | 256 | 300 | 320 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 128-row tiles | 20.5 | 21.2 | 31.1 | 31.7 | 33.4 | 35.9 | 42.0 | 41.3 |
| 64-row tiles | 22.3 | 23.1 | 27.0 | 27.7 | 32.3 | 35.1 | 36.8 | 38.9 |

The rule "64-row tiles where they pad fewer rows" (rows mod 128 in 1-64, up to 384 rows) gains 12-13 % on those steps, but serving
did not follow (Qwen3-8B-FP8, 5 rounds each, same build with the rule on / off: 16 streams 899 → 936, 64 streams 1996 → 2014
tok/s, within the session spread; round-8 behaviour measured 917 / 2030 in the same session), and it changes the rounding of
129-384-token prompts (greedy output of a 182-token prompt diverged after 50 tokens). It is opt-in (`SHARPLIGHT_FP8_SHORT_TILES=1`).
Why the per-step gain does not reach serving is open (the serving steps of 64 streams average 171 rows, 422 per round).

Bug found on the way and fixed (generic): the split-K variants of the int8 / fp8 GEMMs were found by probing row counts in steps of
128 (`NextSplitProbe`), which skips row counts whose tile height differs from that of the probed ones; with the 64-row tiles above
128 rows, prompts of 129-192 rows ran no GEMM launch at all (garbage output) while tests that compile exactly their row count passed.
The quantized GEMMs now probe every 64 rows. Test: `Fp8GemmTests.ShortTilesBelowMaxTokens` (rows 150-300 in a graph compiled for
2048; fails without the fix with the opt-in tiles) and `ShortTilesBelowMaxTokensOptIn` (child run with the switch).

### 47.6 Per-step host overhead at 64 streams

The engine's step metric covers 99.0-99.8 % of a 64-stream round's wall time (`metrics_delta`, 422 steps per round), and
BatchBench's wall time per step exceeds the GPU time by 0.1 ms at 64 decode rows (19.08 against 18.99 ms) and 0.13 ms at 171 rows.
Host work is not part of the gap. Host load from other processes, however, moved whole serving runs: the first before / after
pair ran while the machine's load average was 28 and its "before" rounds took 6-10 % longer step times (gpt-oss 1 / 4 streams 161 /
416 tok/s against 178 / 456 in the repeat), so the table below uses the repeat, run in reverse order (after, then before).

### 47.7 Serving against vLLM (procedure of section 43, median of 3 rounds; Qwen3-8B-FP8 "before" at 16 / 64 streams: 5 rounds)

| model | streams | before | after | after / before | vLLM (§46) | after / vLLM | TTFT p50 after / vLLM (ms) | per-stream decode after / vLLM |
|---|---:|---:|---:|---:|---:|---:|---|---|
| gpt-oss-20b HF | 1 | 178 | 178 | 1.00x | 169 | 1.05x | 32 / 31 | 186 / 175 |
| | 4 | 456 | 468 | 1.03x | 501 | 0.93x | 61 / 77 | 123 / 134 |
| | 16 | 1171 | 1200 | 1.02x | 1428 | 0.84x | 109 / 175 | 81 / 102 |
| | 64 | 1939 | 1979 | 1.02x | 3221 | 0.61x | 184 / 381 | 35 / 61 |
| Qwen3-8B-FP8 HF | 1 | 88 | 88 | 1.00x | 86 | 1.02x | 34 / 32 | 89 / 87 |
| | 4 | 303 | 304 | 1.00x | 326 | 0.93x | 71 / 72 | 79 / 85 |
| | 16 | 917 | 919 | 1.00x | 1123 | 0.82x | 103 / 157 | 61 / 78 |
| | 64 | 2030 | 1958 | 0.96x | 2717 | 0.72x | 196 / 391 | 35 / 50 |

Qwen3-8B-FP8 "after" is the final default build (`serve-after5`), run while the host's load average was 45 (`before3`: 9-20); its
kept changes do not touch FP8 weights (the per-MMA decode is MXFP4-only, the model has no biases), so the 64-stream difference is
host load, not the change (`serve-after2/3` with the then-default FP8 tiles measured 2022 / 2014). The gains are small (gpt-oss
+2-3 % from 4 streams) and the absolute figures equal section 46's after-columns within their spread. The gaps to vLLM are
unchanged: 0.84x / 0.61x (gpt-oss) and 0.82-0.83x / 0.72-0.75x (Qwen3-8B-FP8) at 16 / 64 streams. Dense prefill (bf16 0.87x,
FP8 0.73x) was not changed this round.

### 47.8 Greedy output and tests

Greedy output (200 tokens, prompts of ~100, ~150 and 182 tokens): gpt-oss-20b HF identical with the round's changes on and off
(per-MMA decode and reduction epilogue are bit-identical); Qwen3-8B-FP8 identical with the default build (the 64-row FP8 tiles,
now opt-in, changed the 182-token prompt's output after 50 tokens). New tests: `FusionTests.MatMulBias` (4 cases),
`Fp8GemmTests.ShortTilesBelowMaxTokens` (4) and `ShortTilesBelowMaxTokensOptIn`. After merging `develop` (at `92c451a3`), on GPU 2:

| suite | total | failed | skipped |
|---|---:|---:|---:|
| Sharplight.Vulkan.Tests (full) | 1610 | 0 | 26 |
| Sharplight.Families.Tests (full, batched greedy parity included) | 1005 | 0 | 364 |

### 47.9 Next levers

* **The grouped MXFP4 GEMM at 16-48 rows per expert** (47.2): bound by a weight fetch at ~575 GB/s; 64-stream steps need it at
  ~800. Candidates: weights staged raw (4.25 bits) in shared memory with several stages and decoded at the fragment load, so the
  stages fit 48 KB; a register budget that allows 3 workgroups per SM; the expert block's gate / up and down fused per expert so
  each expert tile is scheduled once.
* **Why per-step FP8 gains do not reach serving** (47.5): profile a serving run per step size.
* **FP8 prefill (0.73x)** and the attention of the decode rows in mixed steps (Qwen3-8B-FP8: 4.6 ms of a 30.5 ms step).

### 47.10 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2 HF_DIR=<hf checkpoints> VLLM_VENV=<venv with vLLM 0.33.0>
# mixed-step profiles (47.1)
SHARPLIGHT_VK_PROFILE=1 bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/hf/gpt-oss-20b --slots 68 --ctx 512 --depth 256 \
  --mixed 64+0,64+107x1,64+128x2 --modes greedy --reps 20 --profile 1,2
# expert block diagnostics (47.2)
SHARPLIGHT_GEMM_MACROS=GEMM_EXP=2 SHARPLIGHT_BENCH=1 SHARPLIGHT_BENCH_REPORT=1 SHARPLIGHT_BENCH_TYPES=mxfp4 SHARPLIGHT_BENCH_ROWS=171 \
  dotnet run --project tests/Sharplight.Vulkan.Tests -c Release -- -method "*MoeDecode"
SHARPLIGHT_MOE_WIDE=2,2,4,2,48 ...   # wide direct expert tiles
# serving (47.7): before = round 8's behaviour
SHARPLIGHT_GEMM_MACROS=SK_PAIR_DECODE=0 SHARPLIGHT_SPLITK_EPILOGUE=0 ENGINES=sharplight \
  bench/s43_serve.sh bench/results/s47/serve-before2 gpt-oss-20b-hf qwen3-8b-fp8-hf
ENGINES=sharplight bench/s43_serve.sh bench/results/s47/serve-after2 gpt-oss-20b-hf qwen3-8b-fp8-hf
# FP8 64-row tiles (47.5)
SHARPLIGHT_FP8_SHORT_TILES=1 bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/hf/Qwen3-8B-FP8 --slots 68 --ctx 512 \
  --depth 256 --mixed 64+0,64+86x1,64+107x1,64+236x1 --modes greedy --reps 15

## 48. Prefix caching by default: hashed KV blocks, cache namespaces, recurrent-state snapshots, against vLLM's APC (2026-11-18)

Branch `serving/prefix-cache` (design: [kv-cache.md](kv-cache.md) section 2.6). Paged KV with prefix caching is now the default of
multi-slot decoders; prompts share cached blocks by content hash within their cache namespace (partition and salt); models with
recurrent state reuse prefixes through state snapshots. Raw results in [`bench/results/s48/`](../bench/results/s48).

### 48.1 Paged by default: no regression (A/B, contiguous vs paged)

Llama-3.2-1B Q8_0, one RTX PRO 4500, the same build with `--kv-tokens 0` (contiguous) and the default (paged, `slots × ctx` pool).
Decode steps from `BatchBench` (greedy on the device, median / min of 40 steps, GPU ms, three alternating runs each); prefill and
end-to-end decode from `sharplight serve --slots 8 --ctx 4096 --no-prefix-cache` with `bench/loadgen.py -c 1` (8 requests of 2001
prompt tokens and 128 output tokens, three rounds):

| measurement | contiguous | paged (default) |
|---|---|---|
| decode step, 1 sequence at depth 2048 (median, min) | 2.07-2.23, 1.98-1.99 ms | 2.06-2.11, 2.00-2.01 ms |
| decode step, 2 sequences at depth 2048 | 2.34-2.73, 2.25-2.26 ms | 2.38-2.41, 2.29 ms |
| decode step, 4 / 16 sequences at depth 1024 (min) | 2.47-2.48 / 3.18 ms | 2.51-2.52 / 3.20-3.21 ms |
| TTFT p50, 2001-token prompt (serve, c = 1) | 47.7-51.6 ms | 47.3-50.2 ms |
| ITL p50 (serve, c = 1) | 2.19-2.40 ms | 2.21-2.29 ms |

Single-stream decode and prefill are unchanged within the noise; batched decode steps are 1-2 % slower (one block-table load per
16-row tile, as measured in 2.5). Greedy parity: the Families suite (batched greedy parity of every family, now on paged decoders
by default) passes in full, and `PrefixCacheParityTests` checks greedy output with the cache on against the cache off (48.4).

### 48.2 Against vLLM 0.33.0 APC: Qwen3-8B-FP8, identical weights and KV capacity

Procedure (`bench/s48_prefix.sh`, `bench/prefix_bench.py`): one server per engine and cache setting, 64 sequences of up to 8192
tokens, the same KV capacity of 73,728 tokens on both (Sharplight `--kv-tokens 73728`; vLLM `--kv-cache-memory-bytes` = 73,728 ×
147,456 bytes, which vLLM reports as "GPU KV cache size: 73,728 tokens"), vLLM `--enable-prefix-caching` / `--no-enable-prefix-caching`,
Sharplight default / `--no-prefix-cache`, both with the prompt-tokens details in the usage. Streaming chat completions, temperature 0,
128 output tokens (`ignore_eos`), thinking off, closed loop; a short unrelated warm-up request, then two rounds per server (round 2
starts with whatever round 1 left in the cache, on both engines):

- **shared**: a ~1530-token system prompt shared by every request plus a unique 30-word question; 16 workers × 4 and 64 workers × 3 requests;
- **multiturn**: 16 conversations of 6 turns, every turn resending the whole history (earlier questions and the server's replies) plus a new 90-word message;
- **rag**: 4 shared 1000-token documents; every request quotes two of them in random order and asks a question (16 workers × 4).

The hit rate is cached prompt tokens / prompt tokens from the usage records. GPU: RTX PRO 4500 (GPU 1) shared with another agent's
Vulkan test process during the runs (about 300 MiB; GPU 2 was taken by the perf and verification agents), so single figures can be off
by tens of percent; round-to-round swings of vLLM's cache-on TTFT (round 2 slower than round 1) are of that kind.

| workload | chats | engine | cache | TTFT p50 / p99 ms, round 1 | round 2 | output tok/s, round 1 / 2 | hit rate (round 2) |
|---|---:|---|---|---|---|---|---:|
| shared | 16 | Sharplight | on | 134 / 491 | 122 / 133 | 646 / 889 | 97.2 % |
| shared | 16 | vLLM | on | 158 / 412 | 223 / 336 | 730 / 879 | 97.2 % |
| shared | 16 | Sharplight | off | 521 / 2468 | 593 / 2448 | 418 / 410 | 0.0 % |
| shared | 16 | vLLM | off | 442 / 1664 | 451 / 1685 | 554 / 550 | 0.0 % |
| shared | 64 | Sharplight | on | 418 / 559 | 312 / 360 | 1747 / 2224 | 97.9 % |
| shared | 64 | vLLM | on | 227 / 497 | 650 / 1090 | 2654 / 2205 | 97.9 % |
| shared | 64 | Sharplight | off | 4519 / 17892 | 4583 / 17799 | 465 / 464 | 0.0 % |
| shared | 64 | vLLM | off | 3064 / 10361 | 3085 / 10355 | 665 / 663 | 0.0 % |
| multiturn | 16 | Sharplight | on | 288 / 440 | 243 / 395 | 767 / 846 | 67.3 % |
| multiturn | 16 | vLLM | on | 242 / 584 | 324 / 533 | 757 / 796 | 66.8 % |
| multiturn | 16 | Sharplight | off | 458 / 810 | 420 / 799 | 669 / 664 | 0.0 % |
| multiturn | 16 | vLLM | off | 308 / 719 | 312 / 691 | 853 / 854 | 0.0 % |
| rag | 16 | Sharplight | on | 93 / 1520 | 167 / 616 | 689 / 746 | 89.4 % |
| rag | 16 | vLLM | on | 162 / 2257 | 579 / 1521 | 716 / 546 | 85.5 % |
| rag | 16 | Sharplight | off | 650 / 3343 | 648 / 3334 | 348 / 349 | 0.0 % |
| rag | 16 | vLLM | off | 593 / 2406 | 594 / 2326 | 453 / 457 | 0.0 % |

- The cache hits the same tokens on both engines: equal hit rates for the shared prompt (97.2-97.9 %) and the conversations
  (67 %; the rest is each turn's new message and the reply, re-rendered by the template), a higher one for RAG in round 2 (89 vs 86 %).
  Without the cache Sharplight is slower than vLLM (FP8 prefill at 0.70-0.75x, section 46), so the cache matters more here: TTFT p50
  with the cache is 0.07-0.63x of without (vLLM 0.07-1.04x).
- With the cache on, Sharplight's TTFT p50 is below vLLM's in 6 of the 8 cache-on rows (all 4 of round 2), and throughput is level
  (1.01-1.37x in round 2); round 1 at 64 chats (418 vs 227 ms, 0.66x tok/s) and multi-turn (288 vs 242 ms) are the exceptions: Sharplight records kernels for new step shapes
  in its first round (section 46.5), vLLM captured its CUDA graphs at startup.
- Concurrent arrivals: in the shared workload all 16 or 64 chats start at once; one prompt computes the system prompt and the others wait for
  its blocks (48.4) instead of all computing it, which is what keeps the 64-chat TTFT at 0.3 s instead of the 4.6 s without the cache.

### 48.3 Hybrid model against vLLM: Qwen3.5-0.8B (gated delta net + attention, HF bf16)

Same procedure on an otherwise idle GPU (GPU 0; another process appeared in 2 of 49 samples). vLLM 0.33.0 caches the gated-delta-net
state in its "align" mode, which it enables with prefix caching for this model: it sets the attention block size to 544-576 tokens so
that a Mamba state page fits one block, and keeps the state at block boundaries that end a scheduler step. It was given 4 GiB
(`--kv-cache-memory-bytes`, attention KV and Mamba states share it); Sharplight used its defaults: 73,728 KV tokens (0.9 GB) and 26
state snapshots of 19 MB (0.5 GB; `DefaultStateSnapshots`: half the KV pool's memory or 512 MiB).

| workload | chats | engine | cache | TTFT p50 / p99 ms, round 1 | round 2 | output tok/s, round 1 / 2 | hit rate, round 1 / 2 |
|---|---:|---|---|---|---|---|---|
| shared | 16 | Sharplight | on | 237 / 547 | 125 / 192 | 2222 / 2991 | 95.1 / 95.1 % |
| shared | 16 | vLLM | on | 112 / 174 | 204 / 659 | 3396 / 1537 | 72.8 / 73.9 % |
| shared | 16 | Sharplight | off | 117 / 564 | 112 / 477 | 2006 / 2066 | 0.0 / 0.0 % |
| shared | 16 | vLLM | off | 537 / 848 | 525 / 818 | 1313 / 1279 | 0.0 / 0.0 % |
| shared | 64 | Sharplight | on | 499 / 627 | 588 / 714 | 5155 / 4399 | 95.6 / 95.6 % |
| shared | 64 | vLLM | on | 294 / 506 | 337 / 890 | 6573 / 5197 | 73.9 / 73.9 % |
| shared | 64 | Sharplight | off | 900 / 3865 | 867 / 3612 | 2207 / 2274 | 0.0 / 0.0 % |
| shared | 64 | vLLM | off | 1630 / 3284 | 1592 / 3167 | 1983 / 2023 | 0.0 / 0.0 % |
| multiturn | 16 | Sharplight | on | 178 / 297 | 161 / 289 | 2857 / 2470 | 54.4 / 60.0 % |
| multiturn | 16 | vLLM | on | 110 / 250 | 99 / 220 | 3178 / 3365 | 28.3 / 28.1 % |
| multiturn | 16 | Sharplight | off | 101 / 185 | 95 / 178 | 2996 / 2999 | 0.0 / 0.0 % |
| multiturn | 16 | vLLM | off | 218 / 635 | 321 / 705 | 1956 / 1686 | 0.0 / 0.0 % |
| rag | 16 | Sharplight | on | 254 / 429 | 279 / 339 | 2248 / 2475 | 68.7 / 69.5 % |
| rag | 16 | vLLM | on | 127 / 452 | 119 / 137 | 2592 / 3365 | 69.7 / 83.6 % |
| rag | 16 | Sharplight | off | 170 / 692 | 131 / 681 | 1707 / 1708 | 0.0 / 0.0 % |
| rag | 16 | vLLM | off | 484 / 994 | 590 / 970 | 1229 / 1242 | 0.0 / 0.0 % |

- Hit rates: Sharplight reuses more of the shared system prompt (95 vs 74 %: snapshots at the 16-token block where the prompts diverge,
  vLLM at the last 576-token boundary before it) and of the conversations (54-60 vs 28 %: the snapshot one block before each prompt's end),
  and as much of the RAG prompts in round 1 (69 vs 70 %); in round 2 vLLM's larger state budget (4 GiB against 1.4 GB) keeps more document
  pairs (84 vs 70 %).
- TTFT: for this small model a prefill pass costs little, and Sharplight pays for its snapshots in passes: a prompt's prefill stops at each
  snapshot boundary (merged when closer than 256 tokens), and prompts arriving together wait for the blocks and snapshots another prompt is
  computing. With the cache on its TTFT p50 is above its own cache-off TTFT on the conversation and RAG loads (161 vs 95, 279 vs 131 ms in
  round 2) and above vLLM's in 7 of 8 cache-on rows, while it computes 2-9x fewer prompt rows (diagnostic runs of the same loads) and its
  cache-off TTFT is 1.8-4.7x below vLLM's cache-off. Next lever: for models whose prefill is cheaper than a pass, recompute instead of waiting for another prompt's blocks, and
  take periodic snapshots only in chunks that end there anyway (no extra pass), as vLLM does.
- The 8B dense rows of 48.2 show the other side: where prefill is expensive, waiting for blocks and attaching them is what keeps TTFT low.

### 48.4 How concurrent prompts and preemption use the cache, and tests

- Prompts arriving together with the same prefix: at admission each prompt registers the blocks it will compute; a later prompt's
  prefill chunk stops before the first block another prompt is computing and attaches it after that prompt's step publishes it (attach
  happens before every prefill chunk, so hits also apply mid-prompt). Waits follow admission order and never cross namespaces.
  `PrefixCacheTests.ConcurrentPrompts_WaitForBlocksAnotherPromptComputes_InsteadOfComputingThemTwice`: 8 prompts with a 128-token shared
  prefix arriving at once compute it once (7 of 8 report ≥ 128 cached tokens).
- Preemption: a preempted sequence's blocks stay cached and are attached again when it resumes (`PreemptedSequences_TakeTheirCachedBlocksBack`).
- Greedy parity: `PrefixCacheParityTests` (Families) decodes prompts sharing a 40-token prefix with the cache off and on (one after another
  and two at once) and requires token-identical output and cache hits, on the CPU and Vulkan, for attention-only (qwen1, granite), gated
  delta net (qwen3.8-flash-next), Mamba, Mamba-2 hybrid (falcon-h1), short-convolution (lfm3) and RWKV-7 tiny models, and for
  Qwen3.5-0.8B and Qwen3-0.6B on Vulkan: 16 / 16 pass.

Suites after merging `develop` (at `84b42062`): Serving 352 (0 failed, 13 skipped; `OpenAiApiTests.ClientDisconnect_CancelsGeneration`
failed once under machine load and passed 3 / 3 alone, a one-slot decoder without paging), Families full 1021 (0 failed, 364 skipped, as
in 46.7), Extensions.AI 38 (0 failed, 3 skipped), Vulkan `PagedAttentionTests`, `StatefulTests`, `DeviceLifetimeTests`,
`RecurrentRollback*` 70 (0 failed, 1 skipped), Vision `MultiSequenceTests` 27 (0 failed, 25 skipped: models not on this machine).

### 48.5 Reproduce

```bash
export CUDA_VISIBLE_DEVICES=<i> SHARPLIGHT_VK_DEVICES=<i> VLLM_VENV=<venv with vLLM 0.33.0>
bench/s48_prefix.sh bench/results/s48/qwen3-8b-fp8 ~/models/hf/Qwen3-8B-FP8          # sharplight, sharplight-nocache, vllm, vllm-nocache
VLLM_KV_BYTES=4294967296 bench/s48_prefix.sh bench/results/s48/qwen3.5-0.8b ~/models/hf/Qwen3.5-0.8B
bench/s48_summarize.py bench/results/s48/qwen3-8b-fp8 2
# paged vs contiguous (48.1)
bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/gguf/Llama-3.2-1B-Instruct-Q8_0.gguf --slots 2 --ctx 4096 --depth 2048 \
  --batch 4096 --sizes 1,2 --modes greedy --reps 40 [--kv-tokens 0]
```

## 49. LLM round 10: serving steps as large as vLLM's, fused gate / up halves and expert epilogues in the grouped expert GEMMs (2026-11-18)

Base: `develop` at `91942038` (round 9), branch `perf/llm-r10`. GPU 2 (RTX PRO 4500 Blackwell, 32 GB, nothing else on it), vLLM
0.33.0 in a virtualenv on local disk, checkpoints on local disk. "Before" is this branch with the switches that restore round 9's
behaviour (`SHARPLIGHT_MOE_DUAL=0 SHARPLIGHT_MOE_BIAS_EPILOGUE=0`, and `--batch 512` for serving), run in the same session. Raw
output in [`bench/results/s49/`](../bench/results/s49).

### 49.1 What a serving round is made of (and why the FP8 64-row tiles of 47.5 did not reach it)

`bench/loadgen.py` now also records the cumulative buckets of the server's `sharplight_step_rows` histogram. One 64-stream round
(192 requests of ~250 prompt and 128 output tokens, closed loop) on round 9's build:

| step rows | ≤ 64 | 65-128 | 129-256 | 257-512 | 513-2048 | steps |
|---|---:|---:|---:|---:|---:|---:|
| gpt-oss-20b, 512-row budget (round 9) | 315 | 0 | 2 | 105 | 0 | 422 |
| gpt-oss-20b, automatic budget (49.2: 2048 rows) | 367 | 1 | 1 | 2 | 24 | 395 |
| gpt-oss-20b at 16 streams, 512 / 2048 rows | 487 / 505 | 0 / 0 | 2 / 1 | 35 / 3 | 0 / 8 | 524 / 517 |

The steps are bimodal: decode-only steps of at most 64 rows (75 %) and prompt waves at the step budget. The budget was 512 rows,
not the 2048 of the automatic choice: `LlmModel.BatchTokens` caps the automatic rows at the context length, and the serving
benchmark gives each of the 64 slots a 512-token context. A step, however, carries the rows of all slots. vLLM schedules up to
8192 rows per step (`max_num_batched_tokens=8192` in its log) with the same `--max-model-len 512`.

This also answers 47.5: the 64-row FP8 tiles gain at 129-192 and 257-320 rows, and a round has almost no steps in those ranges
(2 of 422 between 129 and 256 rows; the prompt steps are full 512-row steps, where 64-row tiles pad as many rows as 128-row ones).
The per-step gain was real but measured at step sizes the scheduler does not produce. The switch stays opt-in.

### 49.2 Rows per serving step bounded by all slots' contexts (kept)

`LlmLoadOptions.Sequences` (new, default 1): the number of sequences decoded together. The automatic rows per pass are now bounded
by `context × Sequences` instead of the context (still at most 2048, at least 512, and halved while the footprint does not fit);
the hosting layer (`Sharplight.Hosting`: `sharplight serve` and other hosts built on it) sets it to the model's slots. With 64
slots of 512 tokens the server now runs 2048-row steps: a 64-stream round takes 395 steps instead of 422 (24 prompt steps of
1025-2048 rows instead of 105 of 257-512). The per-row cost of a 2048-row step is lower (one weight pass of the experts or dense
layers per 2048 prompt rows instead of four); TTFT at 64 streams moves from ~180 ms to ~340-380 ms (vLLM: 381-461 ms) and the mean
inter-token latency falls.

### 49.3 gpt-oss's fused gate / up projection as a dual pair in the grouped expert GEMMs (kept)

gpt-oss stores gate and up as one expert matrix ([E, 2I, H], gate rows first after import); the block computed the 2I-wide product
(plus the experts' gate / up bias rows) into an f32 [items, 5760] tensor, which the clamped SwiGLU then read as two strided halves.
At 2048 prompt rows that is 189 MB written and read again per layer, and 0.3 ms of elementwise time per layer.

* `FindExpertHalves` (lowering): an expert product whose output (after a folded bias Add) is only read, as its two halves along the
  last axis, by one elementwise group marks that group (`LoweringUnit.Halves`).
* The grouped GEMMs (gemm_cm2 with 32- to 128-row tiles and the expert-grouped skinny GEMM) then run their dual form with N = I and
  w2 = the same expert's rows I .. 2I - 1 (a pointer I rows further: every format's rows, MXFP4R's 16-row groups included, stay
  whole for I a multiple of 16); their store adds both halves' bias rows and applies the group (`MOE_EPI`, generated from the
  group's nodes). The [items, 2I] product is never stored; batches on the per-item matvecs run the product, the Add and the group
  as before.
* The same dual form now also serves separate gate / up expert pairs (Qwen3-MoE style, `LowerDualMatMulId`), which ran two grouped
  GEMMs and the group above the int-dot matvec range. Both use `DUAL` with `MOE` in gemm_cm2 (16-row fragments of w and w2
  alternate in the B tile; raw MXFP4 units included) and in gemm_sk.
* Each product is accumulated as in the separate GEMM (same tiles, K steps and MMAs) and the epilogue is the elementwise kernel's
  expression, so outputs are those of the separate launches (gpt-oss greedy outputs byte-identical with the switches on and off).

`SHARPLIGHT_MOE_DUAL=0` restores the separate launches.

### 49.4 Expert epilogues in the grouped GEMMs' store (kept)

* A bias Add of the routed experts' rows that is an expert product's only reader (`FoldExpertBiases`) is applied in the grouped
  GEMMs' store (`MOE_BIAS`): the product's f32 tensor and a launch go away for grouped batches.
* The routed-expert sum's group (down bias and router weight, `LoweringUnit.ExpertSum`) was fused only into the int-dot slot-sum
  matvec of decode-sized batches; the grouped GEMMs now apply it in their store too (`MOE_EPI1`, the slot-sum matvec's generated
  epilogue), so the ReduceSum reads the weighted rows straight from the GEMM.

`SHARPLIGHT_MOE_BIAS_EPILOGUE=0` restores the separate launches.

### 49.5 Step times (BatchBench, gpt-oss-20b HF, GPU ms, minimum of 12)

After merging `develop` (49.9), two alternating runs each, GPU ms per step (minimum of 12, the two runs' values); "before" is the same
build with `SHARPLIGHT_MOE_DUAL=0 SHARPLIGHT_MOE_BIAS_EPILOGUE=0`:

| step | before | after | after / before | dispatches before / after |
|---|---:|---:|---:|---:|
| 16 decode rows | 10.06 / 10.09 | 9.97 / 9.97 | 0.99x | 726 / 654 |
| 64 decode rows | 17.94 / 18.02 | 17.12 / 17.14 | 0.95x | 823 / 751 |
| 64 decode + 107 prompt rows | 29.75 / 29.92 | 28.78 / 28.90 | 0.97x | 750 / 678 |
| 64 decode + 448 prompt rows (a 512-row step) | 55.03 / 55.23 | 51.50 / 51.62 | 0.94x | 798 / 726 |
| 2048 prompt rows | 192.81 / 193.52 | 169.92 / 170.67 | 0.88x | 798 / 726 |

Profile of the 2048-row step before / after (`SHARPLIGHT_VK_PROFILE=1`, per layer): the gate / up GEMM 2.48 + 0.17 ms (half
tiles) both ways (same weight traffic, now storing 2880 instead of 5760 columns), the clamped SwiGLU 0.30 ms and the gate / up bias
Add 0.04 ms gone, the down group (bias, router weight) 0.24 ms gone. At 64 decode rows the dual skinny gate / up launch takes 188 µs
against 211 µs for the single 5760-wide one.

### 49.6 Serving against vLLM (procedure of section 43, median of 3 rounds, same session)

| model | streams | before | after | after / before | vLLM | before / vLLM | after / vLLM | TTFT p50 after / vLLM (ms) | ITL mean after / vLLM (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| gpt-oss-20b HF | 16 | 1191 | 1287 | 1.08x | 1432 | 0.83x | **0.90x** | 198 / 174 | 10.7 / 9.9 |
| | 64 | 2001 | 2244 | 1.12x | 3222 | 0.62x | **0.70x** | 339 / 461 | 24.9 / 16.3 |
| Qwen3-8B-FP8 HF | 16 | 935 | 937 | 1.00x | 1129 | 0.83x | 0.83x | 219 / 159 | 15.1 / 12.9 |
| | 64 | 2032 | 2167 | 1.07x | 2732 | 0.74x | **0.79x** | 380 / 389 | 25.4 / 20.0 |

vLLM was rerun in the same session (gpt-oss 1432 / 3222, Qwen3-8B-FP8 1129 / 2732 tok/s: within 1 % of section 46). The serving
gain comes from both parts: the 2048-row steps (gpt-oss +3-4 % at 16 / 64 streams on round 9's kernels, measured first with the budget
alone, `budget/`) and the expert epilogues and halves (49.3-49.4, gpt-oss only). Qwen3-8B-FP8 (dense) gains only from the step budget,
and at 16 streams not at all: a 16-stream wave of 4000 prompt tokens costs about the same in eight 512-row steps as in two 2048-row
steps on the dense FP8 GEMMs. TTFT at 16 streams doubles (a prompt now waits for a 2048-row step instead of a 512-row one) and is
then 14 % / 38 % above vLLM's for gpt-oss / Qwen3-8B-FP8; at 64 streams it is at or below vLLM's. These figures were measured before
merging `develop` (paged KV by default since section 48). Rerun after the merge (`serve-after-merge/`, median of 3): gpt-oss 1279 /
2235, Qwen3-8B-FP8 934 / 2146 tok/s at 16 / 64 streams, within 1 % of the table.

### 49.7 Tried and not kept

| tried | result |
|---|---|
| Expert tiles along the grid's x (`SHARPLIGHT_MOE_TILE_MAJOR`, opt-in switch kept): the tiles of one expert run side by side, so a second tile of an expert reads its weight chunk from L2 | gpt-oss, 64 + 107 rows: gemm_cm2 29.41 → 29.00 ms, skinny with 2 tiles per expert (`SHARPLIGHT_MOE_SKINNY=24`) 34.04 → 34.51 ms; decode-only 18.37 → 18.14-18.18 ms: within the noise. At 20-30 items per expert gemm_cm2 has one tile per expert, so there is no second read to save |
| 16-row skinny expert tiles up to 24 items per expert (two tiles per expert) | 64 + 107 rows 29.41 → 34.04 ms (as in 47.2) |

### 49.8 The other candidates of the brief

* **Overlapping the down projection with the next layer's gather and routing** (1b of the brief): not built. The next layer's
  routing needs its router logits, which need the norm of the residual that the down projection's slot sum produces; the gather needs
  the routing. Within a decode step there is no independent work to overlap with the down projection except across layers of
  different sequences, which the single step graph does not have. A second compute queue would only help with independent launches
  (the device has 8 compute-only queues); none exist on this path.
* **Weight-stationary expert tiles** (1c): at 20-30 items per expert, gemm_cm2 already runs one 32-row tile per expert, so every
  expert's weights are read once per step; the tile-major order above covers the case with several tiles per expert. The limit
  of 47.2 (the fetch at ~575 GB/s) is unchanged; this round removed work around the GEMMs instead (49.3, 49.4).
* **FP8 64-row tiles in serving** (2): answered in 49.1 (the scheduler does not produce those step sizes); the opt-in stays.
* **FP8 prefill within 48 KB** (3): not attempted this round. The serving analysis moved the priority: at 64 streams FP8 serving
  spends ~45 % of its time in prompt steps, which the 2048-row budget made cheaper per row (2032 → 2167 tok/s). The FP8 GEMM limits
  of 42.3 / 44.3 stand (one 36 KB stage per workgroup, 222 registers, per-128-block scaling needs a second accumulator set).
* **VK_NV_cooperative_matrix2 / VK_NV_cooperative_vector** (4): queried on this driver (595.112, `sharplight --devices -v`).
  Cooperative vectors offer f16 × f16 → **f16** and int8 → int32 matrix-vector products only: no FP8, FP4 or bf16 matrices and no
  f32 accumulation, so neither the MXFP4 experts nor the FP8 / bf16 weights can use them without a precision change. Cooperative
  matrix 2 offers workgroup-scope 32 x 32 x 32 FP8 shapes (256 invocations) with f32 accumulation and tensor-addressed loads; the
  device stages the tiles within the same 48 KB of shared memory, and section 24.4 measured an FP8 GEMM on it 1.6x slower than
  `gemm_fp8` (gemm_nv2 for f16 / q8_0 is opt-in for the same reason). No larger effective tile is available through either
  extension on this driver, so neither was used.

### 49.9 Tests

New tests: `MoeTests.ExpertBiasEpilogue` (6: a gpt-oss-style fused gate / up product with bias rows read as two halves, MXFP4 /
bf16 / Q4_K, on the skinny GEMM and the 32- to 128-row gemm_cm2 tiles), `MoeTests.RoutedExpertBlock` (4 new cases at 90-520 tokens:
the dual gate / up pair with biases and the expert-sum epilogue on the grouped GEMMs), `F16RangeTests.Dual_expert_gemm_reads_rows_beyond_f16_range`
(4: row-ranged rows unscaled before the fused epilogue). The existing `GatedBlockSortedActivations`, `FusedGateUpWithGroupedDown` and
the MoE range tests now run the dual forms. Greedy output: gpt-oss-20b HF, 200 tokens of two prompts (74 and ~600 tokens), with the
switches on and off: byte-identical (`greedy/`). After merging `develop` (at `20021ee9`, then again at `986ac6c3` with the CUDA
provider), on GPU 2:

| suite | after `20021ee9`: total / failed / skipped | after `986ac6c3` |
|---|---|---|
| Sharplight.Vulkan.Tests (full) | 1624 / 0 / 26 | 1624 / 0 / 26 |
| Sharplight.Families.Tests (full, batched greedy parity included) | 1021 / 0 / 364 | 1340 / 0 / 683 (the CUDA parity cases added by the merge, skipped here) |

### 49.10 Next levers

* **gpt-oss decode step at 64 sequences** (now 17.1 ms against a ~14.6 ms weight bound; vLLM's ITL at 64 streams is 16.3 ms with its
  prompt steps included): the expert-grouped skinny GEMMs read at ~750 GB/s; the bf16 attention projections and LM head (5 ms per
  step) run at ~60 % of bandwidth.
* **Prompt steps of gpt-oss** (2048 rows: 165-170 ms, ~12k tok/s): the grouped MXFP4 gemm_cm2 at 256 rows per expert runs at ~90
  TFLOPS against ~120 for the dense bf16 GEMM of the same size.
* **Qwen3-8B-FP8 at 16 streams** (0.83x): its decode steps (15.1 ms ITL against 12.9) and FP8 prefill (0.73x).
* TTFT at 16 streams: a smaller budget while few streams decode would keep TTFT at round 9's level without losing the 64-stream gain.

### 49.11 Reproduce

```bash
export SHARPLIGHT_VK_DEVICES=2 CUDA_VISIBLE_DEVICES=2 VLLM_VENV=<venv with vLLM 0.33.0> ROUNDS=3 SLOTS=64
# serving (49.6): before = round 9's behaviour
SHARPLIGHT_MOE_DUAL=0 SHARPLIGHT_MOE_BIAS_EPILOGUE=0 OUT_DIR=bench/results/s49/serve-before \
  bench/s16_serve.sh sharplight gpt-oss-20b-hf ~/models/hf/gpt-oss-20b "16 64" --batch 512
OUT_DIR=bench/results/s49/serve-after bench/s16_serve.sh sharplight gpt-oss-20b-hf ~/models/hf/gpt-oss-20b "16 64"
OUT_DIR=bench/results/s49/serve-vllm bench/s16_serve.sh vllm gpt-oss-20b-hf ~/models/hf/gpt-oss-20b "16 64"
# step times (49.5)
SHARPLIGHT_MOE_DUAL=0 SHARPLIGHT_MOE_BIAS_EPILOGUE=0 bench/BatchBench/bin/Release/net11.0/BatchBench ~/models/hf/gpt-oss-20b \
  --slots 68 --ctx 2560 --batch 2048 --depth 256 --sizes 16 --mixed 16+0,64+0,64+107x1,64+448x1,0+2048x1 --modes greedy --reps 12
```
