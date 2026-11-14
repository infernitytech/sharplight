# Benchmarks: Sharplight vs llama.cpp vs vLLM

Baseline measured on 2026-11-13. Measurement only: nothing under `src/` or `tests/` was modified. All scripts and raw results
(JSON/JSONL/text, GPU snapshots, server logs) live in [`bench/`](../bench) and [`bench/results/`](../bench/results).

## 1. Setup and conditions

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB, sm_120, 300 W power limit, driver 610.57.04). Boost clock up to 3090 MHz, about 2340 MHz observed during runs |
| CPU / RAM | Ryzen 9 9950X, 91 GB (typically 55-70 GB used by the desktop session, so little free RAM) |
| OS | Fedora 44, Linux 7.1.8 |
| Sharplight | working tree of 2026-11-13 (no git metadata available), `dotnet build -c Release`, .NET SDK 11.0.100, Vulkan provider with Slang kernels; benchmarked binaries: `samples/Sharplight.Cli` (`--bench`), `samples/Sharplight.Server` |
| llama.cpp | b11177 (commit 1ab7e5ad2), builds `runtime.llama-b11177-cuda13` and `runtime.llama-b11177-vulkan` (Vulkan device 0 = the NVIDIA GPU; the RADV iGPU is ignored by default), `-ngl 99 -fa 1/on` |
| vLLM | 0.33.0 in `~/.venvs/vllm` (uv, Python 3.12.13, torch 2.14.0+cu130, flash-attn backend, CUDA graphs on). `VLLM_USE_FLASHINFER_SAMPLER=0` because no CUDA toolkit matching the wheels is installed (FlashInfer JIT needs nvcc). Prefix caching off |

**Shared GPU.** The GPU is shared with the user's desktop (kwin, VS Code, Chrome, Rider, a Python process) and three idle
`sharplight-server` processes that were already running (about 10-13 GB of VRAM in use before every run, 0-2 % utilisation when idle).
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

**Prompt-lookup speculation** (`sharplight-server --speculate-ngram N`; an edit task whose answer copies the prompt, Llama-1B,
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
#   Sharplight:  sharplight-server MODEL --slots 64 --ctx 512 --listen http://127.0.0.1:18080 --temp 0
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

Sharplight serve/CLI binaries used: `samples/Sharplight.Cli/bin/Release/net11.0/sharplight`, `samples/Sharplight.Server/bin/Release/net11.0/sharplight-server`.
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

`sharplight-server MODEL --ctx 4096 --slots S [--kv-tokens N]`; `bench/loadgen.py -c 64 --per-worker 6 --ignore-eos` (251-token
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

Conditions: the GPU was shared with a running game and with other benchmark agents for the whole session, so end-to-end
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

Wall clock (`sharplight MODEL --bench 512 --ctx 768`, best of 3 alternating process runs). These runs shared the GPU with a game
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
were repeated because a game was running on the GPU).

### 16.1 Setup

* Same machine and engine builds as section 1 (RTX PRO 6000 Blackwell Max-Q, 300 W cap; vLLM 0.33.0 with FLASH_ATTN, CUDA
  graphs, async scheduling, `VLLM_USE_FLASHINFER_SAMPLER=0`, prefix caching off; llama.cpp b11177). Sharplight: `main` at
  ed1dd8f, Release build.
* **Identical weights** for Sharplight and vLLM: `Qwen/Qwen3.8-27B-FP8` (block-scaled E4M3; Sharplight loads the vLLM
  checkpoint natively), `Llama-3.2-1B-Instruct` bf16 and `OLMoE-1B-7B-0924-Instruct` bf16 (HF safetensors). llama.cpp cannot
  load the FP8 checkpoint and ran the 27B as Q8_0 GGUF; Sharplight on that GGUF is listed for reference.
* **Conditions.** The GPU sat at its 300 W limit and 85-89 °C during the long serving runs (SM clock 1.75-2.3 GHz), so absolute
  numbers are 10-20 % below the cooler runs of section 2 for *both* engines (vLLM 27B decode 48.5 → 39.5 tok/s). Each comparison
  below was measured in the same period. A game (~1.2 GB, Proton) ran on the GPU from the vLLM small-model single-stream runs to
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
| OLMoE bf16 (game running) | 1 | 386 (356..386) | 346 (345..356) | 1.12x | 16 / 18 | 15 / 18 | 2.49 / 3.6 | 2.79 / 4.1 | 2.57 (2.9) |
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
The single-stream rows ran while the game was on the GPU (baseline 38.4 instead of 39.5), so read the ratios.

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
of them, plus a game), so absolute numbers are about half of 17.1 / 17.2 and only interleaved A/B comparisons mean
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
profiles, `spec/` SpecBench, serving at the top level, `disturbed/` a first pass that ran while a game used the GPU).

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

The goal is to run 15 representative ONNX models, all imported through `Sharplight.Onnx`, as fast as onnxruntime on the
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

## 24. MoE: raw MXFP4 expert fetch, split expert tiles, dual gate/up and slot-sum expert matvecs (2026-11-14)

Follow-up to sections 14 and 22 (gpt-oss-20b prefill at 0.68x and decode at 0.93x of llama.cpp CUDA). All changes are
kernel or fusion rules for every MoE model (MatMulId):

* `matmul.slang` (`gemm_cm2`, `MOE`), `common.slang`, `WeightRepacker.cs`: MXFP4 experts in the expert-grouped GEMM are
  fetched as raw codes and scales one step ahead and decoded after the MMAs; the aligned MXFP4 layout stores its scales
  block-major per 16-row group.
* `GraphLowerer.Moe.cs`: tiles of at most half the tile height run in a second launch on half-height tiles.
* `GraphLowerer.Fusion.cs` (`FoldDualMatMuls`, `FoldExpertSums`), `GraphLowerer.Moe.cs` (`LowerDualMatMulId`,
  `LowerExpertSum`), `mmvq.slang` (`moe_mmvq`, `DUAL`, `SLOT_SUM`): decode fusions on the integer-dot expert matvec.

Base is `main` (`95cc0a2` for 24.1-24.3, `d0d3ea6` for the correctness runs), built side by side; base and new runs
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

### 24.6 Correctness

Greedy CLI output, 200 tokens, long (~650-token) and short prompts, against `main` at `d0d3ea6`: identical for gpt-oss-20b,
OLMoE, Qwen3.6-35B-A3B, Qwen2.5-0.5B, Mistral-7B Q4_K_M, the 27B Q4_K_M, Q8_0 and FP8. MTP k = 3 against
non-speculative output on the 27B Q8_0 and FP8: identical. A 1921-token prompt on gpt-oss (128-row expert tiles,
split) gives the same output with and without split tiles.

Tests: new `MoeTests.RoutedExpertBlock` (gate/up with per-expert biases, activation, down with bias, router weights and
slot sum; mxfp4 / q4_k / q8_0; 1-70 tokens; S = 2, 4, 8) on the dual and slot-sum launches and the prefill path;
`MoeTests` and `FusionTests` pass on the NVIDIA GPU, `MoeTests` on llvmpipe.

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
