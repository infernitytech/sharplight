# KV cache: paged blocks, prefix sharing, quantised storage

Status: quantised KV storage (section 2.4) and paged blocks with admission by blocks, preemption by recomputation and prefix
sharing between slots (section 2.5) are implemented; sections 2.1-2.3 are the original design, 2.5 says where the code
differs. The contiguous cache and its limits are described first (it remains the default without `--kv-tokens`), then the
design, the order in which it landed and how each step is tested. The engine-facing contract
is `IBatchDecoder`.

## 1. Contiguous cache (the default)

- `MultiSequence.Transform` keeps every KV state at its single-sequence shape `[Ctx, heads, head_dim]` and compiles it with
  `Ctx = slots × per-slot context`. Slot `s` owns rows `[s·C, (s+1)·C)`. `CacheWrite` scatters new rows through the
  `SequenceRows` table, `CachedAttention` reads each run's rows up to the query position.
- Memory is reserved for the worst case: every slot can hold a full context, whether its request uses 50 tokens or 4000.
  With 64 slots × 4096 tokens the Llama-1B cache alone is 64 × 4096 × 16 layers × 2 × 512 × 2 bytes = 8 GiB, of which typical
  chat traffic touches a few percent. Slots, not memory, limit concurrency.
- Prefix reuse is per slot: a finished slot keeps its tokens, and a new request whose prompt starts with them is put into
  that slot and skips the shared part (`EngineOptions.PrefixCache`, `GenerationUsage.CachedTokens`). Two concurrent requests
  with the same system prompt each compute and store it.
- Storage is `LlmLoadOptions.KvCacheType` / `AdapterOptions.KvCacheType`: F16 (default), F32, Q8_0 or Q4_0 (section 2.4),
  on the CPU and on Vulkan.
- Recurrent state (Mamba, RWKV, gated delta nets, conv) has a leading `[S]` slot dimension and is small and fixed; it is not
  part of this design except where noted (prefix sharing cannot reuse it).

## 2. Target

### 2.1 Paged storage

Each KV state becomes a pool of fixed-size blocks: `[NumBlocks, BlockSize, heads, head_dim]` with `BlockSize = 16` rows
(tunable per device; 16 matches the attention kernels' row tiling and keeps internal fragmentation below one block per
sequence). All layers share one block numbering, so one table describes a sequence for every layer.

New graph input (added by `MultiSequence.Transform` when the decoder is created paged):

| input | role | shape | meaning |
|---|---|---|---|
| `block_table` | `InputRole.BlockTable` | `[U, MaxBlocksPerSeq]` i32 | per run (sequence in this step): physical block of logical block j; -1 unused |

`SequenceRows` keeps its meaning (row → run, position); the physical row of a token at position p of run u is
`block_table[u, p / BlockSize] · BlockSize + p % BlockSize`. Ops:

- `CacheWrite` (paged form): computes the physical row from the tables instead of `slot · C + p`.
- `CachedAttention` (paged form): iterates the run's blocks through the table. The Vulkan flash-attention path already
  loads K/V in row tiles; a tile becomes one block, so the inner loop gains one table lookup per 16 rows. The split-K decode
  path assigns blocks, not row ranges, to workgroups.
- Recurrent states keep the `[S]` slot dimension; a paged decoder still has a slot count, which now only bounds the batch
  (and recurrent state), while KV capacity is `NumBlocks`.

Backends advertise the paged forms through their op-support tables (`CpuOpSupport.Sequence`, the Vulkan equivalent).
A backend or device without them keeps the contiguous layout: `CreateBatchDecoder` falls back and reports
`IBatchDecoder.Paged = false`, and the engine behaves as today. Neutral feature name: `paged-kv`, switchable off per device
and per compile like the other device features, with the contiguous path as the always-available fallback.

### 2.2 Block manager (host side, `Sharplight.Llm.Serving.Kv`)

- `KvBlockPool`: free list, reference count per block, and an LRU of *cached* blocks (refcount 0 but content still valid).
- A sequence owns a `BlockTable` (list of physical blocks). Appending a token past a block boundary allocates one block;
  `Truncate` (speculation rollback) releases whole blocks past the new length.
- Allocation failure while decoding triggers preemption: the lowest-priority, most recently admitted sequence is evicted
  (its blocks released, its request re-queued at the front with prompt = prompt + generated tokens, recomputed on
  re-admission). Swapping to host memory is a later option; recomputation is simpler and cheap for small models.
- Admission control moves from "free slot" to "free slot and enough free + evictable blocks for the prompt plus a reserve"
  (`EngineOptions.KvReserveBlocks`), which is what lets 64 short chats share memory sized for 8 long ones.

### 2.3 Prefix sharing

- A *full* block is identified by `hash(parent_hash, tokens of the block)` (plus a model/adapter salt). The pool keeps
  `hash → block` for full blocks of live and cached sequences.
- On admission the prompt is cut into full blocks; the longest chain of hashes found in the pool is mapped into the new
  sequence's table with refcount + 1, and prefill starts after it. Blocks are immutable once full, so sharing needs no
  copy; the partial last block is always private (copy-on-write is therefore never needed for prompt prefixes; it is needed
  only for `n > 1` sampling from one prompt, where the shared partial block is copied when the first branch writes to it).
- This replaces the per-slot prefix cache for paged decoders: reuse works across slots and between concurrent requests.
  Metrics: `kv_prefix_hit_tokens_total`, `kv_blocks_shared`.
- Hybrid models: the recurrent state at the end of a shared prefix is not stored, so prefix sharing is disabled for models
  with recurrent state. A later step can snapshot recurrent state at block boundaries. (Speculation works for them: rejected
  drafts are rolled back by record-and-replay of the recurrent ops, see `Graph.RecurrentRollback`; the record state is
  `[slots, 2, k + 1, W]` f32 per recurrent op, about 35 MB per slot for Qwen3.8-27B with 3 drafts, against about 160 MB of
  recurrent state per slot that per-position snapshots would copy k + 1 times.)

### 2.4 Quantised KV storage (implemented)

Formats (per row of `head_dim` values, blocks of 32 along `head_dim`, the GGUF bit layout, so the CPU dequantisers apply):

| type | bytes per 32 values | vs F16 | notes |
|---|---|---|---|
| F16 | 64 | 1x | default |
| Q8_0 | 34 (f16 scale + 32 × i8) | 0.53x | scale amax / 127, round to nearest |
| Q4_0 | 18 (f16 scale + 32 × 4 bit) | 0.28x | scale = value of largest magnitude / -8 (llama.cpp's reference quantiser) |

Selecting it: `LlmLoadOptions.KvCacheType` (overrides `AdapterOptions.KvCacheType`), `--kv-type f16|f32|q8_0|q4_0` in the CLI
and in `sharplight-server`. The adapters declare each KV state through `AdapterOptions.KvCacheTypeFor(row length)`: a cache whose
rows per head are not whole 32-value blocks stays F16, and `LlmModel.Load` logs how many did. K and V share one type, and the
MLA latent cache (one row holds keys and values) is quantised as a whole. `EstimateFootprint` and placement size states from
their types, so the smaller cache shows up in the slot / layer budget.

Kernels:

- `CacheWrite`: Vulkan `cache_write_q` (`data.slang`), one thread per 32-value block, 16-bit stores (blocks are 2-byte
  aligned); CPU `Quant.Quantize`. The rope → cache-write fusion is skipped for quantised caches (one more dispatch per layer).
- `CachedAttention` on Vulkan (`attention.slang`, macro `KV_FMT`): every path reads the cache through one set of helpers
  (`Kv8`, `load_kv8`, `unpack_kv8`, `kv8_to_half`): the scalar `main` kernel, split-K `decode`, the tensor-core `fa` (K/V tiles
  converted to f16 in shared memory) and the shared-row MLA kernel `fa_mqa`, which loads whole blocks with 32-bit loads.
  CPU: rows dequantised per key before the dot product / axpy.
- Not done: int8 dot products for K (the f16 tile path is used), separate K and V types, the fallback when a backend
  lacks the quantised kernels (both backends have them; an unsupported shape falls back to F16 at definition time).

Measured (2026-11-13, RTX PRO 6000 shared with other jobs, `sharplight MODEL --bench 4096 --ctx 32768 --kv-type T`; accuracy:
last-token logits after a 2.8k-token prompt, KL divergence from the F16-cache run):

| model | KV type | weights + state | prefill 4096 tok/s | decode tok/s @ 4k | KL vs F16 | top-1 |
|---|---|---|---|---|---|---|
| Mistral-7B Q4_K_M | f16 | 8.07 GiB | 8109 | 227 | – | – |
| | q8_0 | 6.20 GiB | 7401 | 215 | 0.0006 | same |
| | q4_0 | 5.20 GiB | 7269 | 215 | 0.079 | same |
| DeepSeek-V2-Lite Q4_K_M (MLA) | f16 | 10.64 GiB | 8637 | 304 | – | – |
| | q8_0 | 10.19 GiB | 8201 | 289 | 0.0045 | same |
| | q4_0 | 9.95 GiB | 8272 | 290 | 0.47 | same |
| MiniCPM3-4B (MLA) | q8_0 / q4_0 | | | | 0.000 / 0.0002 | same |
| Qwen3-0.6B Q8_0 | q8_0 / q4_0 | | | | 0.018 / 3.8 | differs |

- Memory: the KV share of the footprint shrinks to 0.53x (q8_0) and 0.28x (q4_0): Mistral at 32k context 4 GiB → 2.1 / 1.1 GiB.
- Speed: 5-9 % slower prefill and decode than F16 (more ALU per loaded value; the decode kernel issues five 16-bit loads per
  8 values). Worth it where memory limits context or concurrency, not for speed.
- Accuracy: q8_0 is close to F16. q4_0 is model dependent: fine on Mistral and MiniCPM3, poor on DeepSeek's latent cache and
  unusable on Qwen3-0.6B, whose keys have large outlier channels. CPU and Vulkan agree to the same KL (Qwen3-0.6B: q8_0 0.012 /
  0.018, q4_0 3.72 / 3.77), so the loss is the format's, not a kernel error.
- Tests: `StatefulTests.CachedAttention*` (Vulkan: q8_0 / q4_0 through decode, split-K, flash and the MLA kernels against a
  reference on the dequantised cache) and `StatefulOpTests.CachedAttentionAcrossCalls` (CPU, q8_0 / q4_0 against the reference
  on rounded keys and values); `BenchTests.LatentAttention` has q8_0 / q4_0 cases.

### 2.5 Paged blocks, admission by blocks, preemption, prefix sharing (implemented)

What landed differs from 2.1-2.3 in a few places; this section describes the code.

Selecting it: `LlmModel.CreateBatchDecoder(slots, ctx, kvCacheTokens: N)`, `ChatModelOptions.KvCacheTokens`, and
`sharplight-server --kv-tokens N` (K / M suffixes, binary). Without it every slot reserves `ctx` rows, as before.

Graph and kernels:

- `MultiSequence.Transform(g, slots, blockSize, blocksPerSlot)` sizes every KV state as a pool of `NumBlocks · 16` rows and
  adds one input `kv_block_table` (`InputRole.KvBlockTable`, `[S, MB]` i32, `MB = ⌈ctx / 16⌉`) as the last input of every
  `CacheWrite` and `CachedAttention` (attribute `block_size`). The table is per *slot*, not per run, so it only changes when a
  slot gains or loses a block and is uploaded only then. Physical row of position `p` in slot `s`:
  `table[s, p / 16] · 16 + p % 16`. All layers share one block numbering.
- Vulkan: one helper, `seq_row(pages, slot, pos)` in `common.slang` (macros `PAGED`, `PAGE_BS`, `PAGE_MB`), replaces
  `slot · SLOT_CTX + pos` in `cache_write`, `cache_write_q`, the fused rope → cache write, and every attention kernel
  (`main`, split-K `decode`, tensor-core `fa`, MLA `fa_mqa`, through `kv_base` / `kv_row`). Row offsets stay 64-bit integer
  arithmetic (the ≥ 4 GiB pointer-offset driver bug). Rows of one block are contiguous, so the existing 16- and 64-row tiles
  read whole blocks; the cost is one table load per tile. Quantised caches (2.4) page the same way.
- CPU: `KvPages` maps rows for `CacheWriteOp` and `AttentionOp`.
- Recurrent state keeps its `[S]` slot dimension; a paged hybrid model pages only its attention layers.

Host side:

- `KvBlockPool` (free stack + reference counts) inside `MultiSequenceDecoder`. A step grows each slot's block list to cover
  its new rows; `Reset` releases a slot's blocks; `Truncate` releases whole blocks past the cut (a cut inside a block still
  shared with another slot rounds down to the block start, and the engine re-feeds the difference).
- `IBatchDecoder` gains `KvBlockSize`, `KvBlocksTotal`, `KvBlocksFree`, `KvBlocksNeeded(slot, tokens)`,
  `KvBlocksReleasable(slot)` and `ShareKvPrefix(slot, source, tokens)`; defaults keep non-paged decoders unchanged.
- `InferenceEngine`:
  - capacity (`kv_capacity_tokens`) is the pool; a sequence's context is capped by `min(ctx, pool)`.
  - admission: a waiting request is admitted when a slot is free *and* the blocks for its prompt plus a growth reserve fit
    in the free blocks plus those idle slots could give back, after what the running sequences still need (the rest of
    their prompts plus the same growth reserve). The reserve is up to 64 generated tokens, at most a quarter of a slot's
    share of the pool. While a preempted sequence is recomputing nothing new is admitted.
  - before each step the blocks the decoding sequences need are freed: first idle slots' cached contexts (LRU), then by
    preempting the newest running sequence. Prefill and replay chunks only evict idle contexts and otherwise wait.
  - preemption by recomputation: the victim's blocks are released, it keeps its slot, and its context (prompt + generated
    tokens) is re-fed in prompt chunks without logits before it continues decoding. Its output is unchanged (greedy
    parity), only delayed. A step in which every running sequence waits for blocks preempts the newest one.
  - prefix sharing (attention-only models, `EngineOptions.PrefixCache`): at admission the engine looks for the slot (running
    or idle) whose context shares the longest prefix with the prompt and maps its full blocks into the new slot's table
    (reference count + 1, no copy, no hashing: slot contexts are compared token by token). Blocks are never written after
    they are full, so sharing needs no copy-on-write. The per-slot reuse of section 1 still applies on top.
  - metrics: `sharplight_kv_preemptions_total`, `sharplight_kv_prefix_hit_tokens_total`.
- Not done: block hashes (lookup is a linear scan of the slots' tokens, fine for tens of slots), an LRU of freed-but-valid
  blocks beyond idle slots, copy-on-write for `n > 1`, host swap, a separate per-device switch (paging needs no device
  feature; the contiguous layout remains when `--kv-tokens` is not given).

Tests: `PagedAttentionTests` (Vulkan vs the CPU oracle on shuffled block tables with holes and partial last blocks, decode,
split-K, flash and MLA kernels, f16 / f32 / q8_0 / q4_0), `MultiSequenceTests.Paged_Matches_Independent_Sessions` (greedy parity of a paged decoder with
a pool smaller than slots × ctx against single-sequence sessions), `PagedEngineTests` (admission by blocks, preemption and
recompute, sequences capped by the pool, prompt prefixes shared across concurrent requests) and the existing serving tests
on the block-accounting `FakeBatchDecoder`.

Measured (2026-11-13, RTX PRO 6000 shared with other jobs; `sharplight-server MODEL --ctx 4096 --slots S [--kv-tokens N]`,
`bench/loadgen.py -c 64 --per-worker 6 --ignore-eos`: 64 streams in a closed loop, 384 requests of ~251 prompt tokens and 128
output tokens; VRAM is the server process's allocation from `nvidia-smi`, "at once" the peak of running sequences; raw
results in `bench/results/paged/`):

| model | KV layout | KV tokens | VRAM | at once | agg tok/s | TTFT p50 / p95 ms | ITL mean ms | preemptions |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-1B Q8_0 | fixed, 4 slots | 16K | 2100 MiB | 4 | 2248 | 3355 / 3580 | 1.7 | – |
| | paged, 64 slots | 16K | 2100 MiB | 51 | 7115 | 160 / 761 | 5.9 | 234 |
| | paged, 64 slots | 24K | 2356 MiB | 64 | 9930 | 56 / 288 | 5.7 | 0 |
| | fixed, 16 slots | 64K | 3636 MiB | 16 | 3957 | 1555 / 1637 | 4.0 | – |
| | paged, 64 slots | 64K | 3636 MiB | 64 | 10187 | 40 / 312 | 5.7 | 0 |
| | fixed, 64 slots | 256K | 9780 MiB | 64 | 10498 | 35 / 284 | 5.6 | – |
| Mistral-7B Q4_K_M | fixed, 16 slots | 64K | 12696 MiB | 16 | 893 | 6342 / 9270 | 17.3 | – |
| | paged, 64 slots | 64K | 12678 MiB | 64 | 1976 | 212 / 1593 | 29.5 | 0 |
| | paged, 64 slots | 24K | 7814 MiB | 64 | 2011 | 210 / 1547 | 29.0 | 0 |
| | fixed, 64 slots | 256K | 37254 MiB | 64 | 2065 | 209 / 1570 | 28.2 | – |

Shared system prompt (`--shared-prefix-words 800 --prompt-words 50`: every prompt starts with the same ~800 tokens, 852
tokens on average; Llama-1B):

| KV layout | KV tokens | VRAM | at once (context tokens) | agg tok/s | TTFT p50 / p95 ms | prefix tokens shared |
|---|---|---|---|---|---|---|
| fixed, 16 slots | 64K | 3636 MiB | 16 (15.6K) | 3839 | 1602 / 1756 | – |
| paged, 64 slots | 16K | 2100 MiB | 64 (61.6K) | 11438 | 83 / 154 | 48000 |
| paged, 64 slots | 64K | 3636 MiB | 64 (62.2K) | 11069 | 78 / 146 | 48000 |
| fixed, 64 slots | 256K | 9780 MiB | 64 (62.2K) | 11399 | 38 / 96 | – (per-slot reuse) |

- Concurrency per memory: in the same VRAM the paged cache runs 4x the sequences of the fixed layout (16 → 64 in 3.6 GiB for
  Llama-1B, 12.4 GiB for Mistral; 64 is the slot count, the 64K pool would hold about 170 such sequences) and 2.2-2.6x the
  throughput at c = 64. A 24K pool (a tenth of the fixed 64-slot cache) is enough for 64 such streams: Mistral at 7.8 GiB
  instead of 37 GiB within 3 % of the throughput. With a shared system prompt 64 contexts of ~980 tokens (62K tokens) fit
  in a 16K pool.
- Cost of paging: 1-4 % lower throughput than the fixed layout at the same concurrency (within the run-to-run noise of this
  shared GPU; one table load per 16-row tile).
- An undersized pool degrades gracefully: at 16K (0.65 of what 64 streams need) 51 run at once and newer ones are preempted
  and recomputed (234 preemptions for 384 requests), still 3.2x the 4-slot fixed layout in the same memory. Admission keeps
  up to 64 tokens of growth room per running sequence (at most a quarter of a slot's share of the pool), which cut the
  preemptions at 16K from 850 to 234 and raised throughput from 5230 to 7115 tok/s.
- TTFT with the shared prefix is higher on the paged decoder than on a fixed 64-slot cache whose every slot already holds the
  prefix: a pool sized for the working set evicts idle contexts and maps shared blocks instead of reusing a slot's own copy.

## 3. Steps

1. Done: quantised KV in the contiguous layout (independent of paging): Q8_0 and Q4_0 `CacheWrite` + `CachedAttention` for
   CPU and Vulkan, `LlmLoadOptions.KvCacheType` / `--kv-type`, footprint estimate from the state types.
2. Done: paged ops (`kv_block_table` input, paged `CacheWrite` / `CachedAttention` on CPU and Vulkan, all attention kernels).
3. Done: `KvBlockPool` + dynamic allocation + admission by blocks + preemption by recompute; `IBatchDecoder` gains
   `KvBlocksFree` / `KvBlockSize` and per-sequence block management (`Truncate`, `Reset` release blocks); `--kv-tokens`.
4. Done: prefix sharing between slots (attention-only models), by comparing slot contexts instead of block hashes.
   Open: `n > 1` sharing with copy-on-write.
5. Optional: host swap, recurrent-state snapshots for hybrid prefix sharing.

## 4. Testing and acceptance

- Op level: paged vs contiguous attention on random tables (holes, shuffled blocks, partial last block), CPU vs Vulkan,
  for F16 and Q8_0; quantised vs F16 within a tolerance per format.
- Model level: `MultiSequenceTests` parity (paged decoder vs single-sequence session) across architectures; perplexity delta
  on a fixed text for Q8_0 / Q4_0 KV (Llama-1B, Qwen3-0.6B, the 27B).
- Engine level (fake decoder with a block budget): admission by blocks, preemption order and recompute correctness,
  prefix hits across concurrent requests, rollback releasing blocks.
- Serving: `bench/loadgen.py` at c = 64 with 512-token contexts and with a shared 1k-token system prompt; acceptance is more
  concurrent sequences per GiB of KV and lower TTFT for shared prefixes at unchanged ITL. `/metrics` already reports
  `kv_capacity_tokens`, `kv_tokens`, `kv_active_tokens`, `kv_utilization`; paging adds block counts and prefix hits.
