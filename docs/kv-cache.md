# KV cache: paged blocks, prefix sharing, quantised storage (design)

Status: step 1 (quantised KV storage, section 2.4) is implemented; the rest is design. Today's cache and its limits are
described first, then the target design, the order in which it lands and how each step is tested. The engine-facing contract
is `IBatchDecoder`.

## 1. Today

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
  with recurrent state (as speculation is). A later step can snapshot recurrent state at block boundaries.

### 2.4 Quantised KV storage (implemented)

Formats (per row of `head_dim` values, blocks of 32 along `head_dim`, the GGUF bit layout, so the CPU dequantisers apply):

| type | bytes per 32 values | vs F16 | notes |
|---|---|---|---|
| F16 | 64 | 1x | default |
| Q8_0 | 34 (f16 scale + 32 × i8) | 0.53x | scale amax / 127, round to nearest |
| Q4_0 | 18 (f16 scale + 32 × 4 bit) | 0.28x | scale = value of largest magnitude / -8 (llama.cpp's reference quantiser) |

Selecting it: `LlmLoadOptions.KvCacheType` (overrides `AdapterOptions.KvCacheType`), `--kv-type f16|f32|q8_0|q4_0` in the CLI
and in `sharplight-serve`. The adapters declare each KV state through `AdapterOptions.KvCacheTypeFor(row length)`: a cache whose
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

## 3. Steps

1. Done: quantised KV in the contiguous layout (independent of paging): Q8_0 and Q4_0 `CacheWrite` + `CachedAttention` for
   CPU and Vulkan, `LlmLoadOptions.KvCacheType` / `--kv-type`, footprint estimate from the state types.
2. Paged ops (`block_table` input, paged `CacheWrite` / `CachedAttention`) with a trivial block manager that gives each slot
   a fixed range of blocks: bit-identical outputs to today, proves the ops.
3. `KvBlockPool` + dynamic allocation + admission by blocks + preemption by recompute; `IBatchDecoder` gains
   `KvBlocksFree` / `KvBlockSize` and per-sequence block management (`Truncate`, `Reset` release blocks).
4. Prefix sharing by block hashes (attention-only models), `n > 1` sharing with copy-on-write.
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
