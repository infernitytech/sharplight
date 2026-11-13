# KV cache: paged blocks, prefix sharing, quantised storage (design)

Status: design. Today's cache and its limits are described first, then the target design, the order in which it lands and
how each step is tested. Nothing here changes the public API until step 3; the engine-facing contract is `IBatchDecoder`.

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
- Storage is `AdapterOptions.KvCacheType` (F16 default, F32 possible). The CPU backend accepts F16/F32 only
  (`CpuOpSupport.IsKvCacheType`), Vulkan likewise.
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

### 2.4 Quantised KV storage

Formats (per row of `head_dim` values, blocks of 32 along `head_dim`, same bit layout as GGUF so existing dequant helpers
apply):

| type | bytes per 32 values | vs F16 | notes |
|---|---|---|---|
| F16 | 64 | 1x | default |
| Q8_0 | 34 (f16 scale + 32 × i8) | 0.53x | perplexity change typically < 0.1 % |
| Q4_0 | 18 (f16 scale + 32 × 4 bit) | 0.28x | opt-in; noticeable quality loss on small models, mainly for V |

- `CacheWrite` quantises the new rows on write (one workgroup per row block: max-abs reduction, scale, round).
- Attention dequantises in the inner loop: K tiles feed the QK dot product as `scale × int8` (on Vulkan with an int8 dot product
  where the device supports it, else converted to f16 in shared memory), V tiles are dequantised when
  loaded. CPU: the same per-block dequant in the vectorised attention kernel.
- Separate K and V types (`KvCacheTypeK`, `KvCacheTypeV`), since K is more sensitive than V.
- Feature detection as above: a device or backend without the quantised kernels rejects the type at compile time and the
  loader falls back to F16 with a log line (never a hard failure).

## 3. Steps

1. Quantised KV in the contiguous layout (independent of paging): Q8_0 `CacheWrite` + `CachedAttention` for CPU and Vulkan,
   `LlmLoadOptions.KvCacheType` / `--kv-type` in `sharplight-serve`, footprint estimate updated. Coordinate with the GPU-kernel
   work (attention shaders are shared).
2. Paged ops (`block_table` input, paged `CacheWrite` / `CachedAttention`) with a trivial block manager that gives each slot
   a fixed range of blocks: bit-identical outputs to today, proves the ops.
3. `KvBlockPool` + dynamic allocation + admission by blocks + preemption by recompute; `IBatchDecoder` gains
   `KvBlocksFree` / `KvBlockSize` and per-sequence block management (`Truncate`, `Reset` release blocks).
4. Prefix sharing by block hashes (attention-only models), `n > 1` sharing with copy-on-write.
5. Optional: Q4_0, host swap, recurrent-state snapshots for hybrid prefix sharing.

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
