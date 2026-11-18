# KV cache: paged blocks, prefix sharing, quantised storage

Status: quantised KV storage (section 2.4), paged blocks with admission by blocks and preemption by recomputation (section 2.5)
and prefix caching by block content with cache namespaces and recurrent-state snapshots (section 2.6) are implemented, and the
paged cache with prefix caching is the default of multi-slot decoders. Sections 2.1-2.3 are the original design; 2.5 and 2.6 say
where the code differs. The contiguous cache and its limits are described first (the fallback where the devices lack the paged
kernels, with `--kv-tokens 0`, or a single slot), then the design, the order in which it landed and how each step is tested. The
engine-facing contract is `IBatchDecoder`.

## 1. Contiguous cache (the fallback)

- `MultiSequence.Transform` keeps every KV state at its single-sequence shape `[Ctx, heads, head_dim]` and compiles it with
  `Ctx = slots × per-slot context`. Slot `s` owns rows `[s·C, (s+1)·C)`. `CacheWrite` scatters new rows through the
  `SequenceRows` table, `CachedAttention` reads each run's rows up to the query position.
- Memory is reserved for the worst case: every slot can hold a full context, whether its request uses 50 tokens or 4000.
  With 64 slots × 4096 tokens the Llama-1B cache alone is 64 × 4096 × 16 layers × 2 × 512 × 2 bytes = 8 GiB, of which typical
  chat traffic touches a few percent. Slots, not memory, limit concurrency.
- Prefix reuse is per slot: a finished slot keeps its tokens, and a new request whose prompt starts with them is put into
  that slot and skips the shared part (`EngineOptions.PrefixCache`, `GenerationUsage.CachedTokens`), only when both requests
  have the same cache namespace (partition and salt, section 2.6) and the same non-text content up to that point. Two concurrent
  requests with the same system prompt each compute and store it.
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
  with recurrent state. A later step can snapshot recurrent state at block boundaries (done, section 2.6). (Speculation works for them: rejected
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
and in `sharplight serve`. The adapters declare each KV state through `AdapterOptions.KvCacheTypeFor(row length)`: a cache whose
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

Selecting it: the default of `LlmModel.CreateBatchDecoder` with more than one slot (or speculation), `ChatModelOptions.KvCacheTokens`
and `sharplight serve --kv-tokens N` (K / M suffixes, binary) size the pool; section 2.6 has the defaults. `--kv-tokens 0`
(`kvCacheTokens: 0`) keeps every slot's fixed `ctx` rows.

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
  - prefix sharing: by block content since section 2.6 (it first compared slot contexts token by token). Blocks are never
    written after they are full, so sharing needs no copy-on-write. The per-slot reuse of section 1 still applies on top.
  - isolation (`GenerationRequest.CachePartition`, set from the request's scope by `Sharplight.AspNetCore`): prefix reuse of a
    slot's own context and block sharing only happen between requests of the same partition, so identical prompts of different
    tenants never share cache entries (no cross-tenant timing side channel); null is one partition of its own, the default.
  - metrics: `sharplight_kv_preemptions_total` and the prefix-cache metrics of section 2.6.
- Not done: copy-on-write for `n > 1`, host swap.

Tests: `PagedAttentionTests` (Vulkan vs the CPU oracle on shuffled block tables with holes and partial last blocks, decode,
split-K, flash and MLA kernels, f16 / f32 / q8_0 / q4_0), `MultiSequenceTests.Paged_Matches_Independent_Sessions` (greedy parity of a paged decoder with
a pool smaller than slots × ctx against single-sequence sessions), `PagedEngineTests` (admission by blocks, preemption and
recompute, sequences capped by the pool, prompt prefixes shared across concurrent requests) and the existing serving tests
on the block-accounting `FakeBatchDecoder`.

Measured (2026-11-13, RTX PRO 6000 shared with other jobs; `sharplight serve MODEL --ctx 4096 --slots S [--kv-tokens N]`,
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

### 2.6 Prefix caching by block content (implemented, the default)

Automatic prefix caching in the sense of vLLM's APC: every full block of a paged cache is named by a hash of its content, a block
that no sequence holds any more keeps its rows until its memory is needed, and a later prompt that starts with the same blocks
attaches them instead of computing them.

Defaults and sizing:

- `LlmModel.CreateBatchDecoder` with more than one slot (or speculation) builds a paged decoder by default: a pool of `slots × ctx`
  tokens (`LlmModel.DefaultKvCacheTokens`, the memory the contiguous cache reserved, so placement and the model host see the same
  footprint as before). The pool is used only where every device that runs the cache ops has their paged forms (`Supports` of the
  transformed nodes, neutral feature name `DeviceFeatures.PagedKv` = `paged_kv`, which `LlmLoadOptions.DisabledFeatures` or
  `SHARPLIGHT_DISABLE_FEATURES=paged_kv` turn off); otherwise, with `kvCacheTokens: 0` / `--kv-tokens 0`, or for one slot without
  speculation (a single-sequence session), the contiguous cache of section 1 remains.
- Hosted chat models (`LlmChatModelLoader`, so `sharplight serve` and Studio): when the model host has to load a model into less memory
  than its footprint (`IPreparedModel.Load(limit)`), a default-sized pool shrinks to fit (to at least one full context plus a block per
  slot) before layers are offloaded; an explicit `ChatModelOptions.KvCacheTokens` is kept.
- `EngineOptions.PrefixCache` (default on; `sharplight serve --no-prefix-cache`) turns all prefix reuse off.

Hashing and namespaces (`KvBlockHash`):

- Block j's hash is SHA-256, truncated to 128 bits, of block j - 1's hash, the block's token ids and the hashes of the non-text
  content overlapping the block (`GenerationRequest.PromptContent`: image or audio placeholder spans with a content hash, the
  equivalent of vLLM's multimodal hashes). The chain starts at a seed of the request's *cache namespace*: `GenerationRequest.CachePartition`
  (the isolation scope a host sets, e.g. the tenant or space) and `GenerationRequest.CacheSalt` (vLLM's `cache_salt`). Requests of
  different namespaces never produce equal hashes, so they never share blocks, never wait for each other's blocks and never reuse
  each other's slots (no cross-tenant timing side channel). Null partition and salt are one shared namespace, the single-scope default.
  A collision-resistant hash keeps a prompt from being crafted to alias another prompt's blocks within a namespace.
- Hashing runs on the scheduler thread once per prompt at admission (about 1 µs per block) and once per block that fills while decoding.
- The model and its adapters are fixed per engine (one engine per loaded model and pool), so they need no part in the key; per-token
  adapter routing (`SwitchedLora`) is part of the model graph.

Block pool (`KvBlockPool`, `KvBlockCache`):

- Reference counts, a content index `hash → block`, and one free queue: blocks without content first, then *cached* blocks (free,
  content kept) in the order they were released. A sequence releases its blocks last first, so its tail is evicted before its prefix,
  and a system prompt shared by many sequences is released last. `Allocate` takes the head and, for a cached block, drops its hash
  (an eviction). Attaching a cached block takes it out of the queue.
- A block is published (named) only when it is full; a published or shared block is never written again (a truncation into one drops
  the whole block, as for shared blocks in 2.5). A duplicate (the same content computed twice) stays private.
- Idle slots keep their contexts as before; evicting one (for blocks) now only moves its blocks to the cached end of the queue.

Engine (`InferenceEngine`):

- Admission: the slot's own context (section 1) or the cached blocks of the prompt, whichever reaches further; the attached blocks
  count as cached prompt tokens (`GenerationUsage.CachedTokens`, `prompt_tokens_details.cached_tokens`). Admission by blocks counts
  only the blocks the prompt must newly take: cached blocks other sequences hold cost nothing (cached free blocks are counted as
  free already).
- Mid-prompt hits: before every prefill chunk the engine attaches blocks published since admission, so a prompt arriving while another
  prompt with the same prefix is still prefilling does not compute the prefix again: the blocks a prefilling prompt will compute are
  registered at its admission, and a later prompt's chunk stops before the first such block of another prompt and attaches it once it
  is published (waits follow admission order, so they cannot cycle). Prefill chunks that do not end the prompt end on a block boundary.
  Attached tokens take no step budget.
- Preemption: a preempted sequence's blocks are released into the cache; when it resumes, the blocks still cached are attached and
  only the rest is recomputed.
- Speculation: drafts are verified as before; published blocks are full and committed, rollbacks only touch the private last block.
  A draft head's own cache treats attached prefixes as a gap (as for reused prefixes before).
- `GenerationRequest.BypassPrefixCache`: nothing is reused (for requests that need every prompt position computed, e.g. prompt
  log-probabilities, which vLLM also computes without the cache); the request's blocks are still published.

Models with recurrent state (gated delta nets, Mamba, Mamba-2, RWKV, short convolutions; hybrids and pure recurrent models):

- KV rows of the attention layers can be attached, but the recurrent state after a prefix cannot be rebuilt from them. The decoder
  keeps *state snapshots* in spare rows of every recurrent state (`MultiSequence.Transform(..., spareStateRows)`, `[slots + N, ...]`),
  copied with `CompiledModel.CopyStateRows` (CPU: memcpy; Vulkan: `vkCmdCopyBuffer` ordered with the steps; a device without it,
  e.g. Metal for now, falls back to no prefix caching for such models). A snapshot is keyed by the hash of the block it ends with;
  blocks are attached only up to the last block with a snapshot, which is restored into the slot's row.
- Where snapshots are taken (bounded memory instead of one per block): at the end of the prefix a new prompt shares with a running
  prompt or a slot's context (so the second prompt with a system prompt leaves the snapshot the third and later ones restore, and
  concurrent prompts waiting for another prompt's blocks find it there), and one block before the last full block of every prompt (a
  later turn of the same conversation restores the state after most of the previous prompt; one block of slack because chat templates
  render the generation prompt, e.g. an empty thinking block, differently once the reply is part of the history), and every
  `EngineOptions.StateSnapshotInterval` (512) prompt tokens, like vLLM's block-aligned Mamba states (prefill chunks end there; the
  snapshots of prompts no later prompt shares are the least recently used and go first). The prefill chunk is cut at such a boundary and the state is
  copied after the step (one extra pass for the prompt end; the copy is the size of one slot's state).
- Capacity: `ChatModelOptions.KvStateSnapshots` / `stateSnapshots` / `sharplight serve --kv-snapshots` (default
  `LlmModel.DefaultStateSnapshots`: as many as fit in half the KV pool's memory or 512 MiB, whichever is more, at least 2 and
  at most two per slot), each the size of one slot's recurrent state (`LlmModel.RecurrentStateBytesPerSlot`, counted in
  `EstimateFootprint`); least recently used first out, and dropped together with their block when the pool evicts it.
- vLLM 0.33 (`mamba_cache_mode="align"`) caches Mamba / GDN state only at multiples of its Mamba block size (aligned to the attention
  page: 544-576 tokens for Qwen3.5-0.8B) at the end of scheduler steps, plus optionally at a shared-prefix junction seen with EAGLE / MTP, and does
  not cache attention-free models (pure Mamba / RWKV). Here the granularity is 16 tokens, junctions are found for every prompt, and pure
  recurrent models are cached the same way. Measured (benchmarks.md section 48.3): higher hit rates than vLLM on shared system prompts and
  conversations, but each snapshot boundary costs the prompt a pass, which for a small model (cheap prefill) raises TTFT above the
  cache-off TTFT; the lever is recomputing instead of waiting where prefill is cheaper than a pass.

API and hosting:

- OpenAI chat / completions and Anthropic messages: the partition is the request context's scope (`SharplightRequestContext.Scope`), the
  salt the context's `CacheSalt`. The request field `cache_salt` (vLLM's name) is accepted only with
  `SharplightServerOptions.AllowClientCacheSalt` (`sharplight serve --allow-cache-salt`); it is appended to the host's salt, so it can
  narrow sharing but never widen it. Without the option a request carrying it is refused with 400 (`param: cache_salt`), so a client
  never believes it is isolated when it is not. Extensions.AI: `ServingChatClientOptions.CacheSalt` and the per-call property
  `sharplight.cache_salt`.
- Metrics, summed over all namespaces (no namespace or tenant labels): `sharplight_kv_prefix_hit_tokens_total`,
  `sharplight_kv_prefix_shared_blocks_total`, `sharplight_kv_blocks_cached`, `sharplight_kv_block_evictions_total`,
  `sharplight_kv_state_snapshots_total`, besides `sharplight_cached_prompt_tokens_total` and the lookups / hits counters
  (`EngineMetricsSnapshot.SharedPrefixTokens`, `SharedPrefixBlocks`, `KvBlocksCached`, `KvBlocksEvicted`, `StateSnapshots`).

What a host such as Sharplight Server passes (one request = one namespace):

- The namespace: `SharplightRequestContext.Scope` = the space (tenant) id, from authentication, never from the request body; optionally
  `SharplightRequestContext.CacheSalt` for a finer split (per user, per data classification). Keep `AllowClientCacheSalt` off unless a
  deployment wants client-chosen salts.
- KV caching on: nothing to set (paged + prefix caching is the default); size the pool with `ChatModelOptions.KvCacheTokens` when the
  default (`slots × ctx`) is not what the replica should hold, and keep `EngineOptions.PrefixCache` on.
- Routing affinity: a prefix cache is per engine (per replica and model), so a router should send requests that share a prefix to
  the same replica: key = (model, scope, salt, a hash of the start of the rendered prompt, e.g. the system prompt or the first 1-2 KB of
  the conversation), consistent hashing over the replicas, with a load bound. A conversation id works as well for multi-turn chats.
  The key must include the scope so that one tenant's prefixes never steer another tenant's routing.

Tests: `PrefixCacheTests` (Serving: blocks outlive their slot, namespaces of partition and salt, content keys, bypass, concurrent prompts
waiting for each other's blocks instead of computing the prefix twice, preempted contexts reattached, tail-first eviction, published
blocks never rewritten, recurrent models restoring snapshots for shared prefixes and the next turn of a conversation, none without
snapshots), `CachePartitionTests`, `MultiTenantApiTests.AClientCacheSaltIsRefusedUnlessTheServerAllowsIt`,
`ServingChatClientTests.TheCacheSaltNarrowsSharingWithinAPartition`, and `PrefixCacheParityTests` (Families: greedy output with the cache
on equals the cache off for shared prefixes, sequential and concurrent, on CPU and Vulkan, for attention-only, gated delta net, Mamba,
Mamba-2, short-convolution and RWKV-7 tiny models and for Qwen3.5-0.8B and Qwen3-0.6B).

## 3. Steps

1. Done: quantised KV in the contiguous layout (independent of paging): Q8_0 and Q4_0 `CacheWrite` + `CachedAttention` for
   CPU and Vulkan, `LlmLoadOptions.KvCacheType` / `--kv-type`, footprint estimate from the state types.
2. Done: paged ops (`kv_block_table` input, paged `CacheWrite` / `CachedAttention` on CPU and Vulkan, all attention kernels).
3. Done: `KvBlockPool` + dynamic allocation + admission by blocks + preemption by recompute; `IBatchDecoder` gains
   `KvBlocksFree` / `KvBlockSize` and per-sequence block management (`Truncate`, `Reset` release blocks); `--kv-tokens`.
4. Done: prefix sharing between slots, first by comparing slot contexts, then by block content hashes with an LRU of cached blocks,
   cache namespaces and recurrent-state snapshots (2.6); paged KV and prefix caching on by default.
   Open: `n > 1` sharing with copy-on-write.
5. Optional: host swap.

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
