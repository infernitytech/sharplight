using System.Globalization;

namespace Sharplight.Shaders;

/// <summary>
/// Every <c>SHARPLIGHT_*</c> environment switch read by the Shaders project, read once per process.
/// The switches are diagnostics and tuning knobs: the defaults are the shipped behavior, and each "disable" switch
/// selects a kernel that is also the portable fallback on devices lacking the feature.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_DUMP_KERNELS=&lt;dir&gt;</c>: write the source and macros of every compiled kernel to the directory.</item>
/// <item><c>SHARPLIGHT_COMPILE_THREADS=n</c>: number of kernels compiled with Slang at the same time (default: half the processor
/// count, between 2 and 8; more threads make the weight upload that overlaps the compilation slower).</item>
/// <item><c>SHARPLIGHT_NO_DUAL=1</c>: do not fuse gate/up matmul pairs into one dual matvec.</item>
/// <item><c>SHARPLIGHT_NO_TOPK_REGISTER_ROW=1</c>: the router top-k keeps per-lane sorted lists instead of the row in registers.</item>
/// <item><c>SHARPLIGHT_NO_GROUPED_KEY_HEADS=1</c>: materialize the in-place head repeats (Tile) of a delta rule's q / k instead of
/// reading key head h / repeats in the kernel.</item>
/// <item><c>SHARPLIGHT_NO_NORM_Q8_COPY=1</c>: int-dot matvecs quantize a row norm's output with their own launch instead of reading
/// q8_1 rows the norm wrote (decode steps).</item>
/// <item><c>SHARPLIGHT_NO_INT_DOT_EPILOGUE=1</c>: run the elementwise group consuming a matmul of int-dot weights (activation, gates)
/// as its own launch instead of in the int-dot matvec's store.</item>
/// <item><c>SHARPLIGHT_GEMM_SG=0</c>: devices with only 8x8 subgroup matrices (Metal) run prefill matmuls on the SIMT GEMM instead of
/// <c>gemm_sg</c>.</item>
/// <item><c>SHARPLIGHT_NO_BIAS_FOLD=1</c>: do not add matmul biases in the single-row matvec's store.</item>
/// <item><c>SHARPLIGHT_NORM_WIDE=0</c>: fused row norms of 4096 values and more keep 256-thread workgroups at decode-size steps on
/// CUDA instead of 1024; <c>SHARPLIGHT_NORM_THREADS=n</c>: threads per workgroup of every fused row norm (experiments).</item>
/// <item><c>SHARPLIGHT_NO_CONV_EPILOGUE=1</c>: run the elementwise group consuming a convolution (activation, residual add) as its
/// own launch instead of in the convolution's store.</item>
/// <item><c>SHARPLIGHT_NO_MATMUL_EPILOGUE=1</c>: run the elementwise group consuming a weight matmul (bias, activation, residual
/// add) as its own launch instead of in the split-f16 GEMM's store.</item>
/// <item><c>SHARPLIGHT_NO_NARROW_CONV=1</c>: run convolutions with up to 8 output channels through the implicit GEMMs instead of
/// the per-pixel conv_narrow kernel.</item>
/// <item><c>SHARPLIGHT_NO_STRIDED_LAYERNORM=1</c>: copy a channel-strided LayerNorm input (the channels-last view of an NCHW
/// tensor) to a dense temporary instead of normalizing it in place.</item>
/// <item><c>SHARPLIGHT_NO_FUSED_CONCAT=1</c>: copy each input of a concatenation with its own launch instead of one launch over
/// the output.</item>
/// <item><c>SHARPLIGHT_NO_GROUP_MERGE=1</c>: fuse only consecutive elementwise nodes (no merging of groups along data flow).</item>
/// <item><c>SHARPLIGHT_NO_GATHER_FOLD=1</c>: materialize row gathers (GatherAxis on axis 0, e.g. the per-expert biases of the
/// selected experts) with their own launch instead of reading the rows in place inside the fused elementwise group consuming them.</item>
/// <item><c>SHARPLIGHT_NO_SCHEDULE=1</c>: keep graph order instead of grouping launches by dependency level.</item>
/// <item><c>SHARPLIGHT_NO_REPACK=1</c>: keep Q8_0 and MXFP4 weights in the file layout (no aligned-quants repack).</item>
/// <item><c>SHARPLIGHT_REPACK_MXFP4=0</c>: keep MXFP4 expert tensors in the file layout (no 16-row-group split into aligned codes
/// and scales, <c>WeightRepack.Mxfp4Split</c>).</item>
/// <item><c>SHARPLIGHT_MMVQ_Q8_DECODE=0|1</c>: serve single-row Q8_0 matvecs with the integer-dot matvec too (default off: the
/// f32 matvec is already at the bandwidth roof there and exact in the activations).</item>
/// <item><c>SHARPLIGHT_ROW_INVARIANT=n</c>: rows up to which a step computes every row like a single-row step, overriding
/// <c>CompileOptions.RowInvariantRows</c> (speculative decoding sets it to the verification rows; 0 turns it off).</item>
/// <item><c>SHARPLIGHT_MMVQ_BLOCK=0</c>: keep the 32-value block formats (Q8_0 as stored, Q4_0, Q5_0, IQ4_NL, MXFP4, and Q4Z / Q8Z / NVFP4 / NVFP4_GS) on the f32
/// matvecs instead of the integer-dot matvec.</item>
/// <item><c>SHARPLIGHT_MMVQ_MAX_ROWS=n</c>: largest activation row count served by the integer-dot matvec (default 8; 16 for
/// Q8_0 on devices without the f16 tensor-core path).</item>
/// <item><c>SHARPLIGHT_MMVQ_LAYOUT=rows,subgroups</c>: force the integer-dot matvec's weight rows and subgroups per workgroup.</item>
/// <item><c>SHARPLIGHT_MMQ=0|all|fmt,fmt</c>: weight formats whose prefill matmuls use the int8 tensor-core GEMM (q8_1
/// activations, <c>mmq.slang</c>) instead of the f16 one: none, every supported format (q8_0, q4_k, q5_k, q6_k, iq4_nl, iq4_xs),
/// or a list (default: iq4_nl, iq4_xs, and q8_0 for matmuls with N and K of at least 4096; an explicit list drops that size
/// limit). Disabling <c>matrix_int8</c> turns it off as well.</item>
/// <item><c>SHARPLIGHT_MMQ_MIN_ROWS=n</c>: activation rows from which the int8 GEMM serves a matmul (default 33).</item>
/// <item><c>SHARPLIGHT_MMQ_CFG</c>, <c>SHARPLIGHT_MMQ_CFG64=wm,wn,fm,fn</c>: force the layout of the 128- and 64-token int8 GEMM
/// tiles. <c>SHARPLIGHT_MMQ_GENERIC=1</c>: scale its accumulators through the per-element (layout independent) path.
/// <c>SHARPLIGHT_MMQ_BK=32|64|128</c>: preferred K step of the int8 GEMM (default 128, 64 for Q6_K).</item>
/// <item><c>SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS=n</c>: rows from which matmuls run on the vendor GEMM library (DeviceFeatures.LibraryGemm;
/// default 256).</item>
/// <item><c>SHARPLIGHT_FP8_GEMM=0</c>: run prefill matmuls with block-scaled E4M3 weights on the f16 tensor-core GEMM instead of the
/// fp8 one (<c>gemm_fp8.slang</c>, E4M3 activations). Disabling <c>matrix_fp8</c> turns it off as well.
/// <c>SHARPLIGHT_FP8_GEMM_MIN_ROWS=n</c>: activation rows from which the fp8 GEMM serves a matmul (default 17).
/// <c>SHARPLIGHT_FP8_DUAL=0</c>: run gate/up style matmul pairs as two fp8 GEMMs plus the elementwise launch instead of one
/// dual fp8 GEMM with the elementwise group as its epilogue. <c>SHARPLIGHT_FP8_DUAL_MAX_ROWS=n</c>: largest batch it serves
/// (default: every batch for which the separate GEMMs would not split K, and 17-64 rows).</item>
/// <item><c>SHARPLIGHT_NATIVE_GEMM_MIN_ROWS=n</c>: activation rows from which the hand-written provider GEMMs serve a matmul
/// (DeviceFeatures.NativeGemmFp8; default 256).</item>
/// <item><c>SHARPLIGHT_GEMM2=0</c>: use the older gemm_cm tensor-core GEMM instead of gemm_cm2.</item>
/// <item><c>SHARPLIGHT_GEMM_WIDE=0</c>: keep large SIMT (f32) matmuls on the 64 x 64 gemm tile instead of the 128 x 128 gemm_wide;
/// <c>=force</c> also uses gemm_wide on devices whose workgroup memory it would fill.</item>
/// <item><c>SHARPLIGHT_GEMM2_CFG=wm,wn,fm,fn</c>: force one gemm_cm2 warp/fragment layout (must give 128x128 tiles).</item>
/// <item><c>SHARPLIGHT_GEMM2_SKINNY=0</c>: always use 128-token gemm_cm2 tiles (no 32/64-token tiles for small batches).</item>
/// <item><c>SHARPLIGHT_GEMM2_CFG32</c>, <c>SHARPLIGHT_GEMM2_CFG64=wm,wn,fm,fn</c>: force the layout of the 32- and 64-token
/// gemm_cm2 tiles.</item>
/// <item><c>SHARPLIGHT_SKINNY_GEMM=0</c>: serve small-batch matmuls (2-16 rows for formats without an integer-dot matvec such
/// as f16, bf16, FP8, NVFP4; 9-16 rows for Q8_0 and the K-quants) with the multi-row f32 matvec, the integer-dot matvec and
/// gemm_cm2 (split-K) instead of the skinny tensor-core GEMM (<c>matmul_skinny.slang</c>: every weight read once, K split
/// inside the workgroup; matrices with fewer than 1024 output rows, or 4096 above 8 rows, stay on the split-K kernels).
/// <c>SHARPLIGHT_SKINNY_MAX_ROWS=n</c>: largest row count it serves at every width (default 16 for N of at least 4096, at most
/// 64). <c>SHARPLIGHT_SKINNY_CFG=fn,sg,bk</c>: force its weight fragments per workgroup, subgroups per workgroup and K block.
/// <c>SHARPLIGHT_SKINNY_X16=0</c>: convert f32 activation rows while staging instead of reading the shared f16 copy.
/// <c>SHARPLIGHT_SKINNY_DIRECT=0</c>: always stage its operands in shared memory (no direct fragment loads).</item>
/// <item><c>SHARPLIGHT_GEMM_WAVE=n</c>: tensor-core GEMM tiles assumed to run at once when choosing K splits (default: the
/// device's compute units, 192 when unknown); <c>fixed</c> restores the earlier 192-tile splits.</item>
/// <item><c>SHARPLIGHT_ATTN_FE_CAUSAL=0</c>: causal single-sequence prefill attention on the general flash kernel <c>fa</c>
/// instead of <c>fe</c>.</item>
/// <item><c>SHARPLIGHT_MV_ROWS=n</c>: output rows per workgroup of the single-row split-K matvec (default 2 for single matrices
/// with at least 1024 rows, else 1; 1 restores one row per workgroup).</item>
/// <item><c>SHARPLIGHT_FP8_DUAL_TALL=0</c>: the dual fp8 gate / up GEMM keeps 64-row tiles above 64 rows.</item>
/// <item><c>SHARPLIGHT_NORM_HALF_COPY=0</c>: row norms and dual GEMMs feeding matmuls do not also write the f16 activation copy the
/// tensor-core GEMMs read; a separate conversion launch makes it.</item>
/// <item><c>SHARPLIGHT_F16_RANGE=0</c>: the f16 activation copies of values no norm bounds are plain f16 rows instead of row-ranged
/// ones (<c>f16_range_scale</c> in common.slang: rows reaching 2^15 scaled by a power of two that the GEMMs undo), so values beyond
/// 65504 become infinities, as before the guard (for A/B measurements).</item>
/// <item><c>SHARPLIGHT_GEMM_BK=32|64</c>: preferred K block of gemm_cm2 (64 when K allows it).</item>
/// <item><c>SHARPLIGHT_GEMM_X16=0</c>: tensor-core GEMMs read f32 activations instead of an f16 copy.</item>
/// <item><c>SHARPLIGHT_GEMM_NV2=1</c>: default of <see cref="Lowering.GraphLowerer.PreferWorkgroupMatrixGemm"/>.</item>
/// <item><c>SHARPLIGHT_GEMM_NV2_TILE=bt,bn,bk</c>: tile of the workgroup-matrix GEMM (default 128,128,32).</item>
/// <item><c>SHARPLIGHT_ATTN_FA=0</c>, <c>SHARPLIGHT_ATTN_SPLIT=0</c>, <c>SHARPLIGHT_ATTN_SEQ_FA=0</c>: disable tensor-core
/// flash attention, split-key decode attention, and flash attention for multi-sequence batches.</item>
/// <item><c>SHARPLIGHT_ATTN_DECODE=0</c>: disable split-K flash decoding (GQA-grouped decode attention with a run-time split
/// count); decode then uses the per-head split kernel. <c>SHARPLIGHT_ATTN_DECODE_CFG=wg,keys,splits[,rows]</c>: workgroups the split
/// count aims for, minimum keys per split, maximum splits (default 1024,128,64), and optionally key rows per lane group and batch.</item>
/// <item><c>SHARPLIGHT_ATTN_DECODE_FUSED=0</c>: merge the partials of split-K decoding in a separate <c>decode_combine</c> launch instead of in the last workgroup of each row.</item>
/// <item><c>SHARPLIGHT_ATTN_MQA=0</c>: disable the tensor-core attention for many query heads on one key row whose values are a
/// row prefix (absorbed multi-head latent attention, <c>fa_mqa</c>); prefill and decode then use the generic kernels.</item>
/// <item><c>SHARPLIGHT_DELTA_LPR</c>, <c>SHARPLIGHT_DELTA_DVB</c>, <c>SHARPLIGHT_SSM_LPR</c>: lanes per row and value-row
/// block of the delta-rule and selective-scan kernels.</item>
/// <item><c>SHARPLIGHT_DELTA_CHUNK=0</c>: run the gated delta rule token by token for every batch (no chunked tensor-core form,
/// <c>chunked.slang</c>). <c>SHARPLIGHT_DELTA_CHUNK_MIN=n</c>: rows of a sequence from which the chunked form serves it (default
/// 64). <c>SHARPLIGHT_DELTA_CHUNK_BV=16|32|64</c>: state columns per workgroup of its sequential pass,
/// <c>SHARPLIGHT_DELTA_CHUNK_THREADS=128|256</c> its threads per workgroup.
/// <c>SHARPLIGHT_DELTA_CHUNK_PARTS=1|2</c>: force one f16 product (reduced precision) or the two-part f16 split (near f32).</item>
/// <item><c>SHARPLIGHT_CONV1D_PARALLEL=0</c>: run the causal conv1d of prefill batches token by token per channel (no
/// token-parallel <c>conv1d_tokens</c> + <c>conv1d_state</c>). <c>SHARPLIGHT_CONV1D_TOKENS=n</c>: tokens per thread of the
/// token-parallel form (default 8).</item>
/// <item><c>SHARPLIGHT_MOE_MV_MAX</c>, <c>SHARPLIGHT_MOE_MMVQ_MAX</c>, <c>SHARPLIGHT_MOE_TILE</c>, <c>SHARPLIGHT_MOE_SCALAR_ROWS</c>,
/// <c>SHARPLIGHT_MOE_NO_COOP=1</c>: expert-matmul tuning (see <c>GraphLowerer.Moe.cs</c>). <c>SHARPLIGHT_MOE_GEMM2=0</c>: run
/// grouped expert matmuls on moe_gemm_cm instead of the expert-grouped gemm_cm2.</item>
/// <item><c>SHARPLIGHT_MOE_X16=0</c>: the expert-grouped gemm_cm2 gathers and converts the f32 activation rows itself (in every
/// column tile) instead of reading the expert-sorted f16 rows written once per routing by <c>moe_gather_x16</c>.</item>
/// <item><c>SHARPLIGHT_MOE_SKINNY=n</c>: average items per expert up to which the expert-grouped skinny GEMM serves the grouped
/// expert products (0: gemm_cm2 at every size).</item>
/// <item><c>SHARPLIGHT_MOE_WIDE=fm,fn,sg,ks,n</c>: wide expert tiles of the direct skinny GEMM (experiment, off by default).</item>
/// <item><c>SHARPLIGHT_MOE_BIAS_EPILOGUE=0</c>: add the routed experts' bias rows to a grouped expert product in a separate
/// launch instead of in the grouped GEMM's store.</item>
/// <item><c>SHARPLIGHT_MOE_DUAL=n</c>: gate / up expert pairs on one dual launch storing the elementwise group (bit 0 grouped
/// gemm_cm2, bit 1 expert-grouped skinny GEMM; default 3, 0: two launches and the group's).</item>
/// <item><c>SHARPLIGHT_SPLITK_EPILOGUE=0</c>: run a matmul's bias / residual group as its own launch instead of in the split-K
/// reduction of the f16 tensor-core GEMM.</item>
/// <item><c>SHARPLIGHT_FP8_SHORT_TILES=1</c>: the fp8 GEMM takes 64-row tiles at 129-192 and 257-320 rows (opt-in).</item>
/// <item><c>SHARPLIGHT_MMQ_SHORT_MAX_ROWS=n</c>: the int8 / fp8 GEMMs take 64-row tiles exactly up to n rows.</item>
/// <item><c>SHARPLIGHT_CONV_NO_MATRIX=1</c>: run convolutions on the SIMT conv_gemm instead of the cooperative-matrix conv_cm.</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>Directory receiving every compiled kernel's source, or null.</summary>
    /// <summary>False (<c>SHARPLIGHT_GEMM_SG=0</c>) when devices with 8x8 subgroup matrices keep the SIMT GEMM for prefill.</summary>
    public static bool SmallMatrixGemm { get; } = Read("SHARPLIGHT_GEMM_SG") != "0";

    public static string? DumpKernelsDirectory { get; } = Read("SHARPLIGHT_DUMP_KERNELS");

    /// <summary>Degree of parallelism of kernel compilation.</summary>
    public static int CompileThreads { get; } = int.Parse(Read("SHARPLIGHT_COMPILE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 2, 8).ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);

    /// <summary>
    /// Disables the expert slot-sum fusion (the down projection of a MoE block, its elementwise group and the sum over the
    /// slots in one decode launch).
    /// </summary>
    public static bool NoExpertSum { get; } = IsSet("SHARPLIGHT_NO_EXPERT_SUM");

    /// <summary>Keeps the router's softmax, top-k and weight normalization in separate launches (<c>SHARPLIGHT_NO_ROUTER_FUSION=1</c>).</summary>
    public static bool NoRouterFusion { get; } = IsSet("SHARPLIGHT_NO_ROUTER_FUSION");

    /// <summary>The router top-k keeps per-lane sorted lists instead of its row in registers (<c>SHARPLIGHT_NO_TOPK_REGISTER_ROW=1</c>).</summary>
    public static bool NoTopKRegisterRow { get; } = IsSet("SHARPLIGHT_NO_TOPK_REGISTER_ROW");

    /// <summary>
    /// Leaves the biases of int-dot q / k / v projections to their own launch after the multi-output GEMM instead of its store
    /// (<c>SHARPLIGHT_NO_INT_DOT_MULTI_BIAS=1</c>).
    /// </summary>
    public static bool NoIntDotMultiBias { get; } = IsSet("SHARPLIGHT_NO_INT_DOT_MULTI_BIAS");

    /// <summary>
    /// Tiles ahead each warp of the native decode attention prefetches its key and value rows into L2 (<c>SHARPLIGHT_ATTN_DECODE_PF=n</c>,
    /// 0 turns it off; unset: the kernel's default of one tile).
    /// </summary>
    public static int? AttentionDecodePrefetch { get; } = int.TryParse(Read("SHARPLIGHT_ATTN_DECODE_PF"), out int n) ? n : null;

    /// <summary>Lowers a layer's query rope, key rope + cache write and value cache write as separate launches (<c>SHARPLIGHT_NO_ROPE_QKV=1</c>).</summary>
    public static bool NoRopeQkv { get; } = IsSet("SHARPLIGHT_NO_ROPE_QKV");

    /// <summary>
    /// Rows per warp of <c>rope_qkv</c>'s warp form (<c>SHARPLIGHT_ROPE_QKV_WARP=n</c>; 0 keeps the 64-thread workgroups per row; unset:
    /// the lowering's default).
    /// </summary>
    public static int? RopeQkvWarpRows { get; } = Read("SHARPLIGHT_ROPE_QKV_WARP") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Names the attention layers whose ropes and cache writes stay separate launches, and why (<c>SHARPLIGHT_ROPE_QKV_DEBUG=1</c>).</summary>
    public static bool RopeQkvDebug { get; } = Read("SHARPLIGHT_ROPE_QKV_DEBUG") == "1";

    /// <summary>Materializes the in-place head repeats of a delta rule's q / k (<c>SHARPLIGHT_NO_GROUPED_KEY_HEADS=1</c>).</summary>
    public static bool NoGroupedKeyHeads { get; } = IsSet("SHARPLIGHT_NO_GROUPED_KEY_HEADS");

    /// <summary>Row norms do not write the q8_1 rows of the int-dot matvecs reading them (<c>SHARPLIGHT_NO_NORM_Q8_COPY=1</c>).</summary>
    public static bool NoNormQ8Copy { get; } = IsSet("SHARPLIGHT_NO_NORM_Q8_COPY");

    /// <summary>Elementwise groups after a matmul of int-dot weights run as their own launch (<c>SHARPLIGHT_NO_INT_DOT_EPILOGUE=1</c>).</summary>
    public static bool NoIntDotEpilogue { get; } = IsSet("SHARPLIGHT_NO_INT_DOT_EPILOGUE");

    /// <summary>Disables the dual (gate/up) matvec fusion.</summary>
    public static bool NoDualMatMul { get; } = IsSet("SHARPLIGHT_NO_DUAL");

    /// <summary>Disables adding matmul biases in the single-row matvec (separate Add launches instead).</summary>
    public static bool NoBiasFold { get; } = IsSet("SHARPLIGHT_NO_BIAS_FOLD");

    /// <summary>Disables the convolution epilogue fusion (elementwise consumers of a conv computed in its store).</summary>
    public static bool NoConvEpilogue { get; } = IsSet("SHARPLIGHT_NO_CONV_EPILOGUE");

    /// <summary>Runs convolutions with few output channels through the implicit GEMMs instead of conv_narrow.</summary>
    public static bool NoNarrowConv { get; } = IsSet("SHARPLIGHT_NO_NARROW_CONV");

    /// <summary>Concatenations run as one launch over the output (off with <c>SHARPLIGHT_NO_FUSED_CONCAT=1</c>: one copy per input).</summary>
    public static bool FusedConcat { get; } = !IsSet("SHARPLIGHT_NO_FUSED_CONCAT");

    /// <summary>
    /// Inputs of a concatenation whose leading axes have extent 1 are computed in their slices of the output, without a copy
    /// (off with <c>SHARPLIGHT_NO_CONCAT_IN_PLACE=1</c>).
    /// </summary>
    public static bool ConcatInPlace { get; } = !IsSet("SHARPLIGHT_NO_CONCAT_IN_PLACE");

    /// <summary>Densifies channel-strided LayerNorm inputs for the row kernel instead of reading them in place (layernorm_cs).</summary>
    public static bool NoStridedLayerNorm { get; } = IsSet("SHARPLIGHT_NO_STRIDED_LAYERNORM");

    /// <summary>Disables the matmul epilogue fusion (elementwise consumers of a weight matmul computed in the split-f16 GEMM's store).</summary>
    public static bool NoMatMulEpilogue { get; } = IsSet("SHARPLIGHT_NO_MATMUL_EPILOGUE");

    /// <summary>False keeps the 32-value block formats off the integer-dot matvec.</summary>
    public static bool IntDotBlockFormats { get; } = IsNotZero("SHARPLIGHT_MMVQ_BLOCK");

    /// <summary>False keeps the grid formats (IQ2 / IQ3) of routed experts off the integer-dot expert matvec (<c>SHARPLIGHT_MMVQ_GRID=0</c>).</summary>
    public static bool IntDotGridFormats { get; } = IsNotZero("SHARPLIGHT_MMVQ_GRID");

    /// <summary>Disables merging elementwise groups along data flow.</summary>
    public static bool NoGroupMerge { get; } = IsSet("SHARPLIGHT_NO_GROUP_MERGE");

    /// <summary>Disables reading row gathers (e.g. selected experts' biases) in place inside their fused consumer group.</summary>
    public static bool NoGatherFold { get; } = IsSet("SHARPLIGHT_NO_GATHER_FOLD");

    /// <summary>Disables level scheduling of units.</summary>
    public static bool NoSchedule { get; } = IsSet("SHARPLIGHT_NO_SCHEDULE");

    /// <summary>Disables the Q8_0 aligned-quants repack.</summary>
    public static bool NoWeightRepack { get; } = IsSet("SHARPLIGHT_NO_REPACK");

    /// <summary>False keeps MXFP4 expert tensors in the file layout (no aligned 16-row-group repack).</summary>
    public static bool Mxfp4Repack { get; } = IsNotZero("SHARPLIGHT_REPACK_MXFP4");

    /// <summary>Repack block-scaled FP8 weights into the aligned split layout (<c>SHARPLIGHT_REPACK_FP8=0</c> disables).</summary>
    public static bool Fp8Repack { get; } = IsNotZero("SHARPLIGHT_REPACK_FP8");

    /// <summary>
    /// <c>SHARPLIGHT_FP8_GROUPED</c>: 1 stores block-scaled E4M3 matrices in the 128-row groups (<c>WeightRepack.Fp8B128Grouped</c>) on
    /// every target, 0 on none; unset: where the hand-written FP8 GEMM reads them (CUDA).
    /// </summary>
    public static bool? Fp8Grouped { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_FP8_GROUPED") is { } v ? v != "0" : null;

    /// <summary>True when single-row Q8_0 matvecs use the integer-dot matvec.</summary>
    public static bool IntDotDecodeQ8 { get; } = IsSet("SHARPLIGHT_MMVQ_Q8_DECODE");

    /// <summary>Forced row-invariance bound (<c>SHARPLIGHT_ROW_INVARIANT</c>), or null for the compile option.</summary>
    public static long? RowInvariantRows { get; } = Read("SHARPLIGHT_ROW_INVARIANT") is { } r ? long.Parse(r, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced largest activation row count served by the integer-dot matvec, or null for the per-format default.</summary>
    public static long? IntDotMaxRows { get; } = Read("SHARPLIGHT_MMVQ_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced integer-dot matvec layout (weight rows per workgroup, subgroups per workgroup), or null.</summary>
    public static int[]? IntDotLayout { get; } = Read("SHARPLIGHT_MMVQ_LAYOUT")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Weight formats of the int8 tensor-core GEMM ("0", "all" or a comma-separated list), or null for the default.</summary>
    public static string? MmqFormats { get; } = Read("SHARPLIGHT_MMQ");

    /// <summary>Activation rows from which the int8 GEMM serves a matmul.</summary>
    public static long? MmqMinRows { get; } = Read("SHARPLIGHT_MMQ_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Activation rows from which the vendor GEMM library serves a matmul, or null for the default.</summary>
    public static long? LibraryGemmMinRows { get; } = Read("SHARPLIGHT_LIBRARY_GEMM_MIN_ROWS") is { } lv ? long.Parse(lv, CultureInfo.InvariantCulture) : null;

    /// <summary>Activation rows from which the fp8 GEMM serves a matmul, or null for the default.</summary>
    public static long? Fp8GemmMinRows { get; } = Read("SHARPLIGHT_FP8_GEMM_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced layout of the 128-token int8 GEMM tile (warps M, warps N, fragments M, fragments N), or null.</summary>
    public static int[]? MmqLayout { get; } = Read("SHARPLIGHT_MMQ_CFG")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Forced layout of the 64-token int8 GEMM tile, or null.</summary>
    public static int[]? MmqLayout64 { get; } = Read("SHARPLIGHT_MMQ_CFG64")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Forced preferred K step of the int8 GEMM, or null for the per-format default.</summary>
    public static int? MmqBlockK { get; } = Read("SHARPLIGHT_MMQ_BK") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>True when the int8 GEMM always scales its accumulators per element.</summary>
    public static bool MmqGenericLayout { get; } = IsSet("SHARPLIGHT_MMQ_GENERIC");

    /// <summary>Activation rows from which the hand-written provider GEMMs serve a matmul, or null for the default.</summary>
    public static long? NativeGemmMinRows { get; } = Read("SHARPLIGHT_NATIVE_GEMM_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// Average routed rows per expert from which the hand-written grouped MXFP4 expert GEMM serves an expert product
    /// (<c>SHARPLIGHT_NATIVE_MOE_MIN_ROWS</c>), or null for the default.
    /// </summary>
    public static long? NativeMoeMinRows { get; } = Read("SHARPLIGHT_NATIVE_MOE_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_FP8_TMA=0</c> keeps the 128-row tiles of the hand-written FP8 GEMM on its per-thread-copy form
    /// (<c>gemm_fp8_native.cu</c>) where the device has <c>tensor_copies</c> (default: the warp-specialized <c>gemm_fp8_tma.cu</c>).
    /// </summary>
    public static bool NativeFp8Tma { get; } = IsNotZero("SHARPLIGHT_FP8_TMA");

    /// <summary>
    /// <c>SHARPLIGHT_FP8_WG=0</c> runs <c>gemm_fp8_tma.cu</c> in its 9-warp form (one copy warp, 168 registers per computing thread)
    /// instead of the copy warpgroup that hands its registers to the computing warps (<c>F8N_WG</c>, the default where the device has
    /// <c>register_reallocation</c>). Off by default in benchmarks.md section 92, when the copy warpgroup's multi-output launch changed
    /// from run to run with NVRTC's machine code; the cause was the missing proxy fence between the computing warps' reads of a stage
    /// and the next tensor copy into it (section 93).
    /// </summary>
    public static bool Fp8CopyWarpgroup { get; } = IsNotZero("SHARPLIGHT_FP8_WG");

    /// <summary>
    /// <c>SHARPLIGHT_FP8_UNIFORM_SCALES=0</c>: <c>gemm_fp8_tma.cu</c> promotes every K block with per-row weight scales even for weights
    /// whose 128-row groups share each block's scale (default: one weight scale per group, <c>F8N_WUNI</c>; the same results).
    /// </summary>
    public static bool Fp8UniformScales { get; } = IsNotZero("SHARPLIGHT_FP8_UNIFORM_SCALES");

    /// <summary>
    /// <c>SHARPLIGHT_NATIVE_ATTN</c>: which causal prefill attentions run on the hand-written flash-attention kernel
    /// (<c>attn_prefill_native.cu</c>) where the device has <c>native_attention</c>: <c>0</c> none, <c>1</c> (default) all it serves,
    /// <c>2</c> only those the generated <c>fe</c> kernel does not serve (windows, chunks, soft caps, sinks).
    /// </summary>
    public static int NativeAttention { get; } = Read("SHARPLIGHT_NATIVE_ATTN") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : 1;

    /// <summary><c>SHARPLIGHT_NATIVE_DECODE=0</c>: split-K decode attention stays on the generated <c>decode</c> kernel instead of <c>attn_decode_native.cu</c>.</summary>
    public static bool NativeDecodeAttention { get; } = IsNotZero("SHARPLIGHT_NATIVE_DECODE");

    /// <summary>
    /// True (default) when decode rows of head sizes 256 and 512 run the wide form of the hand-written decode attention
    /// (<c>SHARPLIGHT_NATIVE_DECODE_WIDE=0</c> keeps the generated <c>decode</c> for them).
    /// </summary>
    public static bool NativeWideDecodeAttention { get; } = IsNotZero("SHARPLIGHT_NATIVE_DECODE_WIDE");

    /// <summary>
    /// <c>SHARPLIGHT_FP8_TALL_COST</c>: time of a 128-row block of the hand-written FP8 GEMM relative to a 64-row one in its tile
    /// choice, or null for the default.
    /// </summary>
    public static double? NativeFp8TallCost { get; } = Read("SHARPLIGHT_FP8_TALL_COST") is { } v ? double.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>False keeps E4M3-weight prefill matmuls off the fp8 tensor-core GEMM.</summary>
    public static bool Fp8Gemm { get; } = IsNotZero("SHARPLIGHT_FP8_GEMM");

    /// <summary>False runs fused matmul pairs (gate/up) as two fp8 GEMMs and an elementwise launch instead of one dual fp8 GEMM.</summary>
    public static bool Fp8DualGemm { get; } = IsNotZero("SHARPLIGHT_FP8_DUAL");

    /// <summary>Largest batch above 64 rows the dual fp8 GEMM serves, or null for no limit.</summary>
    public static long? Fp8DualGemmMaxRows { get; } = Read("SHARPLIGHT_FP8_DUAL_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>False keeps large SIMT matmuls on the 64 x 64 gemm tile (no gemm_wide).</summary>
    public static bool WideSimtGemm { get; } = IsNotZero("SHARPLIGHT_GEMM_WIDE");

    /// <summary>True (<c>SHARPLIGHT_GEMM_WIDE=force</c>) uses gemm_wide also where its 32 KB fill the device's workgroup memory.</summary>
    public static bool ForceWideSimtGemm { get; } = Read("SHARPLIGHT_GEMM_WIDE") == "force";

    /// <summary>False selects gemm_cm instead of gemm_cm2.</summary>
    public static bool Gemm2 { get; } = IsNotZero("SHARPLIGHT_GEMM2");

    /// <summary>Forced gemm_cm2 layout (warps M, warps N, fragments M, fragments N), or null.</summary>
    public static int[]? Gemm2Layout { get; } = Read("SHARPLIGHT_GEMM2_CFG")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>False keeps gemm_cm2 at 128-token tiles for every batch.</summary>
    public static bool Gemm2Skinny { get; } = IsNotZero("SHARPLIGHT_GEMM2_SKINNY");

    /// <summary>Forced layout of the 32-token gemm_cm2 tile, or null.</summary>
    public static int[]? Gemm2Layout32 { get; } = Read("SHARPLIGHT_GEMM2_CFG32")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Forced layout of the 64-token gemm_cm2 tile, or null.</summary>
    public static int[]? Gemm2Layout64 { get; } = Read("SHARPLIGHT_GEMM2_CFG64")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>
    /// Tensor-core GEMM tiles assumed to run at once when choosing K splits (<c>SHARPLIGHT_GEMM_WAVE=n</c>; <c>fixed</c> = 192,
    /// the splits before 2026-11-15), or null for the default (the device's compute units).
    /// </summary>
    public static long? GemmWaveTiles { get; } = Read("SHARPLIGHT_GEMM_WAVE") is { } v
        ? v == "fixed" ? 192 : long.Parse(v, CultureInfo.InvariantCulture)
        : null;

    /// <summary>Forced output rows per workgroup of the single-row split-K matvec, or null for the default.</summary>
    public static long? SplitKMatvecRows { get; } = Read("SHARPLIGHT_MV_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>False keeps small-batch matmuls off the skinny tensor-core GEMM.</summary>
    public static bool SkinnyGemm { get; } = IsNotZero("SHARPLIGHT_SKINNY_GEMM");

    /// <summary>
    /// <c>SHARPLIGHT_FP8_L2_PREFETCH=n</c>: K blocks of its weight rows each block of the hand-written FP8 GEMM's decode form asks L2 for
    /// before it waits for the previous launch (0: none), or null for the default.
    /// </summary>
    public static int? Fp8L2Prefetch { get; } = Read("SHARPLIGHT_FP8_L2_PREFETCH") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced largest row count of the skinny GEMM, or null for the default.</summary>
    public static long? SkinnyGemmMaxRows { get; } = Read("SHARPLIGHT_SKINNY_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced skinny GEMM configuration (weight fragments, subgroups, K block), or null.</summary>
    public static int[]? SkinnyGemmConfig { get; } = Read("SHARPLIGHT_SKINNY_CFG")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Let the skinny GEMM load its operands straight into the fragment registers where the device's layout allows.</summary>
    public static bool SkinnyGemmDirect { get; } = IsNotZero("SHARPLIGHT_SKINNY_DIRECT");

    /// <summary>Forced activation source of the skinny GEMM (true: f16 copy, false: f32 rows), or null for the default.</summary>
    public static bool? SkinnyGemmHalfActivations { get; } = Read("SHARPLIGHT_SKINNY_X16") is { } v ? v != "0" : null;

    /// <summary>Let row norms and dual GEMMs feeding tensor-core matmuls write the f16 activation copy too (no separate conversion launch).</summary>
    public static bool NormHalfCopy { get; } = IsNotZero("SHARPLIGHT_NORM_HALF_COPY");

    /// <summary>False (<c>SHARPLIGHT_F16_RANGE=0</c>) when f16 activation copies are plain f16 rows even for values no norm bounds.</summary>
    public static bool F16RangeGuard { get; } = IsNotZero("SHARPLIGHT_F16_RANGE");

    /// <summary>
    /// False keeps row norms at prefill sizes writing f32 rows only (separate f16 conversion / E4M3 quantization launches)
    /// instead of the activations of the tensor-core GEMMs that read them.
    /// </summary>
    public static bool NormPrefillCopy { get; } = IsNotZero("SHARPLIGHT_NORM_PREFILL_COPY");

    /// <summary>Smallest step from which row norms write the GEMMs' activations instead of f32 rows, or null for the default.</summary>
    public static long? NormCopyMinRows { get; } = Read("SHARPLIGHT_NORM_COPY_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Let the dual fp8 gate / up GEMM use 128-row tiles above 64 rows.</summary>
    public static bool Fp8DualGemmTall { get; } = IsNotZero("SHARPLIGHT_FP8_DUAL_TALL");

    /// <summary>Preferred K block of gemm_cm2.</summary>
    public static int GemmBlockK { get; } = int.Parse(Read("SHARPLIGHT_GEMM_BK") ?? "64", CultureInfo.InvariantCulture);

    /// <summary>True when tensor-core GEMMs read an f16 copy of the activations.</summary>
    public static bool GemmHalfActivations { get; } = IsNotZero("SHARPLIGHT_GEMM_X16");

    /// <summary>
    /// <c>SHARPLIGHT_GEMM2_TILE48=0</c>: batches of 33-48 rows on gemm_cm2 take 64-token tiles instead of 48-token tiles (one 4-warp
    /// subgroup row of three 16-row fragments).
    /// </summary>
    public static bool Gemm2Tile48 { get; } = IsNotZero("SHARPLIGHT_GEMM2_TILE48");

    /// <summary>False runs f16 / bf16 matmul pairs (gate / up) at prefill as two gemm_cm2 launches and an elementwise launch.</summary>
    public static bool DualGemm2 { get; } = IsNotZero("SHARPLIGHT_GEMM2_DUAL");

    /// <summary>
    /// Token tiles per group of the grouped tile order of the tensor-core GEMMs (<c>SHARPLIGHT_GEMM_GROUP=n</c>; 0 keeps the
    /// launch order, column tiles first).
    /// </summary>
    public static int GemmTileGroup { get; } = int.Parse(Read("SHARPLIGHT_GEMM_GROUP") ?? "16", CultureInfo.InvariantCulture);

    /// <summary>
    /// Extra macros for the tensor-core GEMM kernels (<c>SHARPLIGHT_GEMM_MACROS=NAME=value,NAME=value</c>), for kernel experiments.
    /// </summary>
    public static IReadOnlyDictionary<string, string> GemmExtraMacros { get; } = (Read("SHARPLIGHT_GEMM_MACROS") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(kv => kv.Split('=')).ToDictionary(kv => kv[0], kv => kv.Length > 1 ? kv[1] : "1");

    /// <summary>Default of <see cref="Lowering.GraphLowerer.PreferWorkgroupMatrixGemm"/>.</summary>
    public static bool WorkgroupMatrixGemm { get; } = IsSet("SHARPLIGHT_GEMM_NV2");

    /// <summary>Workgroup-matrix GEMM tile: token tile, column tile, K tile.</summary>
    public static int[] WorkgroupMatrixTile { get; } = (Read("SHARPLIGHT_GEMM_NV2_TILE") ?? "128,128,32").Split(',').Select(int.Parse).ToArray();

    /// <summary>Extra macros for the <c>fe</c> attention kernel (<c>SHARPLIGHT_FE_MACROS=NAME=value,...</c>), for kernel experiments.</summary>
    public static IReadOnlyDictionary<string, string> AttentionExtraMacros { get; } = (Read("SHARPLIGHT_FE_MACROS") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(kv => kv.Split('=')).ToDictionary(kv => kv[0], kv => kv.Length > 1 ? kv[1] : "1");

    /// <summary>
    /// <c>SHARPLIGHT_NATIVE_DECODE_Q8=0</c>: decode rows of head sizes 256 / 512 over q8_0 caches keep the generated <c>decode</c> instead of
    /// the wide form of <c>attn_decode_native.cu</c> with raw q8_0 tiles (CUDA).
    /// </summary>
    public static bool NativeWideDecodeQ8 { get; } = IsNotZero("SHARPLIGHT_NATIVE_DECODE_Q8");

    /// <summary><c>SHARPLIGHT_COOP_KQ_NT=1|2</c>: 64-row weight tiles per workgroup of the cooperative-matrix K-quant decode GEMM, or null for the default.</summary>
    public static int? CoopKQuantWeightTiles { get; } = Read("SHARPLIGHT_COOP_KQ_NT") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_NORM_HEAD_TAIL=0</c>: a row norm in front of an elementwise group that already ends in a row norm stays its own launch
    /// (instead of one launch for norm, group and norm, e.g. Gemma's post-attention norm, residual add and pre-feed-forward norm).
    /// </summary>
    public static bool NormHeadAndTail { get; } = IsNotZero("SHARPLIGHT_NORM_HEAD_TAIL");

    /// <summary>Extra macros for the cooperative-matrix K-quant decode GEMM (<c>SHARPLIGHT_COOP_KQ_MACROS=NAME=value,...</c>), for kernel experiments.</summary>
    public static IReadOnlyDictionary<string, string> CoopKQuantExtraMacros { get; } = (Read("SHARPLIGHT_COOP_KQ_MACROS") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries).Select(kv => kv.Split('=')).ToDictionary(kv => kv[0], kv => kv.Length > 1 ? kv[1] : "1");

    /// <summary>
    /// Queries per <c>fe</c> workgroup for head sizes above 256 (split head dimensions, <c>SHARPLIGHT_ATTN_FE_WIDE_ROWS</c>: 32 or 64). 64 shares
    /// each 16-key tile between twice the queries (Q staged in two passes); see <c>FeBlockRows</c>.
    /// </summary>
    public static int AttentionSplitHeadRows { get; } = int.Parse(Read("SHARPLIGHT_ATTN_FE_WIDE_ROWS") ?? "64", CultureInfo.InvariantCulture);

    /// <summary>Queries per <c>fe</c> workgroup with the register softmax (<c>SHARPLIGHT_ATTN_FE_ROWS</c>: 64 or 128).</summary>
    public static int AttentionRegisterSoftmaxRows { get; } = int.Parse(Read("SHARPLIGHT_ATTN_FE_ROWS") ?? "128", CultureInfo.InvariantCulture);

    /// <summary>False keeps the staged softmax of the <c>fe</c> attention kernel on devices with the MMA fragment layout.</summary>
    public static bool AttentionRegisterSoftmax { get; } = IsNotZero("SHARPLIGHT_ATTN_FE_REG");

    /// <summary>
    /// False keeps head sizes above 128 off the register-softmax <c>fe</c> (<c>SHARPLIGHT_ATTN_FE_WIDE=0</c>): they then take <c>fa</c>.
    /// </summary>
    public static bool AttentionWideRegisterSoftmax { get; } = IsNotZero("SHARPLIGHT_ATTN_FE_WIDE");

    /// <summary>False keeps causal sliding-window attention on <c>fa</c> instead of the causal <c>fe</c> (<c>SHARPLIGHT_ATTN_FE_WINDOW=0</c>).</summary>
    public static bool AttentionCausalFlashWindow { get; } = IsNotZero("SHARPLIGHT_ATTN_FE_WINDOW");

    /// <summary>True when tensor-core flash attention may be used.</summary>
    public static bool AttentionFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_FA");

    /// <summary>
    /// True (default) when flash attention over f32 K/V may run in the split-f16 form under strict precision (two f16 parts per
    /// value, three tensor-core products); <c>SHARPLIGHT_ATTN_SPLIT_F16=0</c> leaves those to the f32 scalar kernel.
    /// </summary>
    public static bool AttentionSplitHalf { get; } = IsNotZero("SHARPLIGHT_ATTN_SPLIT_F16");

    /// <summary>
    /// True (default) when bidirectional flash attention with few query blocks may split each block's keys over several
    /// workgroups (merged by a combine kernel); <c>SHARPLIGHT_ATTN_FA_KSPLIT=0</c> keeps one workgroup per block.
    /// </summary>
    public static bool AttentionFlashKeySplit { get; } = IsNotZero("SHARPLIGHT_ATTN_FA_KSPLIT");

    /// <summary>
    /// True (default) when plain bidirectional attention (encoders: no spans, window, causal mask, score biases, softcap or
    /// sinks) may use the encoder flash kernel <c>fe</c>; <c>SHARPLIGHT_ATTN_FE=0</c> uses the general flash kernel.
    /// </summary>
    public static bool AttentionEncoderFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_FE");

    /// <summary>
    /// True (default): plain causal prefill attention of one sequence uses <c>fe</c> in its causal form;
    /// <c>SHARPLIGHT_ATTN_FE_CAUSAL=0</c> uses the general flash kernel <c>fa</c> (the summation order before 2026-11-15).
    /// </summary>
    public static bool AttentionCausalFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_FE_CAUSAL");

    /// <summary>False keeps bidirectional attention on devices without the tensor-core flash kernels off the f32 encoder flash kernel (fes).</summary>
    public static bool AttentionSimtFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_FES");

    /// <summary>
    /// True (default) when f32 matmuls may run on the split-f16 tensor-core GEMM where a compilation allows it
    /// (<c>CompileOptions.SplitHalfMatMul</c>); <c>SHARPLIGHT_GEMM_SPLIT_F16=0</c> keeps them on the f32 SIMT GEMM.
    /// </summary>
    public static bool GemmSplitHalf { get; } = IsNotZero("SHARPLIGHT_GEMM_SPLIT_F16");

    /// <summary>
    /// <c>SHARPLIGHT_GEMM_SPLIT_FILL=1</c>: grids of a wave or more of tensor-core GEMM tiles split their K in two by the earlier rule
    /// (two halves fill the last wave 15 points better) instead of by the time in whole waves.
    /// </summary>
    public static bool GemmSplitFillRule { get; } = Read("SHARPLIGHT_GEMM_SPLIT_FILL") == "1";

    /// <summary>
    /// True (default) when strict-precision convs with constant f32 weights may run in the split-f16 form of conv_cm
    /// (<c>CompileOptions.SplitHalfMatMul</c>); <c>SHARPLIGHT_CONV_SPLIT_F16=0</c> keeps them on the three-part bf16 form.
    /// </summary>
    public static bool ConvSplitHalf { get; } = IsNotZero("SHARPLIGHT_CONV_SPLIT_F16");

    /// <summary>
    /// True (default) when bidirectional attention reads strided f32 K / V (slices of a fused projection) in place;
    /// <c>SHARPLIGHT_ATTN_STRIDED_KV=0</c> copies them to dense rows first.
    /// </summary>
    public static bool AttentionStridedKv { get; } = IsNotZero("SHARPLIGHT_ATTN_STRIDED_KV");

    /// <summary>True when decode attention may split the keys over several workgroups.</summary>
    public static bool AttentionSplit { get; } = IsNotZero("SHARPLIGHT_ATTN_SPLIT");

    /// <summary>True when decode attention may use the GQA-grouped split-K kernel.</summary>
    public static bool AttentionDecode { get; } = IsNotZero("SHARPLIGHT_ATTN_DECODE");

    /// <summary>False merges the partials of split-K decoding in a separate combine launch instead of the last workgroup of each row (<c>SHARPLIGHT_ATTN_DECODE_FUSED=0</c>).</summary>
    public static bool AttentionDecodeFusedCombine { get; } = IsNotZero("SHARPLIGHT_ATTN_DECODE_FUSED");

    /// <summary><c>SHARPLIGHT_ATTN_DECODE_FUSED_WIDTH</c>: largest merged width (heads per workgroup x DV) of the fused decode combine, or null for the default.</summary>
    public static long? AttentionDecodeFusedWidth { get; } = Read("SHARPLIGHT_ATTN_DECODE_FUSED_WIDTH") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Split-K decode tuning: target workgroups, minimum keys per split, maximum splits, [rows per lane group and batch].</summary>
    public static int[] AttentionDecodeConfig { get; } = (Read("SHARPLIGHT_ATTN_DECODE_CFG") ?? "1024,128,64").Split(',').Select(int.Parse).ToArray();

    /// <summary>True when <c>SHARPLIGHT_ATTN_DECODE_CFG</c> is set (its target workgroups then apply on every target).</summary>
    public static bool AttentionDecodeConfigSet { get; } = Read("SHARPLIGHT_ATTN_DECODE_CFG") is not null;

    /// <summary>
    /// True (default) when multi-sequence steps that mix decode rows with prompt chunks split attention between split-K decoding
    /// (short runs) and flash attention (long runs); <c>SHARPLIGHT_ATTN_MIXED=0</c> keeps one kernel for the whole step.
    /// </summary>
    public static bool AttentionMixed { get; } = IsNotZero("SHARPLIGHT_ATTN_MIXED");

    /// <summary>
    /// Mean rows per run below which a multi-sequence step takes the mixed attention form (<c>SHARPLIGHT_ATTN_MIXED_RUN_ROWS=n</c>; unset:
    /// the lowering's default).
    /// </summary>
    public static long? AttentionMixedRunRows { get; } = Read("SHARPLIGHT_ATTN_MIXED_RUN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>True when multi-sequence batches may use flash attention.</summary>
    public static bool AttentionSequenceFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_SEQ_FA");

    /// <summary>True when shared-key-row attention (absorbed MLA) may use the tensor-core <c>fa_mqa</c> kernel.</summary>
    public static bool AttentionMqaFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_MQA");

    /// <summary>
    /// True (default) when decode rows of head sizes 256 and 512 with separate key and value caches may run the tensor-core
    /// <c>fa_mqa</c> form (<c>FM_SEPV</c>) on targets without the hand-written CUDA kernel (<c>SHARPLIGHT_ATTN_DECODE_TC=0</c>: <c>decode</c>).
    /// </summary>
    public static bool AttentionDecodeTensorCores { get; } = IsNotZero("SHARPLIGHT_ATTN_DECODE_TC");

    /// <summary>
    /// <c>SHARPLIGHT_ATTN_DECODE_TC_CFG=waves,minKeys,maxSplits,keys</c>: the split rule and tile of the tensor-core decode form (workgroups
    /// per compute unit the split count aims for, minimum keys per range, maximum ranges, keys per tile), or null for the defaults.
    /// </summary>
    public static int[]? AttentionDecodeTensorCoreConfig { get; } = Read("SHARPLIGHT_ATTN_DECODE_TC_CFG")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Preferred lanes per state row of the gated delta rule.</summary>
    public static int DeltaLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_DELTA_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred state rows per workgroup of the gated delta rule.</summary>
    public static long DeltaRowsPerBlock { get; } = long.Parse(Read("SHARPLIGHT_DELTA_DVB") ?? "16", CultureInfo.CurrentCulture);

    /// <summary>True when the gated delta rule may use its chunked tensor-core form for long runs.</summary>
    public static bool DeltaChunked { get; } = IsNotZero("SHARPLIGHT_DELTA_CHUNK");

    /// <summary>Rows of one sequence from which the chunked delta rule serves it.</summary>
    public static int DeltaChunkMinRows { get; } = int.Parse(Read("SHARPLIGHT_DELTA_CHUNK_MIN") ?? "64", CultureInfo.CurrentCulture);

    /// <summary>State columns per workgroup of the chunked delta rule's sequential pass (0 = chosen by the lowering).</summary>
    public static int DeltaChunkColumns { get; } = int.Parse(Read("SHARPLIGHT_DELTA_CHUNK_BV") ?? "0", CultureInfo.CurrentCulture);

    /// <summary>Forced operand parts of the chunked delta rule (1 or 2; 0 = from the precision policy).</summary>
    public static int DeltaChunkParts { get; } = int.Parse(Read("SHARPLIGHT_DELTA_CHUNK_PARTS") ?? "0", CultureInfo.CurrentCulture);

    /// <summary>Threads per workgroup (128 or 256) of the sequential pass of the chunked delta rule (0 = by column block).</summary>
    public static int DeltaChunkThreads { get; } = int.Parse(Read("SHARPLIGHT_DELTA_CHUNK_THREADS") ?? "0", CultureInfo.CurrentCulture);

    /// <summary>Whether prefill batches run the causal conv1d token-parallel (<c>SHARPLIGHT_CONV1D_PARALLEL=0</c> disables).</summary>
    public static bool Conv1dParallel { get; } = IsNotZero("SHARPLIGHT_CONV1D_PARALLEL");

    /// <summary>Tokens per thread of the token-parallel causal conv1d.</summary>
    public static int Conv1dTokens { get; } = int.Parse(Read("SHARPLIGHT_CONV1D_TOKENS") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred lanes per row of the selective scan.</summary>
    public static int SsmLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_SSM_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred lanes per state row of the RWKV recurrences.</summary>
    public static int WkvLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_WKV_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred state rows per workgroup of the RWKV recurrences.</summary>
    public static long WkvRowsPerBlock { get; } = long.Parse(Read("SHARPLIGHT_WKV_RB") ?? "16", CultureInfo.CurrentCulture);

    /// <summary>(token, slot) items up to which a per-item expert matvec beats grouping by expert.</summary>
    public static int MoeMatvecMaxItems { get; } = int.Parse(Read("SHARPLIGHT_MOE_MV_MAX") ?? "16", CultureInfo.InvariantCulture);

    /// <summary>
    /// (token, slot) items up to which the per-item int-dot expert matvec beats grouping by expert; 0 (default) for half the expert
    /// count (16-128) with the tensor-core grouped GEMM, 128 without.
    /// </summary>
    public static int MoeIntDotMaxItems { get; } = int.Parse(Read("SHARPLIGHT_MOE_MMVQ_MAX") ?? "0", CultureInfo.InvariantCulture);

    /// <summary>Forced expert-matmul token tile, or 0 for the heuristic.</summary>
    public static int MoeTile { get; } = int.Parse(Read("SHARPLIGHT_MOE_TILE") ?? "0", CultureInfo.InvariantCulture);

    /// <summary>
    /// Without tensor cores the 64-row scalar tile only pays off when experts see many rows: below this average the
    /// per-item matvec (which re-reads shared experts' weights from cache) is faster.
    /// </summary>
    public static int MoeScalarRows { get; } = int.Parse(Read("SHARPLIGHT_MOE_SCALAR_ROWS") ?? "12", CultureInfo.InvariantCulture);

    /// <summary>False selects moe_gemm_cm instead of the expert-grouped gemm_cm2 for tensor-core expert matmuls.</summary>
    public static bool MoeGemm2 { get; } = IsNotZero("SHARPLIGHT_MOE_GEMM2");

    /// <summary>
    /// True when the expert-grouped gemm_cm2 reads the routed activation rows as expert-sorted f16 rows gathered once per routing
    /// (shared by the projections reading the same x) instead of gathering and converting f32 rows in every column tile.
    /// </summary>
    public static bool MoeSortedHalfActivations { get; } = IsNotZero("SHARPLIGHT_MOE_X16");

    /// <summary>
    /// False (<c>SHARPLIGHT_MOE_SORTED_STORE=0</c>) keeps the f32 store of the gate / up expert GEMM on the hand-written grouped GEMM
    /// and the down projection's gather, instead of the gate / up GEMM storing the down projection's expert-sorted f16 rows.
    /// </summary>
    public static bool MoeSortedStore { get; } = IsNotZero("SHARPLIGHT_MOE_SORTED_STORE");

    /// <summary>
    /// False (<c>SHARPLIGHT_MOE_TOKEN_ROWS=0</c>) gathers the expert-sorted f16 rows of the gate / up GEMM's activations for the
    /// hand-written grouped GEMM instead of letting it read its items' token rows from the f16 copy another GEMM (the router) reads.
    /// </summary>
    public static bool MoeTokenRows { get; } = IsNotZero("SHARPLIGHT_MOE_TOKEN_ROWS");

    /// <summary>
    /// True when the expert-grouped gemm_cm2 with 64- or 128-row tiles leaves the tiles of at most half that many rows (an
    /// expert's last, partly filled tile) to a second launch with half-height tiles, which multiplies less padding.
    /// </summary>
    public static bool MoeSplitTiles { get; } = IsNotZero("SHARPLIGHT_MOE_SPLIT_TILES");

    /// <summary>
    /// True when a grouped expert product whose only reader adds the routed experts' bias rows adds them in the grouped GEMM's
    /// store (<c>SHARPLIGHT_MOE_BIAS_EPILOGUE=0</c>: a separate launch).
    /// </summary>
    public static bool MoeBiasEpilogue { get; } = IsNotZero("SHARPLIGHT_MOE_BIAS_EPILOGUE");

    /// <summary>
    /// Grouped expert GEMMs whose gate / up pairs (products feeding one elementwise group) run as one dual launch that stores the
    /// group: bit 0 the grouped gemm_cm2, bit 1 the expert-grouped skinny GEMM (<c>SHARPLIGHT_MOE_DUAL=n</c>, default 3; 0: two
    /// grouped GEMMs and the group's launch).
    /// </summary>
    public static int MoeDual { get; } = int.Parse(Read("SHARPLIGHT_MOE_DUAL") ?? "3", CultureInfo.InvariantCulture);

    /// <summary>
    /// Grouped expert GEMMs launched with the expert tiles along the grid's x (bit 0: the expert-grouped skinny GEMM, bit 1:
    /// gemm_cm2), so the tiles of one expert read each weight chunk at about the same time (<c>SHARPLIGHT_MOE_TILE_MAJOR=n</c>).
    /// </summary>
    public static int MoeTileMajor { get; } = int.Parse(Read("SHARPLIGHT_MOE_TILE_MAJOR") ?? "0", CultureInfo.InvariantCulture);

    /// <summary>
    /// Average (token, slot) items per expert up to which the expert-grouped skinny GEMM (16-row expert tiles) serves the grouped
    /// expert products (<c>SHARPLIGHT_MOE_SKINNY=n</c>; 0 keeps them on gemm_cm2), or null for the default.
    /// </summary>
    public static int? MoeSkinnyItemsPerExpert { get; } = Read("SHARPLIGHT_MOE_SKINNY") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// Configuration of the wide expert tiles of the direct skinny GEMM (<c>SHARPLIGHT_MOE_WIDE=fm,fn,sg,ks,n</c>: token fragments,
    /// weight fragments per subgroup, subgroups, subgroups splitting K, items per expert up to which it serves; <c>0</c> disables
    /// them), or null for the default.
    /// </summary>
    public static int[]? MoeWideConfig { get; } = Read("SHARPLIGHT_MOE_WIDE")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>
    /// False (<c>SHARPLIGHT_SPLITK_EPILOGUE=0</c>) keeps the elementwise group after an f16 tensor-core matmul (bias, residual) as
    /// its own launch instead of applying it in the split-K reduction.
    /// </summary>
    public static bool SplitKReduceEpilogue { get; } = IsNotZero("SHARPLIGHT_SPLITK_EPILOGUE");

    /// <summary>
    /// False (<c>SHARPLIGHT_SKINNY_EPILOGUE=0</c>) keeps the elementwise group after a skinny GEMM (2-16 rows: a projection's bias,
    /// a residual add) as its own launch instead of applying it in the skinny GEMM's store.
    /// </summary>
    public static bool SkinnyEpilogue { get; } = IsNotZero("SHARPLIGHT_SKINNY_EPILOGUE");

    /// <summary>
    /// True (<c>SHARPLIGHT_SPLITK_FUSED=1</c>) runs the split-K reduction of gemm_cm2 and the fp8 GEMM in the last workgroup of each
    /// output tile (splitk.slang) instead of its own launch (<c>splitk_reduce</c>). Off by default: on the RTX PRO 4500 the last
    /// workgroup's serial read of the tile's slices cost more than the launch it saves (benchmarks.md section 61).
    /// </summary>
    public static bool SplitKFused { get; } = Read("SHARPLIGHT_SPLITK_FUSED") == "1";

    /// <summary>
    /// True (<c>SHARPLIGHT_MULTI_FUSED=1</c>) reduces the multi-output GEMM in its last workgroups instead of a <c>splitk_reduce</c>
    /// launch over its partials.
    /// </summary>
    public static bool MultiGemmFused { get; } = Read("SHARPLIGHT_MULTI_FUSED") == "1";

    /// <summary>
    /// False (<c>SHARPLIGHT_MULTI_GEMM_TALL=0</c>) keeps the members of a q / k / v group on their own launches of the hand-written f16
    /// GEMM's prompt-sized tiles instead of one multi-output launch over their concatenated weight rows.
    /// </summary>
    public static bool MultiGemmTall { get; } = Read("SHARPLIGHT_MULTI_GEMM_TALL") != "0";

    /// <summary><c>SHARPLIGHT_MULTI_GEMM_MAX_ROWS=n</c>: the largest batch the split-K multi-output GEMM serves, or null for the default.</summary>
    public static long? MultiGemmMaxRows { get; } = Read("SHARPLIGHT_MULTI_GEMM_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_MULTI_DEBUG=1</c>: report on stderr why a multi-output group got no GEMM.</summary>
    public static bool MultiGemmDebug { get; } = Read("SHARPLIGHT_MULTI_DEBUG") == "1";

    /// <summary>
    /// False (<c>SHARPLIGHT_MULTI_GEMM=0</c>) lowers sibling projections sharing their activations (q / k / v) as separate GEMMs
    /// instead of one multi-output GEMM (splitk.slang, MULTI) at the row counts that split K.
    /// </summary>
    public static bool MultiGemm { get; } = IsNotZero("SHARPLIGHT_MULTI_GEMM");

    /// <summary>
    /// <c>SHARPLIGHT_SPLIT_FILL=1</c>: grids of one token tile row choose their K split by wave fill (see GraphLowerer.WaveFillSplit)
    /// instead of aiming for 4/3 waves.
    /// </summary>
    public static bool SplitWaveFill { get; } = Read("SHARPLIGHT_SPLIT_FILL") == "1";

    /// <summary><c>SHARPLIGHT_FP8_DUAL_SPLIT=1</c>: the dual fp8 GEMM splits K at decode sizes (see GraphLowerer.DualFp8Split).</summary>
    public static bool Fp8DualSplit { get; } = Read("SHARPLIGHT_FP8_DUAL_SPLIT") == "1";

    /// <summary><c>SHARPLIGHT_NATIVE_MOE_BN=n</c>: weight rows per tile of the grouped MXFP4 GEMM (experiment), or null for the chosen width.</summary>
    public static int? NativeMoeColumns { get; } = Read("SHARPLIGHT_NATIVE_MOE_BN") is { } c ? int.Parse(c, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_MOE_SHORT=n</c>: the grouped MXFP4 GEMM routes 64-row tiles up to n items per expert (0: never), or null for the default.</summary>
    public static long? NativeMoeShortItems { get; } = Read("SHARPLIGHT_NATIVE_MOE_SHORT") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_MOE_BLOCKS=1</c>: the short tiles of the grouped MXFP4 GEMM keep one block per SM, or null for the default.</summary>
    public static int? NativeMoeBlocks { get; } = Read("SHARPLIGHT_NATIVE_MOE_BLOCKS") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_MOE_TABLE=1</c>: the short dual tiles of the grouped MXFP4 GEMM keep the E8M0 factor table (three stages), or null for the default.</summary>
    public static int? NativeMoeTable { get; } = Read("SHARPLIGHT_NATIVE_MOE_TABLE") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_NATIVE_F16_DECODE_BN=n</c>: weight rows per tile of the hand-written f16 GEMM's decode form (32 or 64), or null for
    /// the default (chosen per matrix); 0 turns the decode form off (decode batches of f16 projections keep gemm_cm2 with split-K).
    /// </summary>
    public static int? NativeF16DecodeColumns { get; } = Read("SHARPLIGHT_NATIVE_F16_DECODE_BN") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_F16_DECODE_MIN_ROWS=n</c>: fewest rows the hand-written f16 GEMM's decode form serves, or null for the default.</summary>
    public static long? NativeF16DecodeMinRows { get; } = Read("SHARPLIGHT_NATIVE_F16_DECODE_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_KQ_MIN_ROWS=n</c>: fewest rows the hand-written K-quant decode GEMM serves, or null for the default (the first batch past the matvec).</summary>
    public static long? NativeKQuantMinRows { get; } = Read("SHARPLIGHT_NATIVE_KQ_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_KQ_BM=32|64</c>: token rows per tile of the hand-written K-quant decode GEMM, or null for the default.</summary>
    public static int? NativeKQuantTileRows { get; } = Read("SHARPLIGHT_NATIVE_KQ_BM") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_KQ_WM=1|2</c>: warps along the token rows of the hand-written K-quant decode GEMM's tile, or null for the default (chosen per tile).</summary>
    public static int? NativeKQuantWarpRows { get; } = Read("SHARPLIGHT_NATIVE_KQ_WM") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_NATIVE_KQ_DUAL=0</c>: K-quant gate / up pairs at decode batches run two hand-written K-quant GEMM launches and the
    /// elementwise launch instead of the GEMM's dual form.
    /// </summary>
    public static bool NativeKQuantDual { get; } = IsNotZero("SHARPLIGHT_NATIVE_KQ_DUAL");

    /// <summary>
    /// <c>SHARPLIGHT_COOP_KQ=0</c>: targets without CUDA C++ (Vulkan) keep K-quant decode batches of 9-64 rows on the generated gemm_cm2
    /// instead of the cooperative-matrix K-quant decode GEMM (<c>gemm_kq.slang</c>).
    /// </summary>
    public static bool CoopKQuant { get; } = IsNotZero("SHARPLIGHT_COOP_KQ");

    /// <summary><c>SHARPLIGHT_NATIVE_KQ_SPLIT=n</c>: K split of the hand-written K-quant decode GEMM, or null for the default (chosen per matrix and batch).</summary>
    public static int? NativeKQuantSplit { get; } = Read("SHARPLIGHT_NATIVE_KQ_SPLIT") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_FP8_DECODE_KERNEL=1</c> / <c>=0</c>: batches of at most 16 rows of the hand-written FP8 GEMM's decode form run on
    /// the register-streaming <c>gemm_fp8_decode.cu</c> / on <c>gemm_fp8_native.cu</c>; null for the default.
    /// </summary>
    public static bool? Fp8DecodeKernel { get; } = Read("SHARPLIGHT_FP8_DECODE_KERNEL") is { } v ? v != "0" : null;

    /// <summary>
    /// <c>SHARPLIGHT_FP8_BLOCK_SCALED=1|0</c>: where the device has <c>block_scaled_fp8</c>, run the 128-row tiles of the FP8 GEMM on the
    /// block-scaled MMA (<c>gemm_fp8_tma.cu</c> with <c>F8N_BLOCK_SCALED</c>) with power-of-two activation scales and E4M3 weights requantized to power-of-two scales at
    /// load (1, the default since 2026-11-23 where a segment has such weights), or keep the f32 scales (0). Null: the default (on where it
    /// applies; a load turns it off with <c>TaskLoadOptions.BlockScaledFp8</c>, which disables the feature).
    /// </summary>
    public static bool? Fp8BlockScaled { get; } = Read("SHARPLIGHT_FP8_BLOCK_SCALED") is { } v ? v != "0" : null;

    /// <summary>
    /// <c>SHARPLIGHT_FP8_UE8M0_WEIGHTS=1</c> (diagnostic): requantize grouped E4M3 weights to power-of-two scales as the block-scaled mode
    /// does, but keep the f32-scale kernels and activation scales (separates the weights' share of that mode's numerics change).
    /// </summary>
    public static bool Fp8Ue8m0WeightsOnly { get; } = Read("SHARPLIGHT_FP8_UE8M0_WEIGHTS") == "1";

    /// <summary>
    /// <c>SHARPLIGHT_FP8_QSTORE_MIN_ROWS=n</c>: in the block-scaled FP8 mode, the dual gate / up FP8 GEMM on 128-row tiles stores its
    /// readers' E4M3 rows from n rows on (0: never; default 256).
    /// </summary>
    public static long? Fp8StoreMinRows { get; } = Read("SHARPLIGHT_FP8_QSTORE_MIN_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_NATIVE_FP8_DECODE_BN=n</c>: weight rows per tile of the hand-written FP8 GEMM at decode batches (at most 64 rows), or null for the default (128: no decode form).</summary>
    public static int? NativeFp8DecodeColumns { get; } = Read("SHARPLIGHT_NATIVE_FP8_DECODE_BN") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_ATTN_STAGES=n</c>: stages of the key / value ring of the hand-written prefill attention, or null for the default.</summary>
    public static int? NativeAttentionStages { get; } = Read("SHARPLIGHT_ATTN_STAGES") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary><c>SHARPLIGHT_ATTN_MR=n</c>: 16-row query fragments per warp (1 or 2) of the hand-written prefill attention, or null for the default.</summary>
    public static int? NativeAttentionFragments { get; } = Read("SHARPLIGHT_ATTN_MR") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// <c>SHARPLIGHT_ATTN_HG=1</c> runs the query heads of a KV head in one block of the hand-written prefill attention (<c>FN_HG</c>)
    /// wherever it can, <c>=0</c> nowhere; unset, the lowering's rule.
    /// </summary>
    public static bool? NativeAttentionHeadGroups { get; } = Read("SHARPLIGHT_ATTN_HG") switch { "1" => true, "0" => false, _ => null };

    /// <summary><c>SHARPLIGHT_NORM_THREADS=n</c>: threads per workgroup of the fused row norms (experiments), or null for the default.</summary>
    public static int? RowNormThreads { get; } = Read("SHARPLIGHT_NORM_THREADS") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>False (<c>SHARPLIGHT_NORM_WIDE=0</c>): fused row norms of decode-size steps keep 256-thread workgroups instead of 1024.</summary>
    public static bool RowNormWide { get; } = IsNotZero("SHARPLIGHT_NORM_WIDE");

    /// <summary><c>SHARPLIGHT_MULTI_SPLIT=n</c>: the K split of every multi-output GEMM (experiments), or null for the chosen one.</summary>
    public static int? MultiGemmSplit { get; } = Read("SHARPLIGHT_MULTI_SPLIT") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>
    /// False (<c>SHARPLIGHT_NATIVE_GEMM_BIAS=0</c>) keeps matmuls with a bias row on the generated GEMMs at prefill sizes instead of the
    /// hand-written f16 GEMM with the bias in its store.
    /// </summary>
    public static bool NativeGemmBias { get; } = IsNotZero("SHARPLIGHT_NATIVE_GEMM_BIAS");

    /// <summary>Opt-in (<c>SHARPLIGHT_FP8_SHORT_TILES=1</c>): the fp8 GEMM takes 64-row tiles above 64 rows where they pad fewer rows.</summary>
    public static bool Fp8ShortTiles { get; } = IsSet("SHARPLIGHT_FP8_SHORT_TILES");

    /// <summary>Rows up to which the int8 / fp8 GEMMs take 64-row tiles (<c>SHARPLIGHT_MMQ_SHORT_MAX_ROWS</c>), or null for the default.</summary>
    public static long? MmqShortTileMaxRows { get; } = Read("SHARPLIGHT_MMQ_SHORT_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Disables the tensor-core expert matmul.</summary>
    public static bool MoeNoCoop { get; } = IsSet("SHARPLIGHT_MOE_NO_COOP");

    /// <summary>Disables the cooperative-matrix convolution (conv_cm).</summary>
    public static bool ConvNoMatrix { get; } = IsSet("SHARPLIGHT_CONV_NO_MATRIX");

    /// <summary>Keeps conv_cm's channel-major reduction order for the split-f16 forms (no tap-major weights).</summary>
    public static bool ConvNoTapMajor { get; } = IsSet("SHARPLIGHT_CONV_NO_TAP_MAJOR");

    /// <summary>One output per thread in the depthwise conv (conv_dw) instead of up to 4 of a row.</summary>
    public static bool ConvDwSingle { get; } = IsSet("SHARPLIGHT_CONV_DW_SINGLE");

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name);

    private static bool IsSet(string name) => Read(name) == "1";

    private static bool IsNotZero(string name) => Read(name) != "0";
}
