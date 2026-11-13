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
/// <item><c>SHARPLIGHT_NO_GROUP_MERGE=1</c>: fuse only consecutive elementwise nodes (no merging of groups along data flow).</item>
/// <item><c>SHARPLIGHT_NO_SCHEDULE=1</c>: keep graph order instead of grouping launches by dependency level.</item>
/// <item><c>SHARPLIGHT_NO_REPACK=1</c>: keep Q8_0 weights in the file layout (no aligned-quants repack).</item>
/// <item><c>SHARPLIGHT_MMVQ_Q8_DECODE=0|1</c>: serve single-row Q8_0 matvecs with the integer-dot matvec too (default off: the
/// f32 matvec is already at the bandwidth roof there and exact in the activations).</item>
/// <item><c>SHARPLIGHT_MMVQ_BLOCK=0</c>: keep the 32-value block formats (Q8_0 as stored, Q4_0, Q5_0, IQ4_NL, MXFP4) on the f32
/// matvecs instead of the integer-dot matvec.</item>
/// <item><c>SHARPLIGHT_MMVQ_MAX_ROWS=n</c>: largest activation row count served by the integer-dot matvec (default 16 for
/// Q8_0, 8 for the K-quants).</item>
/// <item><c>SHARPLIGHT_MMVQ_LAYOUT=rows,subgroups</c>: force the integer-dot matvec's weight rows and subgroups per workgroup.</item>
/// <item><c>SHARPLIGHT_MMQ=0|all|fmt,fmt</c>: weight formats whose prefill matmuls use the int8 tensor-core GEMM (q8_1
/// activations, <c>mmq.slang</c>) instead of the f16 one: none, every supported format (q8_0, q4_k, q5_k, q6_k, iq4_nl, iq4_xs),
/// or a list (default: iq4_nl, iq4_xs, and q8_0 for matmuls with N and K of at least 4096; an explicit list drops that size
/// limit). Disabling <c>matrix_int8</c> turns it off as well.</item>
/// <item><c>SHARPLIGHT_MMQ_MIN_ROWS=n</c>: activation rows from which the int8 GEMM serves a matmul (default 65).</item>
/// <item><c>SHARPLIGHT_MMQ_CFG</c>, <c>SHARPLIGHT_MMQ_CFG64=wm,wn,fm,fn</c>: force the layout of the 128- and 64-token int8 GEMM
/// tiles. <c>SHARPLIGHT_MMQ_GENERIC=1</c>: scale its accumulators through the per-element (layout independent) path.
/// <c>SHARPLIGHT_MMQ_BK=32|64|128</c>: preferred K step of the int8 GEMM (default 128, 64 for Q6_K).</item>
/// <item><c>SHARPLIGHT_FP8_GEMM=0</c>: run prefill matmuls with block-scaled E4M3 weights on the f16 tensor-core GEMM instead of the
/// fp8 one (<c>gemm_fp8.slang</c>, E4M3 activations). Disabling <c>matrix_fp8</c> turns it off as well.</item>
/// <item><c>SHARPLIGHT_GEMM2=0</c>: use the older gemm_cm tensor-core GEMM instead of gemm_cm2.</item>
/// <item><c>SHARPLIGHT_GEMM2_CFG=wm,wn,fm,fn</c>: force one gemm_cm2 warp/fragment layout (must give 128x128 tiles).</item>
/// <item><c>SHARPLIGHT_GEMM2_SKINNY=0</c>: always use 128-token gemm_cm2 tiles (no 32/64-token tiles for small batches).</item>
/// <item><c>SHARPLIGHT_GEMM2_CFG32</c>, <c>SHARPLIGHT_GEMM2_CFG64=wm,wn,fm,fn</c>: force the layout of the 32- and 64-token
/// gemm_cm2 tiles.</item>
/// <item><c>SHARPLIGHT_GEMM_BK=32|64</c>: preferred K block of gemm_cm2 (64 when K allows it).</item>
/// <item><c>SHARPLIGHT_GEMM_X16=0</c>: tensor-core GEMMs read f32 activations instead of an f16 copy.</item>
/// <item><c>SHARPLIGHT_GEMM_NV2=1</c>: default of <see cref="Lowering.GraphLowerer.PreferWorkgroupMatrixGemm"/>.</item>
/// <item><c>SHARPLIGHT_GEMM_NV2_TILE=bt,bn,bk</c>: tile of the workgroup-matrix GEMM (default 128,128,32).</item>
/// <item><c>SHARPLIGHT_ATTN_FA=0</c>, <c>SHARPLIGHT_ATTN_SPLIT=0</c>, <c>SHARPLIGHT_ATTN_SEQ_FA=0</c>: disable tensor-core
/// flash attention, split-key decode attention, and flash attention for multi-sequence batches.</item>
/// <item><c>SHARPLIGHT_ATTN_DECODE=0</c>: disable split-K flash decoding (GQA-grouped decode attention with a run-time split
/// count); decode then uses the per-head split kernel. <c>SHARPLIGHT_ATTN_DECODE_CFG=wg,keys,splits[,rows]</c>: workgroups the split
/// count aims for, minimum keys per split, maximum splits (default 1024,128,64), and optionally key rows per lane group and batch.</item>
/// <item><c>SHARPLIGHT_ATTN_MQA=0</c>: disable the tensor-core attention for many query heads on one key row whose values are a
/// row prefix (absorbed multi-head latent attention, <c>fa_mqa</c>); prefill and decode then use the generic kernels.</item>
/// <item><c>SHARPLIGHT_DELTA_LPR</c>, <c>SHARPLIGHT_DELTA_DVB</c>, <c>SHARPLIGHT_SSM_LPR</c>: lanes per row and value-row
/// block of the delta-rule and selective-scan kernels.</item>
/// <item><c>SHARPLIGHT_MOE_MV_MAX</c>, <c>SHARPLIGHT_MOE_MMVQ_MAX</c>, <c>SHARPLIGHT_MOE_TILE</c>, <c>SHARPLIGHT_MOE_SCALAR_ROWS</c>,
/// <c>SHARPLIGHT_MOE_NO_COOP=1</c>: expert-matmul tuning (see <c>GraphLowerer.Moe.cs</c>). <c>SHARPLIGHT_MOE_GEMM2=0</c>: run
/// grouped expert matmuls on moe_gemm_cm instead of the expert-grouped gemm_cm2.</item>
/// <item><c>SHARPLIGHT_CONV_NO_MATRIX=1</c>: run convolutions on the SIMT conv_gemm instead of the cooperative-matrix conv_cm.</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>Directory receiving every compiled kernel's source, or null.</summary>
    public static string? DumpKernelsDirectory { get; } = Read("SHARPLIGHT_DUMP_KERNELS");

    /// <summary>Degree of parallelism of kernel compilation.</summary>
    public static int CompileThreads { get; } = int.Parse(Read("SHARPLIGHT_COMPILE_THREADS") ?? Math.Clamp(Environment.ProcessorCount / 2, 2, 8).ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);

    /// <summary>Disables the dual (gate/up) matvec fusion.</summary>
    public static bool NoDualMatMul { get; } = IsSet("SHARPLIGHT_NO_DUAL");

    /// <summary>False keeps the 32-value block formats off the integer-dot matvec.</summary>
    public static bool IntDotBlockFormats { get; } = IsNotZero("SHARPLIGHT_MMVQ_BLOCK");

    /// <summary>Disables merging elementwise groups along data flow.</summary>
    public static bool NoGroupMerge { get; } = IsSet("SHARPLIGHT_NO_GROUP_MERGE");

    /// <summary>Disables level scheduling of units.</summary>
    public static bool NoSchedule { get; } = IsSet("SHARPLIGHT_NO_SCHEDULE");

    /// <summary>Disables the Q8_0 aligned-quants repack.</summary>
    public static bool NoWeightRepack { get; } = IsSet("SHARPLIGHT_NO_REPACK");

    /// <summary>True when single-row Q8_0 matvecs use the integer-dot matvec.</summary>
    public static bool IntDotDecodeQ8 { get; } = IsSet("SHARPLIGHT_MMVQ_Q8_DECODE");

    /// <summary>Forced largest activation row count served by the integer-dot matvec, or null for the per-format default.</summary>
    public static long? IntDotMaxRows { get; } = Read("SHARPLIGHT_MMVQ_MAX_ROWS") is { } v ? long.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>Forced integer-dot matvec layout (weight rows per workgroup, subgroups per workgroup), or null.</summary>
    public static int[]? IntDotLayout { get; } = Read("SHARPLIGHT_MMVQ_LAYOUT")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Weight formats of the int8 tensor-core GEMM ("0", "all" or a comma-separated list), or null for the default.</summary>
    public static string? MmqFormats { get; } = Read("SHARPLIGHT_MMQ");

    /// <summary>Activation rows from which the int8 GEMM serves a matmul.</summary>
    public static long MmqMinRows { get; } = long.Parse(Read("SHARPLIGHT_MMQ_MIN_ROWS") ?? "65", CultureInfo.InvariantCulture);

    /// <summary>Forced layout of the 128-token int8 GEMM tile (warps M, warps N, fragments M, fragments N), or null.</summary>
    public static int[]? MmqLayout { get; } = Read("SHARPLIGHT_MMQ_CFG")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Forced layout of the 64-token int8 GEMM tile, or null.</summary>
    public static int[]? MmqLayout64 { get; } = Read("SHARPLIGHT_MMQ_CFG64")?.Split(',').Select(int.Parse).ToArray();

    /// <summary>Forced preferred K step of the int8 GEMM, or null for the per-format default.</summary>
    public static int? MmqBlockK { get; } = Read("SHARPLIGHT_MMQ_BK") is { } v ? int.Parse(v, CultureInfo.InvariantCulture) : null;

    /// <summary>True when the int8 GEMM always scales its accumulators per element.</summary>
    public static bool MmqGenericLayout { get; } = IsSet("SHARPLIGHT_MMQ_GENERIC");

    /// <summary>False keeps E4M3-weight prefill matmuls off the fp8 tensor-core GEMM.</summary>
    public static bool Fp8Gemm { get; } = IsNotZero("SHARPLIGHT_FP8_GEMM");

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

    /// <summary>Preferred K block of gemm_cm2.</summary>
    public static int GemmBlockK { get; } = int.Parse(Read("SHARPLIGHT_GEMM_BK") ?? "64", CultureInfo.InvariantCulture);

    /// <summary>True when tensor-core GEMMs read an f16 copy of the activations.</summary>
    public static bool GemmHalfActivations { get; } = IsNotZero("SHARPLIGHT_GEMM_X16");

    /// <summary>Default of <see cref="Lowering.GraphLowerer.PreferWorkgroupMatrixGemm"/>.</summary>
    public static bool WorkgroupMatrixGemm { get; } = IsSet("SHARPLIGHT_GEMM_NV2");

    /// <summary>Workgroup-matrix GEMM tile: token tile, column tile, K tile.</summary>
    public static int[] WorkgroupMatrixTile { get; } = (Read("SHARPLIGHT_GEMM_NV2_TILE") ?? "128,128,32").Split(',').Select(int.Parse).ToArray();

    /// <summary>True when tensor-core flash attention may be used.</summary>
    public static bool AttentionFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_FA");

    /// <summary>True when decode attention may split the keys over several workgroups.</summary>
    public static bool AttentionSplit { get; } = IsNotZero("SHARPLIGHT_ATTN_SPLIT");

    /// <summary>True when decode attention may use the GQA-grouped split-K kernel.</summary>
    public static bool AttentionDecode { get; } = IsNotZero("SHARPLIGHT_ATTN_DECODE");

    /// <summary>Split-K decode tuning: target workgroups, minimum keys per split, maximum splits, [rows per lane group and batch].</summary>
    public static int[] AttentionDecodeConfig { get; } = (Read("SHARPLIGHT_ATTN_DECODE_CFG") ?? "1024,128,64").Split(',').Select(int.Parse).ToArray();

    /// <summary>True when multi-sequence batches may use flash attention.</summary>
    public static bool AttentionSequenceFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_SEQ_FA");

    /// <summary>True when shared-key-row attention (absorbed MLA) may use the tensor-core <c>fa_mqa</c> kernel.</summary>
    public static bool AttentionMqaFlash { get; } = IsNotZero("SHARPLIGHT_ATTN_MQA");

    /// <summary>Preferred lanes per state row of the gated delta rule.</summary>
    public static int DeltaLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_DELTA_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred state rows per workgroup of the gated delta rule.</summary>
    public static long DeltaRowsPerBlock { get; } = long.Parse(Read("SHARPLIGHT_DELTA_DVB") ?? "16", CultureInfo.CurrentCulture);

    /// <summary>Preferred lanes per row of the selective scan.</summary>
    public static int SsmLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_SSM_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred lanes per state row of the RWKV recurrences.</summary>
    public static int WkvLanesPerRow { get; } = int.Parse(Read("SHARPLIGHT_WKV_LPR") ?? "8", CultureInfo.CurrentCulture);

    /// <summary>Preferred state rows per workgroup of the RWKV recurrences.</summary>
    public static long WkvRowsPerBlock { get; } = long.Parse(Read("SHARPLIGHT_WKV_RB") ?? "16", CultureInfo.CurrentCulture);

    /// <summary>(token, slot) items up to which a per-item expert matvec beats grouping by expert.</summary>
    public static int MoeMatvecMaxItems { get; } = int.Parse(Read("SHARPLIGHT_MOE_MV_MAX") ?? "16", CultureInfo.InvariantCulture);

    /// <summary>(token, slot) items up to which the per-item int-dot expert matvec beats grouping by expert.</summary>
    public static int MoeIntDotMaxItems { get; } = int.Parse(Read("SHARPLIGHT_MOE_MMVQ_MAX") ?? "128", CultureInfo.InvariantCulture);

    /// <summary>Forced expert-matmul token tile, or 0 for the heuristic.</summary>
    public static int MoeTile { get; } = int.Parse(Read("SHARPLIGHT_MOE_TILE") ?? "0", CultureInfo.InvariantCulture);

    /// <summary>
    /// Without tensor cores the 64-row scalar tile only pays off when experts see many rows: below this average the
    /// per-item matvec (which re-reads shared experts' weights from cache) is faster.
    /// </summary>
    public static int MoeScalarRows { get; } = int.Parse(Read("SHARPLIGHT_MOE_SCALAR_ROWS") ?? "12", CultureInfo.InvariantCulture);

    /// <summary>False selects moe_gemm_cm instead of the expert-grouped gemm_cm2 for tensor-core expert matmuls.</summary>
    public static bool MoeGemm2 { get; } = IsNotZero("SHARPLIGHT_MOE_GEMM2");

    /// <summary>Disables the tensor-core expert matmul.</summary>
    public static bool MoeNoCoop { get; } = IsSet("SHARPLIGHT_MOE_NO_COOP");

    /// <summary>Disables the cooperative-matrix convolution (conv_cm).</summary>
    public static bool ConvNoMatrix { get; } = IsSet("SHARPLIGHT_CONV_NO_MATRIX");

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name);

    private static bool IsSet(string name) => Read(name) == "1";

    private static bool IsNotZero(string name) => Read(name) != "0";
}
