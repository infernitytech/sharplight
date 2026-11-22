using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// Keys per tile of <c>attn_prefill_native.cu</c> (<c>FN_BC</c>) for head size <paramref name="headSize"/>: 64, 32 above 128, where
    /// a warp's output accumulators take 128 registers and a 64-key stage would take 64 KiB of shared memory, and 16 above 256, where
    /// warp pairs split the head dimensions (<c>FN_DS</c>) and a 32-key stage of 512-wide rows would take 64 KiB.
    /// </summary>
    private static int NativeAttentionKeys(long headSize) => headSize > 256 ? 16 : headSize > 128 ? 32 : 64;

    /// <summary>
    /// Bytes of the score exchange of <c>attn_prefill_native.cu</c> behind its stages: above head size 256 each warp of the
    /// <see cref="NativeAttentionWarps"/> publishes its partial 16 x <see cref="NativeAttentionKeys"/> scores (f32) to its partner.
    /// </summary>
    private static int NativeAttentionExchangeBytes(AttentionPlan plan)
    {
        long headSize = Math.Max(plan.D, plan.DV);
        return headSize > 256 ? NativeAttentionWarps(headSize) * NativeAttentionFragments(headSize) * 16 * NativeAttentionKeys(headSize) * 4 : 0;
    }

    /// <summary>
    /// Warps of an <c>attn_prefill_native.cu</c> block for head size <paramref name="headSize"/> (16 query rows each): 4 up to 64, where
    /// a 32 KB two-stage ring lets three blocks share an SM, and 8 above, where the 64 KB ring leaves room for one block, which then
    /// needs the warps itself (RTX PRO 4500, 2048 causal queries, head size 128: 4 warps 0.39 ms, 8 warps 0.27 ms per layer).
    /// </summary>
    private static int NativeAttentionWarps(long headSize) => headSize <= 64 ? 4 : 8;

    /// <summary>
    /// 16-row query fragments per warp of <c>attn_prefill_native.cu</c> (<c>FN_MR</c>) for head size <paramref name="headSize"/>: two up to
    /// 64, where each key and value fragment then feeds twice the MMAs (255 registers, two blocks per SM), one above (two would spill).
    /// RTX PRO 4500, gpt-oss-20b shapes (bench/s58_attn, 2048 causal queries after n_past keys): n_past 0 0.254 → 0.248 ms, 14,336
    /// 3.54 → 3.29 ms, 63,488 15.8 → 14.1 ms, 126,976 31.5 → 28.0 ms; sliding-window layers unchanged. <c>SHARPLIGHT_ATTN_MR=n</c> sets it.
    /// </summary>
    private static int NativeAttentionFragments(long headSize) => DiagnosticSwitches.NativeAttentionFragments ?? (headSize <= 64 ? 2 : 1);

    /// <summary>
    /// Stages of the key / value ring of <c>attn_prefill_native.cu</c> (<c>FN_STAGES</c>): three above head size 64, where one block has
    /// the SM to itself and a tile's copy takes about as long as its multiplication, when they fit the shared memory; else two (at head
    /// size 64 a third stage would leave room for two blocks per SM instead of three). <c>SHARPLIGHT_ATTN_STAGES=n</c> sets the count.
    /// </summary>
    private int NativeAttentionStages(AttentionPlan plan)
    {
        int stages = DiagnosticSwitches.NativeAttentionStages ?? (Math.Max(plan.D, plan.DV) > 64 ? 3 : 2);
        return stages > 2 && NativeAttentionStageBytes(plan) * stages + NativeAttentionExchangeBytes(plan) > _opt.MaxSharedMemory ? 2 : stages;
    }

    /// <summary>Bytes of one stage of <c>attn_prefill_native.cu</c>: a key and a value tile in f16.</summary>
    private static int NativeAttentionStageBytes(AttentionPlan plan) => (int)(NativeAttentionKeys(Math.Max(plan.D, plan.DV)) * (plan.D + plan.DV) * 2);

    /// <summary>Dynamic shared memory of <c>attn_prefill_native.cu</c>: <see cref="NativeAttentionStages"/> stages of a key and a value tile in f16.</summary>
    private int NativeAttentionSharedBytes(AttentionPlan plan) => NativeAttentionStageBytes(plan) * NativeAttentionStages(plan) + NativeAttentionExchangeBytes(plan);

    /// <summary>
    /// True when causal attention of many queries against a KV cache runs on the hand-written flash-attention kernel
    /// (<see cref="DeviceFeatures.NativeAttention"/>, <c>attn_prefill_native.cu</c>) instead of <c>fa</c> / <c>fe</c>: f16 caches, head sizes
    /// that are multiples of 64 up to 256 or of 128 up to 512, grouped-query heads, optionally a sliding window, chunks, a soft cap and sinks, single-sequence or multi-sequence (also
    /// paged); no score biases, spans, masks, selections, compressed entries or relative biases. <c>SHARPLIGHT_NATIVE_ATTN=2</c> keeps
    /// the attentions <c>fe</c> serves (plain causal) on <c>fe</c>, <c>=0</c> all on the generated kernels.
    /// </summary>
    private bool NativeFlashAttention(AttentionPlan plan)
    {
        var a = plan.Node.Attributes;
        int mode = DiagnosticSwitches.NativeAttention;
        long split = Math.Max(plan.D, plan.DV) > 256 ? 2 : 1;
        bool shapes = plan.K.Type == DataType.F16 && plan.V.Type == DataType.F16 && plan.D % (64 * split) == 0 && plan.DV % (64 * split) == 0
                      && plan.D <= 512 && plan.DV <= 512 && plan.HQ % plan.HKV == 0 && plan.V.Shape[2] % 8 == 0
                      && plan.Q.Strides[0] % 2 == 0 && plan.Q.Strides[1] % 2 == 0 && plan.Q.ByteOffset % 8 == 0;
        bool form = !plan.Full && (plan.Seq is null || plan.Runs is not null) && plan.Spans is null && plan.Mask is null && plan.Selection is null
                    && plan.Compressed is null && plan.RelativeBias is null && a.Get("causal", true)
                    && a.Get<float[]?>("alibi_slopes", null) is null && a.Get<float[]?>("position_bias", null) is null;
        return mode > 0 && _opt.Has(DeviceFeatures.NativeAttention) && _compiler.CompilesCuda(_opt.Target) && shapes && form
               && NativeAttentionStageBytes(plan) * 2 + NativeAttentionExchangeBytes(plan) <= _opt.MaxSharedMemory && (mode == 1 || !CausalFlash(plan));
    }

    /// <summary>
    /// The launch variant of <c>attn_prefill_native.cu</c> for the bindings <paramref name="when"/> selects: grid (query heads, query tiles):
    /// with runs, ceil(rows / tile rows) + runs, a bound on the tiles of all runs together, which the kernel enumerates (no empty blocks for
    /// the tiles a short run does not have). <paramref name="mixRun"/> &gt; 0 skips runs shorter than that (mixed steps, where split-K decoding takes them).
    /// </summary>
    private KernelVariant NativeFlashVariant(AttentionPlan plan, Func<Bindings, bool> when, long mixRun = 0)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        int warps = NativeAttentionWarps(Math.Max(plan.D, plan.DV));
        int fragments = NativeAttentionFragments(Math.Max(plan.D, plan.DV));
        var macros = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in (string[])["HQ", "HKV", "D", "DV", "VRS", "QS0", "QS1", "SCALE", "WINDOW", "CHUNK", "HAS_SOFTCAP", "SOFTCAP", "HAS_SINKS",
                     "MULTI_SEQ", "SLOT_CTX", "PAGED", "PAGE_BS", "PAGE_MB", "RT_N_PAST"])
        {
            macros[name] = plan.Macros[name];
        }

        macros["MIX_RUN"] = mixRun.ToString(CultureInfo.InvariantCulture);
        macros["FN_WARPS"] = warps.ToString(CultureInfo.InvariantCulture);
        macros["FN_STAGES"] = NativeAttentionStages(plan).ToString(CultureInfo.InvariantCulture);
        macros["FN_MR"] = fragments.ToString(CultureInfo.InvariantCulture);
        macros["FN_BC"] = NativeAttentionKeys(Math.Max(plan.D, plan.DV)).ToString(CultureInfo.InvariantCulture);
        var kernel = _compiler.CompileCuda("attn_prefill_native.cu", "", "attn_prefill_native", warps * 32, NativeAttentionSharedBytes(plan), _opt.Target,
            macros, _opt.Cache);
        // Above head size 256 two warps share each row group (FN_DS).
        uint rows = (uint)(16 * warps * fragments / (Math.Max(plan.D, plan.DV) > 256 ? 2 : 1));
        return runs is null
            ? new KernelVariant(when, kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), rows), 1))
            : new KernelVariant(when, kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), rows) + (uint)Rows(b, runs), 1));
    }

    /// <summary>Warps of an <c>attn_decode_native.cu</c> block (each runs its own key tiles and softmax).</summary>
    private const int NativeDecodeWarps = 4;

    /// <summary>
    /// Waves of blocks per SM the split count of the wide form of <c>attn_decode_native.cu</c> (head sizes 256 and 512) aims for: its
    /// ring takes most of an SM's shared memory, so one block runs per SM and the split count is chosen so that the blocks of a launch
    /// fill whole waves (a 3.1-wave grid of 256 blocks would leave most SMs idle in its last wave).
    /// </summary>
    private const int NativeWideDecodeWaves = 2;

    /// <summary>
    /// True when the split-K decode attention of <paramref name="plan"/> runs on the hand-written tensor-core kernel
    /// (<see cref="DeviceFeatures.NativeAttention"/>, <c>attn_decode_native.cu</c>) in place of <c>decode</c>: f16 caches, equal key and
    /// value head sizes of 64, 128, 256 or 512 (the wide form above 128 when its ring fits the shared memory), at most 8 heads per block,
    /// the forms <see cref="NativeFlashAttention"/> serves (no score biases, spans, masks, selections, compressed entries or relative
    /// biases). The caller keeps <c>decode</c> where it merges its partials itself (the fused combine of narrow groups).
    /// <c>SHARPLIGHT_NATIVE_DECODE=0</c> keeps <c>decode</c> everywhere, <c>SHARPLIGHT_NATIVE_DECODE_WIDE=0</c> for head sizes above 128.
    /// </summary>
    private bool NativeDecodeAttention(AttentionPlan plan, DecodeAttentionPlan decode)
        => NativeDecodeForm(plan) && long.Parse(decode.Macros["HPW"], CultureInfo.InvariantCulture) <= 8;

    /// <summary>The conditions of <see cref="NativeDecodeAttention"/> that do not depend on the split plan.</summary>
    private bool NativeDecodeForm(AttentionPlan plan)
    {
        var a = plan.Node.Attributes;
        bool wide = plan.D is 256 or 512;
        bool shapes = plan.K.Type == DataType.F16 && plan.V.Type == DataType.F16 && plan.D == plan.DV && (plan.D is 64 or 128 || wide)
                      && plan.V.Shape[2] % 8 == 0 && plan.Q.Strides[0] % 2 == 0 && plan.Q.Strides[1] % 2 == 0 && plan.Q.ByteOffset % 8 == 0
                      && (!wide || (DiagnosticSwitches.NativeWideDecodeAttention && NativeWideDecodeStages() >= 2));
        bool form = !plan.Full && !plan.Cross && plan.Spans is null && plan.Mask is null && plan.Selection is null && plan.Compressed is null
                    && plan.RelativeBias is null && a.Get<float[]?>("alibi_slopes", null) is null && a.Get<float[]?>("position_bias", null) is null;
        return DiagnosticSwitches.NativeDecodeAttention && DiagnosticSwitches.NativeAttention > 0 && _opt.Has(DeviceFeatures.NativeAttention)
               && _compiler.CompilesCuda(_opt.Target) && shapes && form;
    }

    /// <summary>True when <paramref name="plan"/>'s decode rows run the wide form of <c>attn_decode_native.cu</c> (head sizes 256 and 512).</summary>
    private bool NativeWideDecodeAttention(AttentionPlan plan) => plan.D > 128 && NativeDecodeForm(plan);

    /// <summary>
    /// Ring stages of the wide form of <c>attn_decode_native.cu</c> (<c>NWD_ST</c>): three (two tiles in flight while one is computed,
    /// 97 KiB) where the device's shared memory takes them, else two (65 KiB), else 0 (the form does not fit; <c>decode</c> serves).
    /// </summary>
    private int NativeWideDecodeStages()
    {
        for (int stages = 3; stages >= 2; stages--)
        {
            if (NativeWideDecodeSharedBytes(stages) <= _opt.MaxSharedMemory)
            {
                return stages;
            }
        }

        return 0;
    }

    /// <summary>Dynamic shared memory of the wide form with <paramref name="stages"/> stages: 32 KiB of keys and values per stage, the scores.</summary>
    private static int NativeWideDecodeSharedBytes(int stages) => stages * 32768 + NativeDecodeWarps * 32 * 8;

    /// <summary>
    /// <c>attn_decode_native.cu</c> for the macros <paramref name="decodeMacros"/> of <c>decode</c> (the plan's and the split rule's); it
    /// takes the launch grid and the argument block of <c>decode</c>.
    /// </summary>
    private CompiledKernel NativeDecodeKernel(AttentionPlan plan, Dictionary<string, string> decodeMacros)
    {
        var macros = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in (string[])["HQ", "HKV", "D", "VRS", "QS0", "QS1", "SCALE", "WINDOW", "CHUNK", "HAS_SOFTCAP", "SOFTCAP", "HAS_SINKS",
                     "MULTI_SEQ", "SLOT_CTX", "PAGED", "PAGE_BS", "PAGE_MB", "RT_N_PAST", "MIX_RUN", "HPW", "DEC_BC", "DEC_MIN_KEYS", "DEC_MAX_SPLITS",
                     "DEC_TARGET_WG", "DEC_INVARIANT_ROWS", "DEC_SPLIT_ROWS"])
        {
            if (decodeMacros.TryGetValue(name, out var value))
            {
                macros[name] = value;
            }
        }

        macros["ND_WARPS"] = NativeDecodeWarps.ToString(CultureInfo.InvariantCulture);
        if (DiagnosticSwitches.AttentionDecodePrefetch is { } prefetch)
        {
            macros["ND_PF"] = prefetch.ToString(CultureInfo.InvariantCulture);
        }

        int shared = 0;
        if (plan.D > 128)
        {
            int stages = NativeWideDecodeStages();
            macros["NWD_ST"] = stages.ToString(CultureInfo.InvariantCulture);
            shared = NativeWideDecodeSharedBytes(stages);
        }

        return _compiler.CompileCuda("attn_decode_native.cu", "", "attn_decode_native", NativeDecodeWarps * 32, shared, _opt.Target, macros, _opt.Cache);
    }
}
