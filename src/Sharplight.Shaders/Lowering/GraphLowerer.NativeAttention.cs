using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Keys per tile of <c>attn_prefill_native.cu</c>.</summary>
    private const int NativeAttentionKeys = 64;

    /// <summary>
    /// Warps of an <c>attn_prefill_native.cu</c> block for head size <paramref name="headSize"/> (16 query rows each): 4 up to 64, where
    /// a 32 KB two-stage ring lets three blocks share an SM, and 8 above, where the 64 KB ring leaves room for one block, which then
    /// needs the warps itself (RTX PRO 4500, 2048 causal queries, head size 128: 4 warps 0.39 ms, 8 warps 0.27 ms per layer).
    /// </summary>
    private static int NativeAttentionWarps(long headSize) => headSize <= 64 ? 4 : 8;

    /// <summary>Dynamic shared memory of <c>attn_prefill_native.cu</c>: two stages of a key and a value tile in f16.</summary>
    private static int NativeAttentionSharedBytes(AttentionPlan plan) => (int)(2 * NativeAttentionKeys * (plan.D + plan.DV) * 2);

    /// <summary>
    /// True when causal attention of many queries against a KV cache runs on the hand-written flash-attention kernel
    /// (<see cref="DeviceFeatures.NativeAttention"/>, <c>attn_prefill_native.cu</c>) instead of <c>fa</c> / <c>fe</c>: f16 caches, head sizes
    /// 64 or 128, grouped-query heads, optionally a sliding window, chunks, a soft cap and sinks, single-sequence or multi-sequence (also
    /// paged); no score biases, spans, masks, selections, compressed entries or relative biases. <c>SHARPLIGHT_NATIVE_ATTN=2</c> keeps
    /// the attentions <c>fe</c> serves (plain causal) on <c>fe</c>, <c>=0</c> all on the generated kernels.
    /// </summary>
    private bool NativeFlashAttention(AttentionPlan plan)
    {
        var a = plan.Node.Attributes;
        int mode = DiagnosticSwitches.NativeAttention;
        bool shapes = plan.K.Type == DataType.F16 && plan.V.Type == DataType.F16 && plan.D % 64 == 0 && plan.DV % 64 == 0
                      && plan.D <= 128 && plan.DV <= 128 && plan.HQ % plan.HKV == 0 && plan.V.Shape[2] % 8 == 0
                      && plan.Q.Strides[0] % 2 == 0 && plan.Q.Strides[1] % 2 == 0 && plan.Q.ByteOffset % 8 == 0;
        bool form = !plan.Full && (plan.Seq is null || plan.Runs is not null) && plan.Spans is null && plan.Mask is null && plan.Selection is null
                    && plan.Compressed is null && plan.RelativeBias is null && a.Get("causal", true)
                    && a.Get<float[]?>("alibi_slopes", null) is null && a.Get<float[]?>("position_bias", null) is null;
        return mode > 0 && _opt.Has(DeviceFeatures.NativeAttention) && _compiler.CompilesCuda(_opt.Target) && shapes && form
               && NativeAttentionSharedBytes(plan) <= _opt.MaxSharedMemory && (mode == 1 || !CausalFlash(plan));
    }

    /// <summary>
    /// The launch variant of <c>attn_prefill_native.cu</c> for the bindings <paramref name="when"/> selects: grid (query heads, query tiles
    /// of the longest run, runs). <paramref name="mixRun"/> &gt; 0 skips runs shorter than that (mixed steps, where split-K decoding takes them).
    /// </summary>
    private KernelVariant NativeFlashVariant(AttentionPlan plan, Func<Bindings, bool> when, long mixRun = 0)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        int warps = NativeAttentionWarps(Math.Max(plan.D, plan.DV));
        var macros = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in (string[])["HQ", "HKV", "D", "DV", "VRS", "QS0", "QS1", "SCALE", "WINDOW", "CHUNK", "HAS_SOFTCAP", "SOFTCAP", "HAS_SINKS",
                     "MULTI_SEQ", "SLOT_CTX", "PAGED", "PAGE_BS", "PAGE_MB", "RT_N_PAST"])
        {
            macros[name] = plan.Macros[name];
        }

        macros["MIX_RUN"] = mixRun.ToString(CultureInfo.InvariantCulture);
        macros["FN_WARPS"] = warps.ToString(CultureInfo.InvariantCulture);
        var kernel = _compiler.CompileCuda("attn_prefill_native.cu", "", "attn_prefill_native", warps * 32, NativeAttentionSharedBytes(plan), _opt.Target,
            macros, _opt.Cache);
        uint rows = (uint)(16 * warps);
        return runs is null
            ? new KernelVariant(when, kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), rows), 1))
            : new KernelVariant(when, kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), rows), (uint)Rows(b, runs)));
    }
}
