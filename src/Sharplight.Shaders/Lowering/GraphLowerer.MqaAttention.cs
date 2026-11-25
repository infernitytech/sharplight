using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Query heads per key head from which shared-key-row attention runs on <c>fa_mqa</c>.</summary>
    private const long MqaMinGroup = 8;

    /// <summary>
    /// Keys per tile of <c>fa_mqa</c> (<c>FM_BC</c>); split ranges are multiples of it, and as short as one tile: a tile costs
    /// about as much as the kernel's fixed part, so decode rows spread over as many workgroups as the split limits allow.
    /// </summary>
    private const long MqaKeysPerTile = 32;

    /// <summary>
    /// Attention whose values are the first DV entries of the key rows and many query heads per key head (the absorbed
    /// multi-head latent attention cache: one [latent | rope] row per token for all heads): tensor-core <c>fa_mqa</c> for
    /// prefill and decode alike, with the key range of few-row launches split over workgroups at run time and merged by
    /// <c>decode_combine</c>. Returns null when the shapes or the device do not allow it (the generic kernels are the fallback).
    /// </summary>
    private DecodeAttentionPlan? MqaAttention(AttentionPlan plan)
    {
        long group = plan.HQ / plan.HKV;
        long paddedD = (plan.D + 15) / 16 * 16;
        long sharedBytes = 32 * (paddedD + 8) * 2 + 4 * 16 * 32 * 4 + 16 * 40 * 2 + 16 * 4;
        bool sharedRow = plan.K.Buffer == plan.V.Buffer && plan.K.ByteOffset == plan.V.ByteOffset && plan.V.Shape[2] == plan.D;
        bool typeAllowed = plan.K.Type == DataType.F16 || IsQuantizedKv(plan.K.Type) || (plan.K.Type == DataType.F32 && HalfActivations);
        bool alibi = plan.Node.Attributes.Has("alibi_slopes") || plan.Node.Attributes.Has("position_bias");   // the latent-attention kernel has no score bias
        if (plan.Full || alibi || !sharedRow || !typeAllowed || !DiagnosticSwitches.AttentionMqaFlash
            || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16) || !_opt.Has(DeviceFeatures.SubgroupOps) || _opt.SubgroupSize != 32
            || plan.HQ % plan.HKV != 0 || group < MqaMinGroup || plan.D % 8 != 0 || plan.DV > plan.D || sharedBytes > _opt.MaxSharedMemory)
        {
            return null;
        }

        int[] cfg = DiagnosticSwitches.AttentionDecodeConfig;
        long groups = plan.HKV * ((group + 15) / 16);
        long splitRows = plan.Seq is null ? SplitMaxRows : DecodeMaxSequenceRows;
        var macros = new Dictionary<string, string>(plan.Macros);
        (string Name, long Value)[] decodeMacros =
        [
            ("DEC_BC", MqaKeysPerTile), ("DEC_MAX_SPLITS", cfg[2]), ("DEC_MIN_KEYS", MqaKeysPerTile), ("DEC_TARGET_WG", cfg[0]),
            ("DEC_GROUPS", groups), ("DEC_SPLIT_ROWS", splitRows),
        ];
        foreach (var (name, value) in decodeMacros)
        {
            macros[name] = value.ToString(CultureInfo.InvariantCulture);
        }

        if (InvariantRows > 1)
        {
            macros["DEC_INVARIANT_ROWS"] = InvariantRows.ToString(CultureInfo.InvariantCulture);
        }

        var q = plan.Q;
        return new DecodeAttentionPlan
        {
            Macros = macros,
            Groups = groups,
            MaxRows = splitRows,
            TargetWorkgroups = cfg[0],
            MaxSplitCount = cfg[2],
            InvariantRows = InvariantRows,
            When = b => Rows(b, q) <= splitRows,
        };
    }

    /// <summary>
    /// The tensor-core decode form for head sizes 256 and 512 with separate key and value caches (Qwen3.5 / 3.8, Gemma 4's window and
    /// global layers) on targets without the hand-written CUDA kernel: <c>fa_mqa</c> with <c>FM_SEPV</c>, the decode rows' key and value
    /// tiles staged side by side, a group's heads (up to 16) as the rows of the cooperative-matrix products, so each key and value
    /// row is read once per KV head (f16 caches, or q8_0 / q4_0 caches whose 32-value blocks are converted to f16 as they are staged).
    /// The generic <c>decode</c> runs these rows on scalar lanes, whose registers hold 8 heads x 16
    /// values of queries and accumulators each at head size 512. Returns null where the shapes, the cache type or the device do not
    /// allow it (<c>decode</c> is the fallback).
    /// </summary>
    private DecodeAttentionPlan? TensorCoreWideDecode(AttentionPlan plan, long maxRows, Func<Bindings, bool>? flashWhen)
    {
        var a = plan.Node.Attributes;
        long group = plan.HQ / plan.HKV;
        int[] tc = DiagnosticSwitches.AttentionDecodeTensorCoreConfig ?? [];
        long keys = tc.Length > 3 ? tc[3] : plan.D > 256 ? 16 : 32;
        long ks = plan.D + 8;
        long sharedBytes = 2 * keys * ks * 2 + 4 * 16 * keys * 4 + 16 * (keys + 8) * 2 + 16 * 4 + 16 * 16 * 4;
        bool form = !plan.Full && !plan.Cross && plan.Spans is null && plan.Mask is null && plan.Selection is null && plan.Compressed is null
                    && plan.RelativeBias is null && !a.Has("alibi_slopes") && !a.Has("position_bias");
        if (!form || !DiagnosticSwitches.AttentionDecodeTensorCores || NativeWideDecodeAttention(plan) || plan.D != plan.DV || plan.D is not (256 or 512)
            || (plan.K.Type != DataType.F16 && !IsQuantizedKv(plan.K.Type)) || plan.V.Type != plan.K.Type || plan.HQ % plan.HKV != 0 || group > 16
            || keys is not (16 or 32)
            || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16) || !_opt.Has(DeviceFeatures.SubgroupOps) || _opt.SubgroupSize != 32
            || sharedBytes > _opt.MaxSharedMemory)
        {
            return null;
        }

        // Split rule: ranges that fill whole waves of the compute units (when the device reports them), at least two tiles each.
        long waves = tc.Length > 0 ? tc[0] : 2;
        long target = _opt.ComputeUnits > 0 ? _opt.ComputeUnits * waves : DiagnosticSwitches.AttentionDecodeConfig[0];
        long minKeys = tc.Length > 1 ? tc[1] : 2 * keys;
        long maxSplits = tc.Length > 2 ? tc[2] : DiagnosticSwitches.AttentionDecodeConfig[2];
        long groups = plan.HKV;
        var macros = new Dictionary<string, string>(plan.Macros);
        (string Name, long Value)[] decodeMacros =
        [
            ("FM_SEPV", 1), ("FM_BC", keys), ("HPW", group), ("DEC_BC", keys), ("DEC_MAX_SPLITS", maxSplits), ("DEC_MIN_KEYS", minKeys),
            ("DEC_TARGET_WG", target), ("DEC_GROUPS", groups),
        ];
        foreach (var (name, value) in decodeMacros)
        {
            macros[name] = value.ToString(CultureInfo.InvariantCulture);
        }

        if (InvariantRows > 1)
        {
            macros["DEC_INVARIANT_ROWS"] = InvariantRows.ToString(CultureInfo.InvariantCulture);
        }

        var q = plan.Q;
        long window = long.Parse(plan.Macros.GetValueOrDefault("WINDOW", "0"), CultureInfo.InvariantCulture);
        return new DecodeAttentionPlan
        {
            Macros = macros,
            Groups = groups,
            MaxRows = maxRows,
            TargetWorkgroups = target,
            MaxSplitCount = maxSplits,
            InvariantRows = InvariantRows,
            SingleRange = window > 0 && window <= minKeys,
            Kernel = "fa_mqa",
            When = b => Rows(b, q) <= maxRows && (flashWhen is null || !flashWhen(b)),
        };
    }

    /// <summary>Launches <c>fa_mqa</c> for every row count and the combine of the split rows.</summary>
    private void EmitMqaAttention(AttentionPlan plan, DecodeAttentionPlan mqa)
    {
        var q = plan.Q;
        long maxRows = q.Symbol is null ? q.Shape[0] : Bound(Dim.Sym(q.Symbol));
        var part = DecodeScratch(plan, mqa, maxRows);
        var kernel = Lib("attention.slang", "fa_mqa", mqa.Macros);
        List<KernelVariant> variants = [Always(kernel, b => new Grid((uint)mqa.Groups, (uint)Rows(b, q), (uint)mqa.MaxSplits(Rows(b, q))))];
        EmitAttentionLaunch(plan, variants, part, 0, split: true);
        EmitDecodeCombine(plan, part, mqa);
    }
}
