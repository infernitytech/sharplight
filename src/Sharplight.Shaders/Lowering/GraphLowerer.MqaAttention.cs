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
        bool typeAllowed = plan.K.Type == DataType.F16 || (plan.K.Type == DataType.F32 && _opt.Compile.AllowReducedPrecision);
        if (plan.Full || !sharedRow || !typeAllowed || !DiagnosticSwitches.AttentionMqaFlash
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

        var q = plan.Q;
        return new DecodeAttentionPlan
        {
            Macros = macros,
            Groups = groups,
            MaxRows = splitRows,
            TargetWorkgroups = cfg[0],
            MaxSplitCount = cfg[2],
            When = b => Rows(b, q) <= splitRows,
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
