using Sharplight.Graph;

namespace Sharplight.Cpu;

// Node dispatch: each graph op is lowered either to a zero-copy view or to one executable step.
internal sealed unsafe partial class CpuCompiledSegment
{
    // Elementwise op names equal the enum member names. Scale and Clip carry their own attributes and have dedicated lowerings.
    private static readonly Dictionary<string, UnaryKind> s_unaryKinds = Enum.GetValues<UnaryKind>()
        .Where(k => k is not (UnaryKind.Scale or UnaryKind.Clip))
        .ToDictionary(k => k.ToString(), StringComparer.Ordinal);

    private static readonly Dictionary<string, BinaryKind> s_binaryKinds = Enum.GetValues<BinaryKind>()
        .ToDictionary(k => k.ToString(), StringComparer.Ordinal);

    private static readonly Dictionary<string, RowKind> s_rowKinds = Enum.GetValues<RowKind>()
        .ToDictionary(k => k.ToString(), StringComparer.Ordinal);

    private void Lower(Node n, CpuDevice device)
    {
        string op = n.Op.Name;
        if (n.Op.Category != OpCategory.View)
        {
            var support = device.Supports(n);
            if (!support.IsSupported)
            {
                throw new NotSupportedException($"CPU provider cannot run {op} (node {n.Id}): {support.Reason}");
            }
        }

        switch (op)
        {
            case "Reshape":
                LowerReshape(n);
                return;
            case "Slice":
                LowerSlice(n);
                return;
            case "Permute":
                LowerPermute(n);
                return;
            case "Expand":
                LowerExpand(n);
                return;
            case "Tile":
                LowerTile(n);
                return;
            case "Concat":
                LowerConcat(n);
                return;
            case "Cast":
                LowerCast(n);
                return;
            case "Contiguous" or "Transfer":
                LowerCopy(n);
                return;

            case "Clip":
                LowerClip(n);
                return;
            case "Scale":
                LowerScale(n);
                return;
            case "Where":
                LowerWhere(n);
                return;

            case "Conv":
                LowerConv(n);
                return;
            case "ConvTranspose":
                LowerConvTranspose(n);
                return;
            case "Pool":
                LowerPool(n);
                return;
            case "Reduce":
                LowerReduce(n);
                return;
            case "Pad":
                LowerPad(n);
                return;
            case "Resize":
                LowerResize(n);
                return;
            case "GridSample":
                LowerGridSample(n);
                return;
            case "GroupNorm":
                LowerGroupNorm(n);
                return;
            case "RoiAlign":
                LowerRoiAlign(n);
                return;

            case "GatherAxis":
                LowerGatherAxis(n);
                return;
            case "GatherElements":
                LowerGatherElements(n);
                return;
            case "GatherND":
                LowerGatherNd(n);
                return;
            case "ScatterElements" or "ScatterND":
                LowerScatter(n);
                return;
            case "SequencePool":
                LowerSequencePool(n);
                return;
            case "NGramHash":
                LowerNGramHash(n);
                return;
            case "CumSum":
                LowerCumSum(n);
                return;
            case "ArgReduce":
                LowerArgReduce(n);
                return;
            case "TopK":
                LowerTopK(n);
                return;
            case "Sample":
                LowerSample(n);
                return;
            case "Trilu":
                LowerTrilu(n);
                return;
            case "OneHot":
                LowerOneHot(n);
                return;

            case "Gather":
                LowerGather(n);
                return;
            case "MatMul":
                LowerMatMul(n);
                return;
            case "MatMulId":
                LowerMatMulId(n);
                return;
            case "BatchMatMul":
                LowerBatchMatMul(n);
                return;

            case "Rope":
                LowerRope(n);
                return;
            case "CacheWrite":
                LowerCacheWrite(n);
                return;
            case "CachedAttention":
                LowerCachedAttention(n);
                return;
            case "Attention":
                LowerAttention(n);
                return;

            case "CausalConv1d":
                LowerCausalConv1d(n);
                return;
            case "SelectiveScan":
                LowerSelectiveScan(n);
                return;
            case "GatedDeltaRule":
                LowerGatedDeltaRule(n);
                return;
            case "Wkv7":
                LowerWkv7(n);
                return;
            case "Wkv6":
                LowerWkv6(n);
                return;
            case "Fill":
                LowerFill(n);
                return;
            case "NonZero":
                LowerNonZero(n);
                return;
            case "NonMaxSuppression":
                LowerNonMaxSuppression(n);
                return;
            case "Unique":
                LowerUnique(n);
                return;
            case "Dft":
                LowerDft(n);
                return;
            case "Random":
                LowerRandom(n);
                return;
            case "Multinomial":
                LowerMultinomial(n);
                return;
            case "TfIdf":
                LowerTfIdf(n);
                return;
            case "TreeEnsemble":
                LowerTreeEnsemble(n);
                return;
            case "OneVsOne":
                LowerOneVsOne(n);
                return;
            case "LabelEncode":
                LowerLabelEncode(n);
                return;
        }

        if (s_unaryKinds.TryGetValue(op, out var unaryKind))
        {
            LowerUnary(n, unaryKind);
        }
        else if (s_binaryKinds.TryGetValue(op, out var binaryKind))
        {
            LowerBinary(n, binaryKind);
        }
        else if (s_rowKinds.TryGetValue(op, out var rowKind))
        {
            LowerRow(n, rowKind);
        }
        else
        {
            throw new NotSupportedException($"Op {op} is not implemented by the CPU provider.");
        }
    }
}
