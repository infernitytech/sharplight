using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Op dispatch. The lowerings themselves live in partial files by category: views, matmul, transformer building
/// blocks (norm, rope, gather), attention and caches, recurrent layers, elementwise codegen, spatial ops, general ops
/// and mixture-of-experts.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Scratch allocations of the node being lowered; freed as soon as the node is done.</summary>
    private readonly List<object> _temps = [];

    private void LowerNode(Node n)
    {
        LowerNodeOp(n);
        FreeTemps();
    }

    /// <summary>Releases the scratch allocations of the unit just lowered.</summary>
    private void FreeTemps()
    {
        foreach (var temp in _temps)
        {
            _alloc.Free(temp);
        }

        _temps.Clear();
    }

    private void LowerNodeOp(Node n)
    {
        switch (n.Op.Name)
        {
            case "Reshape":
                LowerReshape(n);
                break;
            case "Slice":
                _refs[n.Output] = n.Inputs[0].Shape[0].IsSymbolic && n.Attributes.Get<int>("axis") == 0
                    ? Ref(n.Inputs[0]) with { Shape = BoundShape(n.Output.Shape), Symbol = n.Output.Shape.Symbol }   // prefix of the symbolic axis
                    : ViewSlice(Ref(n.Inputs[0]), n.Attributes);
                break;
            case "Permute":
                _refs[n.Output] = ViewPermute(Ref(n.Inputs[0]), n.Attributes.Get<int[]>("perm"));
                break;
            case "Contiguous":
                EmitCopy(Ref(n.Inputs[0]), Produce(n.Output), "contiguous", [n]);
                break;
            case "Concat":
                LowerConcat(n);
                break;
            case "MatMul":
                LowerMatMul(n);
                break;
            case "MatMulId":
                LowerMatMulId(n);
                break;
            case "RmsNorm":
                LowerNorm(n, mode: 0);
                break;
            case "L2Norm":
                LowerNorm(n, mode: 1);
                break;
            case "Rope":
                LowerRope(n);
                break;
            case "Gather":
                LowerGather(n);
                break;
            case "CacheWrite":
                LowerCacheWrite(n);
                break;
            case "CachedAttention" or "Attention":
                LowerAttention(n);
                break;
            case "CausalConv1d":
                LowerConv(n);
                break;
            case "SequencePool":
                LowerSequencePool(n);
                break;
            case "NGramHash":
                LowerNGramHash(n);
                break;
            case "GatedDeltaRule":
                LowerDeltaRule(n);
                break;
            case "SelectiveScan":
                LowerSelectiveScan(n);
                break;
            case "Wkv7":
                LowerWkv7(n);
                break;
            case "Wkv6":
                LowerWkv6(n);
                break;
            case "Sample":
                LowerSample(n);
                break;
            default:
                LowerOtherOp(n);
                break;
        }
    }

    /// <summary>Ops outside the dedicated lowerings: general ops, standalone elementwise nodes, spatial ops, then detection ops.</summary>
    private void LowerOtherOp(Node n)
    {
        if (LowerGeneral(n))
        {
            return;
        }

        if (s_elementwise.Contains(n.Op.Name))
        {
            LowerElementwise([n], [n.Output]);
            return;
        }

        if (LowerSpatial(n))
        {
            return;
        }

        if (LowerDetection(n))
        {
            return;
        }

        throw new NotSupportedException($"No lowering for op {n.Op.Name}.");
    }
}
