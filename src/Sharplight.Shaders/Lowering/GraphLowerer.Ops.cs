using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    private readonly List<object> _temps = [];

    private void LowerNode(Node n)
    {
        switch (n.Op.Name)
        {
            case "Reshape": LowerReshape(n); break;
            case "Slice": _refs[n.Output] = ViewSlice(Ref(n.Inputs[0]), n.Attributes); break;
            case "Permute": _refs[n.Output] = ViewPermute(Ref(n.Inputs[0]), n.Attributes.Get<int[]>("perm")); break;
            case "Contiguous": EmitCopy(Ref(n.Inputs[0]), Produce(n.Output), "contiguous", [n]); break;
            case "Concat": LowerConcat(n); break;
            case "MatMul": LowerMatMul(n); break;
            case "MatMulId": LowerMatMulId(n); break;
            case "RmsNorm": LowerNorm(n, mode: 0); break;
            case "L2Norm": LowerNorm(n, mode: 1); break;
            case "Rope": LowerRope(n); break;
            case "Gather": LowerGather(n); break;
            case "CacheWrite": LowerCacheWrite(n); break;
            case "CachedAttention" or "Attention": LowerAttention(n); break;
            case "CausalConv1d": LowerConv(n); break;
            case "SequencePool": LowerSequencePool(n); break;
            case "GatedDeltaRule": LowerDeltaRule(n); break;
            case "SelectiveScan": LowerSelectiveScan(n); break;
            default:
                if (LowerGeneral(n))
                {
                    break;
                }

                if (s_elementwise.Contains(n.Op.Name))
                {
                    LowerElementwise([n], [n.Output]);
                    break;
                }
                if (LowerSpatial(n))
                {
                    break;
                }

                throw new NotSupportedException($"No lowering for op {n.Op.Name}.");
        }
        foreach (var t in _temps)
        {
            _alloc.Free(t);
        }

        _temps.Clear();
    }
}
