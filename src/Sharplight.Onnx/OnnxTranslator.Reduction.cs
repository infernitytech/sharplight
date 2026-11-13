using Sharplight.Graph;

namespace Sharplight.Onnx;

// Reductions, arg-reductions, TopK and cumulative sums.
internal sealed partial class OnnxTranslator
{
    private static readonly Dictionary<string, string> s_reduceModes = new()
    {
        ["ReduceMean"] = "mean",
        ["ReduceSum"] = "sum",
        ["ReduceMax"] = "max",
        ["ReduceMin"] = "min",
        ["ReduceProd"] = "prod",
        ["ReduceL1"] = "l1",
        ["ReduceL2"] = "l2",
        ["ReduceSumSquare"] = "sumsquare",
        ["ReduceLogSumExp"] = "logsumexp",
    };

    private void Reduce(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long[]? axes = n.Has("axes") ? n.Ints("axes") : OptConst(ins, 1, "reduction axes")?.Ints();
        bool keepDims = n.Int("keepdims", 1) != 0;
        if (axes is null || axes.Length == 0)
        {
            if (n.Int("noop_with_empty_axes", 0) != 0 && axes is not null)
            {
                SetDyn(n, 0, x);
                return;
            }

            axes = Enumerable.Range(0, x.Rank).Select(i => (long)i).ToArray();
        }

        var normalizedAxes = axes.Select(a => a < 0 ? a + x.Rank : a).ToArray();
        Attributes ReduceAttributes(string mode) => Attributes.From(("mode", mode), ("axes", normalizedAxes), ("keepdims", keepDims));

        string name = Nm(n);
        if (n.OpType == "ReduceLogSum")
        {
            var sum = Emit("Reduce", [x], ReduceAttributes("sum"));
            SetDyn(n, 0, Emit("Log", [sum], null, name));
            return;
        }

        if (!s_reduceModes.TryGetValue(n.OpType, out var irMode))
        {
            throw new NotSupportedException(n.OpType);
        }

        SetDyn(n, 0, Emit("Reduce", [x], ReduceAttributes(irMode), name));
    }

    /// <summary>One-hot of the arg-max along an axis (computed on the axis moved to the end).</summary>
    private void Hardmax(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", Opset >= 13 ? -1 : 1), x.Rank);
        var perm = Enumerable.Range(0, x.Rank).ToArray();
        (perm[axis], perm[^1]) = (perm[^1], perm[axis]);
        var xp = Permute(x, perm);
        var idx = Emit("ArgReduce", [xp], Attributes.From(("mode", "max"), ("axis", (long)(x.Rank - 1)), ("keepdims", false)));
        var oh = Emit("OneHot", [idx], Attributes.From(("depth", xp.Shape[-1].Fixed), ("axis", -1L), ("on", 1f), ("off", 0f)));
        SetDyn(n, 0, Permute(oh, perm));
    }

    private void ArgReduce(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int axis = NormalizeAxis(n.Int("axis", 0), x.Rank);
        var attributes = Attributes.From(("mode", n.OpType == "ArgMax" ? "max" : "min"), ("axis", (long)axis),
            ("keepdims", n.Int("keepdims", 1) != 0), ("select_last_index", n.Int("select_last_index", 0) != 0));
        SetDyn(n, 0, Emit("ArgReduce", [x], attributes, Nm(n)));
    }

    private void TopK(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        long k = n.Has("k") ? n.Int("k", 1) : ConstIn(ins, 1, "TopK k", symbolic: true).Ints()[0];
        var attributes = Attributes.From(("axis", n.Int("axis", -1)), ("largest", n.Int("largest", 1) != 0), ("sorted", n.Int("sorted", 1) != 0));
        attributes = SymbolicInts.IsSymbolic(k) ? attributes.With("k_symbol", SymbolicInts.Format(k)) : attributes.With("k", k);
        var outputs = _b.Emit("TopK", [x], attributes, Nm(n));
        SetDyn(n, 0, outputs[0]);
        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, outputs[1]);
        }
    }

    private void CumSum(OnnxNode n, TranslatedTensor?[] ins)
    {
        var attributes = Attributes.From(("axis", ConstIn(ins, 1, "CumSum axis").ScalarInt()), ("exclusive", n.Int("exclusive", 0) != 0), ("reverse", n.Int("reverse", 0) != 0));
        SetDyn(n, 0, Emit("CumSum", [X(ins, 0)], attributes, Nm(n)));
    }
}
