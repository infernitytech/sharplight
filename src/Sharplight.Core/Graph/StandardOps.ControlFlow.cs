using Sharplight.Tensors;

namespace Sharplight.Graph;

// Control flow (see docs/dynamic-shapes.md): the subgraphs are separate ComputeGraphs held in attributes; the node runs in a host
// segment of its own that executes the subgraphs as nested compiled models.
public static partial class StandardOps
{
    /// <summary>
    /// Runtime conditional. Inputs: cond (one element; non-zero selects the then branch) followed by the captured values, which bind to the
    /// branch inputs named by <c>inputs</c> (string[]; a branch may leave some unused). Attributes: <c>then_branch</c>, <c>else_branch</c>
    /// (<see cref="ComputeGraph"/>), <c>outputs</c> (string[]: output names, the same in both branches), <c>output_symbols</c> (string[]:
    /// the dynamic symbol of the outer dimension of an output whose branches disagree on it, else empty). The outputs take the types and
    /// inner dimensions of the then branch.
    /// </summary>
    public static readonly OpSchema If = new("If", OpCategory.ControlFlow, 1, int.MaxValue, (i, a) =>
    {
        var then = a.Get<ComputeGraph>("then_branch");
        var symbols = a.Get<string[]>("output_symbols");
        return a.Get<string[]>("outputs").Select((name, k) =>
        {
            var v = then.Outputs[name];
            var shape = symbols[k].Length == 0 ? v.Shape : new Shape([Dim.Sym(symbols[k]), .. v.Shape.Dims.Skip(1)]);
            return new TensorType(v.Type, shape);
        }).ToArray();
    });
}
