using Sharplight.Tensors;

namespace Sharplight.Graph;

// Runtime loop (see docs/dynamic-shapes.md, "Runtime control flow"): the body is a separate ComputeGraph held in an attribute; the node
// runs in a host segment of its own (LoopSegment) that executes the body as a nested compiled model once per iteration.
public static partial class StandardOps
{
    /// <summary>
    /// Runtime loop with loop-carried values and scan inputs/outputs.
    /// <para>
    /// Inputs: <c>trip</c> (one element: the maximum number of iterations), <c>cond</c> (one element; zero runs no iteration), the S initial
    /// loop-carried values, the I scan inputs (iteration <c>t</c> reads slice <c>t</c> of their outermost axis) and the captured values the
    /// body reads from the enclosing graph.
    /// </para>
    /// <para>
    /// Attributes: <c>body</c> (<see cref="ComputeGraph"/>); <c>body_inputs</c> (string[]: the body inputs [iteration number (i32
    /// <c>[1]</c>), condition, S carried, I scan slices, captured...]; entry <c>k</c> ≥ 1 receives node input <c>k</c>; a body may lack
    /// inputs it does not read); <c>body_outputs</c> (string[]: the continue condition — empty when the body has none — the S carried values, then the scan
    /// outputs);
    /// <c>state_count</c> and <c>scan_input_count</c> (long); <c>scan_input_reverse</c> and <c>scan_output_reverse</c> (long[]: 1 walks
    /// that scan input or fills that scan output from the last slice to the first);
    /// <c>trip_symbol</c> (string: when non-empty, a symbol expression that also limits the iterations); <c>scan_extent</c> (string: the
    /// outermost extent of the scan outputs, a number or symbol expression) and <c>count_symbol</c> (string: a dynamic symbol of the graph
    /// that the node binds to the number of iterations it ran, or empty).
    /// </para>
    /// <para>
    /// Outputs: the S final carried values (the types of the initial values, which the body keeps), then the scan outputs: the per-iteration
    /// body outputs stacked along a new outermost axis of extent <c>scan_extent</c>.
    /// </para>
    /// </summary>
    public static readonly OpSchema Loop = new("Loop", OpCategory.ControlFlow, 2, int.MaxValue, (i, a) =>
    {
        var body = a.Get<ComputeGraph>("body");
        var outputs = a.Get<string[]>("body_outputs");
        int states = (int)a.Get<long>("state_count");
        var extent = Dim.Sym(a.Get<string>("scan_extent"));
        var types = new List<TensorType>();
        for (int k = 0; k < states; k++)
        {
            types.Add(new TensorType(i[2 + k].Type, i[2 + k].Shape));
        }

        foreach (var name in outputs.Skip(1 + states))
        {
            var element = body.Outputs[name];
            types.Add(new TensorType(element.Type, new Shape([extent, .. element.Shape.Dims])));
        }

        return types;
    });
}
