using Sharplight.Graph;

namespace Sharplight.Onnx;

/// <summary>A translated runtime loop body (see <see cref="StandardOps.Loop"/>).</summary>
/// <param name="Graph">The body graph.</param>
/// <param name="Inputs">The <c>body_inputs</c> attribute: iteration, condition, carried, scan slices, captured (empty names for absent inputs).</param>
/// <param name="Outputs">The <c>body_outputs</c> attribute: condition (empty when absent), carried, scan outputs.</param>
/// <param name="Captured">The enclosing values the body reads, in input order.</param>
/// <param name="ScalarScans">Per scan output: the ONNX element is a scalar (the body produces it as <c>[1]</c>).</param>
internal sealed record LoopBody(ComputeGraph Graph, string[] Inputs, string[] Outputs, List<string> Captured, bool[] ScalarScans);
