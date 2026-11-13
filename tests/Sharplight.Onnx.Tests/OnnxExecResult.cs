using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Onnx.Tests;

/// <summary>Outputs and timings of one <see cref="OnnxExec.Run"/>.</summary>
/// <param name="Outputs">One array per graph output, in graph order.</param>
/// <param name="ImportMs">Time spent importing and translating the model.</param>
/// <param name="CompileMs">Time spent compiling the graph for the device.</param>
/// <param name="RunMs">Best timed run, or 0 when none was timed.</param>
/// <param name="Graph">The translated graph.</param>
/// <param name="Artifacts">The compilation artifacts the graph was compiled with, or null.</param>
internal sealed record OnnxExecResult(float[][] Outputs, double ImportMs, double CompileMs, double RunMs, ComputeGraph Graph, CompilationArtifacts? Artifacts);
