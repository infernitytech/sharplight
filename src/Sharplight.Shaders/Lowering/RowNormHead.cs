namespace Sharplight.Shaders.Lowering;

/// <summary>
/// A row norm in front of a fused elementwise group whose kernel also ends in a row norm (<see cref="RowNormKernelSource.HeadNormExpression"/>).
/// </summary>
/// <param name="Expression">The head norm's input expression in the generated kernel.</param>
/// <param name="Weight">Its per-column f32 weight, or null.</param>
/// <param name="Mode">0 for RMS norm, 1 for L2 norm.</param>
/// <param name="Epsilon">Its epsilon.</param>
internal sealed record RowNormHead(string Expression, TensorRef? Weight, int Mode, float Epsilon);
