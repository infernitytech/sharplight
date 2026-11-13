namespace Sharplight.Cpu;

/// <summary>Row-wise operations supported by <see cref="RowOp"/>; the member names equal the graph op names.</summary>
internal enum RowKind { RmsNorm, L2Norm, LayerNorm, Softmax, ReduceSum, ReduceMax, ArgMax }
