namespace Sharplight.Cpu;

/// <summary>Reductions supported by <see cref="ReduceOp"/>.</summary>
internal enum ReduceKind { Sum, Mean, Max, Min, Prod, L1, L2, SumSquare, LogSumExp }
