namespace Sharplight.Architectures.SeedVr;

/// <summary>Equally sized attention windows packed into one batched attention.</summary>
/// <param name="Count">Windows in the group.</param>
/// <param name="Sequence">Rows per window: its video tokens followed by the text tokens.</param>
/// <param name="Gather">The source row of every packed row (window-major), in the rows [video tokens..., text tokens...].</param>
internal sealed record WindowGroup(int Count, int Sequence, int[] Gather);
