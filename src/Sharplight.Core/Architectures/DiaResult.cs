namespace Sharplight.Architectures;

/// <summary>The decoder's sequence of one Dia generation.</summary>
/// <param name="Rows">The rows of the delayed sequence (the start row, the prompt rows, the generated rows), one token per codebook.</param>
/// <param name="Start">The first generated frame (the frames of an audio prompt come before it).</param>
/// <param name="End">The frame after the last generated one (the row where the first codebook ended).</param>
internal sealed record DiaResult(IReadOnlyList<int[]> Rows, int Start, int End);
