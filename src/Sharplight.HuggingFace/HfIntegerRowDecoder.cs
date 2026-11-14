namespace Sharplight.HuggingFace;

/// <summary>Decodes one row of a packed integer weight (see <see cref="HfIntegerLayout.DecodeRow"/>).</summary>
/// <param name="row">The output row.</param>
/// <param name="codes">Receives the row's K unsigned codes, in column order.</param>
/// <param name="scales">Receives each group's scale (exactly the stored value).</param>
/// <param name="zeros">Receives each group's zero point.</param>
internal delegate void HfIntegerRowDecoder(long row, Span<byte> codes, Span<float> scales, Span<float> zeros);
