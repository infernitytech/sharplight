namespace Sharplight.FeedTool;

/// <summary>
/// The routed experts of a mixture-of-experts decoder, from its decode graph: the bytes of the expert banks a token selects only part of, and the
/// bytes of the experts one token selects over every layer (the manifest's <c>expertWeights</c> and <c>expertBytesPerToken</c>).
/// </summary>
/// <param name="Bytes">Bytes of the routed expert banks.</param>
/// <param name="BytesPerToken">Bytes of the experts one token selects.</param>
internal sealed record ExpertMemory(long Bytes, long BytesPerToken);
