namespace Sharplight.Llm.Serving.Hosting;

/// <param name="Domain">Where the memory is needed.</param>
/// <param name="Bytes">Estimated bytes (weights + state + activations).</param>
public sealed record MemoryRequirement(IMemoryDomain Domain, long Bytes);
