namespace Sharplight.FeedTool;

/// <summary>
/// Per-slot memory of a decoder's state from its decode graph (PlacementPlanner estimates, no device): the KV bytes per token of the
/// layers whose cache grows with the context, the state that does not (recurrent layers, saturated sliding windows) and the exact state
/// at the default and the maximum context.
/// </summary>
/// <param name="KvBytesPerToken">Bytes per context token of one slot (16-bit cache).</param>
/// <param name="StateBytesPerSlot">Bytes of one slot that do not grow with the context.</param>
/// <param name="DefaultTokens">The default context length.</param>
/// <param name="DefaultBytes">State bytes of one slot at <paramref name="DefaultTokens"/>.</param>
/// <param name="MaxTokens">The maximum context length.</param>
/// <param name="MaxBytes">State bytes of one slot at <paramref name="MaxTokens"/>.</param>
/// <param name="ActivationBytes">Activation and scratch estimate for a prefill batch.</param>
internal sealed record ContextMemory(long KvBytesPerToken, long StateBytesPerSlot, int DefaultTokens, long DefaultBytes, int MaxTokens, long MaxBytes, long ActivationBytes);
