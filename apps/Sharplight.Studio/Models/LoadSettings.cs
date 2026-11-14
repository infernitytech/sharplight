namespace Sharplight.Studio.Models;

/// <summary>How a model is loaded: device, context, batching, KV cache and speculation.</summary>
internal sealed record LoadSettings
{
    /// <summary>Default context length in tokens.</summary>
    public const int DefaultContextLength = 8192;

    /// <summary>The device id (<c>provider:index</c>, e.g. <c>vulkan:0</c> or <c>cpu:0</c>); null picks the first GPU, else the CPU.</summary>
    public string? DeviceId { get; init; }

    /// <summary>Context length per sequence in tokens.</summary>
    public int ContextLength { get; init; } = DefaultContextLength;

    /// <summary>Maximum rows per forward pass (prompt chunk size).</summary>
    public int BatchTokens { get; init; } = 512;

    /// <summary>KV cache element type: <c>f16</c>, <c>f32</c>, <c>q8_0</c> or <c>q4_0</c>.</summary>
    public string KvCacheType { get; init; } = "f16";

    /// <summary>Concurrent sequences batched together (chat plus API requests).</summary>
    public int Slots { get; init; } = 2;

    /// <summary>Run layers that do not fit, and ops the device cannot execute, on the CPU.</summary>
    public bool CpuOffload { get; init; }

    /// <summary>Draft length of prompt-lookup (n-gram) speculative decoding; 0 disables it.</summary>
    public int NgramDraftTokens { get; init; }

    /// <summary>Draft length of multi-token-prediction speculation with the model's own draft head; 0 disables it (ignored without a head).</summary>
    public int MtpDraftTokens { get; init; }

    /// <summary>Opt-in: let kernels trade precision for speed (f16 products where the default keeps near-f32 accuracy).</summary>
    public bool AllowReducedPrecision { get; init; }
}
