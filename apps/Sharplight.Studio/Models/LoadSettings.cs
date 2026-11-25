using Sharplight.Llm.Serving;

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

    /// <summary>KV cache element type: <c>f16</c>, <c>f32</c>, <c>q8_0</c>, <c>f8e4m3_b32</c> or <c>q4_0</c>.</summary>
    public string KvCacheType { get; init; } = "f16";

    /// <summary>
    /// Concurrent sequences batched together (chat plus API requests). One by default: a single-sequence session, whose window-sized caches
    /// make long contexts of sliding-window models fit; more serve concurrent API clients at the cost of a cache per slot.
    /// </summary>
    public int Slots { get; init; } = 1;

    /// <summary>Run layers that do not fit, and ops the device cannot execute, on the CPU.</summary>
    public bool CpuOffload { get; init; }

    /// <summary>
    /// With <see cref="CpuOffload"/>: routed experts of mixture-of-experts models that fit neither the GPU nor system memory stay on disk and are
    /// read on demand (models larger than RAM + VRAM, slow: about 1 token/s on a fast NVMe drive).
    /// </summary>
    public bool ExpertsOnDisk { get; init; }

    /// <summary>Draft length of prompt-lookup (n-gram) speculative decoding; 0 disables it.</summary>
    public int NgramDraftTokens { get; init; }

    /// <summary>Draft length of multi-token-prediction speculation with the model's own draft head; 0 disables it (ignored without a head).</summary>
    public int MtpDraftTokens { get; init; }

    /// <summary>Opt-in: let kernels trade precision for speed (f16 products where the default keeps near-f32 accuracy).</summary>
    public bool AllowReducedPrecision { get; init; }

    /// <summary>
    /// Block-scaled FP8 (default on): on GPUs whose tensor cores apply a power-of-two scale per 32 values, block-scaled E4M3 weights are
    /// requantized to such scales at load and prompt-sized products run on the block-scaled MMA. Off keeps the checkpoint's f32 scales.
    /// Ignored for models without such weights and on other devices.
    /// </summary>
    public bool BlockScaledFp8 { get; init; } = true;

    /// <summary>
    /// FP4 activations (default on): on GPUs with the block-scaled E2M1 MMA, prompt-sized products of NVFP4 weights quantize their
    /// activations to FP4 per 16 values and run on the FP4 GEMM (W4A4). Off keeps 16-bit activations. Ignored for models without NVFP4 weights
    /// and on other devices.
    /// </summary>
    public bool Fp4Activations { get; init; } = true;

    /// <summary>When the floor of the engine's load-adaptive step budget applies, or null for the engine's default (several slots only).</summary>
    public StepFloorPolicy? StepFloor { get; init; }

    /// <summary>Rows per sequence holding a slot that the adaptive step budget allows, or null for the engine's default (several slots only).</summary>
    public int? StepRowsPerSequence { get; init; }

    /// <summary>What happens to new requests when the KV cache is full, or null for the engine's default, queueing (several slots only).</summary>
    public KvAdmissionPolicy? KvAdmission { get; init; }
}
