using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>Options for loading and compiling a language model.</summary>
public sealed class LlmLoadOptions
{
    /// <summary>Devices in pipeline order (layers are split across them). Required.</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }
    /// <summary>Fallback device (e.g. CPU) for layers that don't fit and unsupported ops.</summary>
    public DeviceSlot? Fallback { get; init; }
    /// <summary>
    /// Bytes kept free on each device for activations and scratch when splitting layers (null = placement default, 1 GiB).
    /// The serving host sets it from <see cref="LlmModel.EstimateFootprint"/> together with a memory limit.
    /// </summary>
    public long? ReservePerDevice { get; init; }
    public int ContextLength { get; init; } = 8192;
    /// <summary>Max tokens per forward pass (prefill chunk size).</summary>
    public int MaxBatchTokens { get; init; } = 512;
    public AdapterOptions Adapter { get; init; } = new();
    /// <summary>Device features (<see cref="DeviceFeatures"/>) not to use when compiling this model, e.g. "matrix_f16".</summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();
    /// <summary>Auxiliary model files, e.g. ("mmproj", path) for a vision projector.</summary>
    public IReadOnlyList<(string Role, string Path)> Auxiliary { get; init; } = [];
    /// <summary>
    /// Image token budget (output tokens per image): the vision encoder is compiled for at most
    /// <c>MaxTokens</c> (default 1024) tokens per image and images are resized to fit; <c>MinTokens</c> upsizes small images.
    /// </summary>
    public ImageTokenLimits? ImageTokens { get; init; }
    public Action<string>? Log { get; init; }
}
