using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>Options for loading and compiling a language model with <see cref="LlmModel.Load"/>.</summary>
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
    /// <summary>Context window in tokens (prompt plus generated text) of each session; the KV cache is sized for it.</summary>
    public int ContextLength { get; init; } = 8192;
    /// <summary>Max tokens per forward pass (prefill chunk size).</summary>
    public int MaxBatchTokens { get; init; } = 512;
    /// <summary>
    /// Element type of the attention KV caches, overriding <see cref="AdapterOptions.KvCacheType"/> of <see cref="Adapter"/>: f16,
    /// f32, or the block-quantized q8_0 (about half the memory of f16) / q4_0 (about a quarter). Caches whose rows per head are not
    /// whole 32-value blocks stay f16. Null keeps the adapter setting (default f16).
    /// </summary>
    public DataType? KvCacheType { get; init; }
    /// <summary>Architecture-adapter settings (RoPE scaling, context overrides, ...).</summary>
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
    /// <summary>
    /// Choose tokens on the accelerator that computes the logits (default true): greedy and temperature / top-k / top-p / min-p sampling
    /// then download only token ids instead of the full logits. Sampling features the device path cannot express (repetition
    /// penalties, logit bias, alternatives for logprobs) still run on the CPU. Also disabled by <c>SHARPLIGHT_DEVICE_SAMPLING=0</c>.
    /// </summary>
    public bool DeviceSampling { get; init; } = true;
    /// <summary>Receives progress and compile-plan messages; null for silence.</summary>
    public Action<string>? Log { get; init; }
    /// <summary>
    /// Compilation artifacts to compile with: consulted before compiling, and recording everything compilation uses
    /// (<see cref="LlmModel.ExportCompilation"/>). Null (default) = one instance per model, seeded from the artifact sets stored in
    /// the model file. Share one instance (seeded from the file's <see cref="ModelSource.CompilationCaches"/>) to collect what
    /// several loads of the same model compile (e.g. its chat, embedding and reranking graphs) into one export.
    /// </summary>
    public CompilationArtifacts? Artifacts { get; init; }
}
