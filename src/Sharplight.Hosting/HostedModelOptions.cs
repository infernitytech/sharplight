using System.Text.Json.Serialization;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting;

/// <summary>
/// One model of a host and its overrides; every nullable setting falls back to <see cref="ModelHostingOptions.Defaults"/>. Bound from a
/// <c>models</c> entry of the hosting JSON file or created in code (<see cref="IModelHostManager.Register"/>).
/// </summary>
public sealed class HostedModelOptions
{
    /// <summary>The name clients request (unique per host).</summary>
    public string Name { get; set; } = "";

    /// <summary>Model file or directory (<c>.slm</c>, <c>.gguf</c>, a Hugging Face checkpoint, ...).</summary>
    public string Path { get; set; } = "";

    /// <summary>What the model does; selects the loader (<see cref="Loaders.IHostedModelLoaderFactory"/>).</summary>
    [JsonConverter(typeof(JsonStringEnumConverter<ModelKind>))]
    public ModelKind Kind { get; set; } = ModelKind.Chat;

    /// <summary>
    /// For <see cref="ModelKind.Task"/> models: the task kind (<c>speech-recognition</c>, <c>speech-synthesis</c>, <c>image-generation</c>,
    /// ... as in <c>Sharplight.Models.ModelKinds</c>).
    /// </summary>
    public string? Task { get; set; }

    /// <summary>Where the model runs (see <see cref="HostedModelDefaults.Device"/>); null = the default.</summary>
    [JsonConverter(typeof(DeviceSpecificationJsonConverter))]
    public string? Device { get; set; }

    /// <summary>Context length per sequence.</summary>
    public int? ContextLength { get; set; }

    /// <summary>Maximum rows of one forward pass.</summary>
    public int? BatchTokens { get; set; }

    /// <summary>Concurrent sequences (chat models).</summary>
    public int? Slots { get; set; }

    /// <summary>Paged KV cache size in tokens (chat models).</summary>
    public long? KvCacheTokens { get; set; }

    /// <summary>Element type of the KV cache.</summary>
    public string? KvCacheType { get; set; }

    /// <summary>Vision projector file (chat models).</summary>
    public string? Projector { get; set; }

    /// <summary>Jinja chat template file that replaces the model's own (chat models).</summary>
    public string? ChatTemplate { get; set; }

    /// <summary>Further files by role (e.g. a vocoder or a VAE of a task model).</summary>
    public Dictionary<string, string>? Auxiliary { get; set; }

    /// <summary>Load on the first request instead of at startup; null = <see cref="ModelHostingOptions.Lazy"/>.</summary>
    public bool? Lazy { get; set; }

    /// <summary>Never evict the model for memory and never unload it when idle.</summary>
    public bool? Pinned { get; set; }

    /// <summary>Run layers that do not fit on the GPU on the CPU.</summary>
    public bool? CpuOffload { get; set; }

    /// <summary>Unload after this many idle seconds (zero or less: never); null = the host's timeout.</summary>
    public double? IdleTimeoutSeconds { get; set; }

    /// <summary>Free text shown in listings (default: the path).</summary>
    public string? Description { get; set; }
}
