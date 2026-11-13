using System.Text.Json.Serialization;

namespace Sharplight.Serve;

/// <summary>
/// One served model and its per-model overrides. Every nullable property falls back to the matching global option when
/// unset. Bound from <c>--model NAME=PATH,key=value</c> or from a <c>models</c> entry of the JSON configuration file.
/// </summary>
internal sealed class ModelSpec
{
    /// <summary>Model id clients send in the <c>model</c> field. Unique across all specs.</summary>
    public string Name { get; set; } = "";

    /// <summary>Model file or directory (<c>.gguf</c>, <c>.slm</c> or a Hugging Face checkpoint).</summary>
    public string Path { get; set; } = "";

    /// <summary>Context length per sequence.</summary>
    public int? Ctx { get; set; }

    /// <summary>Concurrent sequences batched per forward pass.</summary>
    public int? Slots { get; set; }

    /// <summary>Maximum rows per forward pass.</summary>
    public int? Batch { get; set; }

    /// <summary>GPU index, or <c>cpu</c> to run this model on the CPU.</summary>
    [JsonConverter(typeof(DeviceNameConverter))]
    public string? Device { get; set; }

    /// <summary>Vision projector file.</summary>
    public string? Mmproj { get; set; }

    /// <summary>Jinja chat template file that replaces the model's own.</summary>
    public string? Template { get; set; }

    /// <summary>Load on the first request instead of at startup.</summary>
    public bool? Lazy { get; set; }

    /// <summary>Never evict this model to make room for another.</summary>
    public bool? Pin { get; set; }

    /// <summary>Run layers that do not fit on the device, and unsupported ops, on the CPU.</summary>
    public bool? Offload { get; set; }

    /// <summary>Unload after this many idle seconds (zero or less: never).</summary>
    public double? Idle { get; set; }
}
