using System.Text.Json.Serialization;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Tasks;

namespace Sharplight.Hosting;

/// <summary>
/// What a <see cref="IModelHostManager"/> hosts and its memory and lifetime policy: the models, the defaults they fall back to, the
/// memory kept free on every device, a total budget, the resident model limit (least recently used idle models are evicted first) and
/// the idle timeout. Configure through <c>services.AddSharplightHosting(...)</c>, or read the JSON form with
/// <see cref="ModelHostingJson"/>.
/// </summary>
public sealed class ModelHostingOptions
{
    /// <summary>Device memory kept free for other applications on every device, in bytes (JSON: a number or <c>"2G"</c>).</summary>
    [JsonConverter(typeof(MemorySizeJsonConverter))]
    public long Reserve { get; set; } = 1L << 30;

    /// <summary>Upper bound on the memory used on each GPU, in bytes; null = what the driver reports as available.</summary>
    [JsonConverter(typeof(NullableMemorySizeJsonConverter))]
    public long? Budget { get; set; }

    /// <summary>At most this many models resident at once (0 = no limit); loading another evicts the least recently used idle one.</summary>
    public int MaxLoadedModels { get; set; }

    /// <summary>Unload models that served nothing for this many seconds (they reload on demand); null or 0 = never.</summary>
    public double? IdleTimeoutSeconds { get; set; }

    /// <summary>Models without their own <see cref="HostedModelOptions.Lazy"/> load on their first request instead of at startup.</summary>
    public bool Lazy { get; set; }

    /// <summary>Load the startup models when the application starts (the hosted service of <c>AddSharplightHosting</c>).</summary>
    public bool LoadStartupModels { get; set; } = true;

    /// <summary>
    /// Ask the registered <see cref="IHostedModelResolver"/> for names that are not registered (on-demand models, e.g. every model of a
    /// local store), and list the models it knows up front.
    /// </summary>
    public bool OnDemand { get; set; }

    /// <summary>Device features no model may use (see <c>DeviceFeatures</c>); for fallback testing and broken drivers.</summary>
    public List<string> DisabledFeatures { get; set; } = [];

    /// <summary>Settings of the models that do not set their own.</summary>
    public HostedModelDefaults Defaults { get; set; } = new();

    /// <summary>The models, in registration order (the first chat model is the default one).</summary>
    public List<HostedModelOptions> Models { get; set; } = [];

    /// <summary>Receives load, unload and eviction messages (default: the <c>ILogger</c> of the manager when logging is registered).</summary>
    [JsonIgnore]
    public Action<string>? Log { get; set; }

    /// <summary>Adjusts the load options of each model before it loads (progress reporting, logging, compilation artifacts, ...).</summary>
    [JsonIgnore]
    public Action<HostedModelSettings, TaskLoadOptions>? ConfigureLoad { get; set; }

    /// <summary>Adjusts the options of each chat model before it loads (request callbacks, speculators, service options, ...).</summary>
    [JsonIgnore]
    public Action<HostedModelSettings, ChatModelOptions>? ConfigureChat { get; set; }

    /// <summary>The idle timeout as a span, or null.</summary>
    internal TimeSpan? IdleTimeout => IdleTimeoutSeconds is > 0 ? TimeSpan.FromSeconds(IdleTimeoutSeconds.Value) : null;
}
