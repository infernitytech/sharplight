using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Tasks;

/// <summary>
/// How a model is loaded (<see cref="TaskModel.Load(ModelLoader, TaskLoadOptions, CancellationToken)"/>) and how its graphs are compiled:
/// the settings every model kind shares. Kind-specific loaders extend it (<c>LlmLoadOptions</c>, <c>EmbeddingLoadOptions</c>, ...).
/// Configured in code or through Microsoft.Extensions.Options (named after the service key when registered with
/// <c>services.AddTaskClient&lt;TClient&gt;(...)</c>).
/// </summary>
public class TaskLoadOptions
{
    /// <summary>Model file or directory, as understood by the registered importers (an <c>.slm</c> package, a GGUF file, ...).</summary>
    public string? ModelPath { get; set; }

    /// <summary>Devices in pipeline order (layers are split across them). At least one is required.</summary>
    public IList<DeviceSlot> Devices { get; set; } = [];

    /// <summary>Fallback device (e.g. CPU) for layers that do not fit and ops the devices cannot run.</summary>
    public DeviceSlot? Fallback { get; set; }

    /// <summary>Bytes kept free on each device for activations and scratch when placing a graph (null: 1 GiB).</summary>
    public long? ReservePerDevice { get; set; }

    /// <summary>
    /// Upper bound of the context symbol (cached positions) of graphs with persistent state, e.g. the KV cache of a decoder; 0 uses the
    /// model's own limit (text generation: 8192 at most).
    /// </summary>
    public int ContextLength { get; set; }

    /// <summary>
    /// Upper bound of the row symbol (tokens, frames, latent positions) per forward pass. Language models (<see cref="Llm.LlmLoadOptions"/>)
    /// take 0 as their default: rows chosen from the model's footprint and the free device memory (<see cref="Llm.LlmModel.BatchTokens"/>).
    /// </summary>
    public int MaxBatchTokens { get; set; } = 512;

    /// <summary>
    /// Auxiliary model files by role (<see cref="Vision.AuxiliaryRoles"/>), opened with the primary one (e.g. a separately shipped
    /// vision projector or VAE).
    /// </summary>
    public IDictionary<string, string> Auxiliary { get; set; } = new Dictionary<string, string>(StringComparer.Ordinal);

    /// <summary>Architecture-adapter settings (KV cache type, graph optimization, ...).</summary>
    public AdapterOptions Adapter { get; set; } = new();

    /// <summary>Device features (<see cref="DeviceFeatures"/>) not to use when compiling.</summary>
    public ISet<string> DisabledFeatures { get; set; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>Opt-in: let kernels trade precision for speed (see <see cref="CompileOptions.AllowReducedPrecision"/>).</summary>
    public bool AllowReducedPrecision { get; set; }

    /// <summary>
    /// Precompiled-artifact routing shared with other loads, or <see langword="null"/> to consult the artifact sets stored in the model
    /// package (see <see cref="CompilationArtifacts"/>).
    /// </summary>
    public CompilationArtifacts? Artifacts { get; set; }

    /// <summary>Receives progress and compile-plan messages; null for silence.</summary>
    public Action<string>? Log { get; set; }

    /// <summary>Receives load and compile progress; null for none.</summary>
    public IProgress<LoadProgress>? Progress { get; set; }
}
