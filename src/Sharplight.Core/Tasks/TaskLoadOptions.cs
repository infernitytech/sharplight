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

    /// <summary>
    /// Opt-in: with a CPU <see cref="Fallback"/>, routed experts that fit neither the devices nor host memory stay in the model file and the
    /// CPU reads the ones each pass selects into a cache (<see cref="Runtime.Experts.ExpertStore"/>), so mixture-of-experts models larger
    /// than RAM and VRAM together run (decode at the disk's speed; a fast NVMe drive gives about 1 token/s on models of several hundred GB).
    /// The results equal those of resident experts. Without it such a model pages its mapped experts in and out (slow) or does not load.
    /// </summary>
    public bool ExpertsOnDisk { get; set; }

    /// <summary>
    /// With <see cref="ExpertsOnDisk"/>: bytes of host memory for the cache of experts read from disk, or null to size it from the free host
    /// memory. A cache smaller than the experts on the CPU leaves them on disk even when they would fit.
    /// </summary>
    public long? ExpertCacheBytes { get; set; }

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

    /// <summary>
    /// The block-scaled FP8 mode (default on): where the device multiplies E4M3 operands with a power-of-two scale per 32 values in the
    /// tensor cores (<see cref="DeviceFeatures.BlockScaledFp8"/>, CUDA on compute capability 12.x), matrices of block-scaled E4M3 weights
    /// (128-value scale groups) are requantized to power-of-two scales at load, activations get power-of-two scales, and prompt-sized
    /// products run on the block-scaled MMA. It is the form vLLM uses by default (DeepGEMM's E8M0 scales): about 2 % more serving throughput
    /// at 64 streams, perplexity +0.3 to +0.8 % against f32 scales (benchmarks.md §86.3, §89.4). False keeps the checkpoint's f32 scales
    /// (as <see cref="DisabledFeatures"/> with <see cref="DeviceFeatures.BlockScaledFp8"/> does). Ignored for models without such weights
    /// and on devices without the feature. The environment variable <c>SHARPLIGHT_FP8_BLOCK_SCALED=1|0</c> overrides it for the process.
    /// </summary>
    public bool BlockScaledFp8 { get; set; } = true;

    /// <summary>Opt-in: let kernels trade precision for speed (see <see cref="CompileOptions.AllowReducedPrecision"/>).</summary>
    public bool AllowReducedPrecision { get; set; }

    /// <summary>
    /// The device features compilations of this load must not use: <see cref="DisabledFeatures"/>, plus
    /// <see cref="DeviceFeatures.BlockScaledFp8"/> when <see cref="BlockScaledFp8"/> (or <c>SHARPLIGHT_FP8_BLOCK_SCALED</c>) turns the mode off.
    /// </summary>
    internal HashSet<string> CompileDisabledFeatures()
    {
        var features = new HashSet<string>(DisabledFeatures, StringComparer.Ordinal);
        if (!(Llm.DiagnosticSwitches.Fp8BlockScaled ?? BlockScaledFp8))
        {
            features.Add(DeviceFeatures.BlockScaledFp8);
        }
        else if (Llm.DiagnosticSwitches.Fp8BlockScaled == true)
        {
            features.Remove(DeviceFeatures.BlockScaledFp8);
        }

        return features;
    }

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
