using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Tasks;

/// <summary>Options for loading a model with <see cref="TaskModel.Load"/> and compiling its component graphs.</summary>
public sealed class TaskLoadOptions
{
    /// <summary>Devices in pipeline order (layers are split across them). Required.</summary>
    public required IReadOnlyList<DeviceSlot> Devices { get; init; }

    /// <summary>Fallback device (e.g. CPU) for layers that do not fit and unsupported ops.</summary>
    public DeviceSlot? Fallback { get; init; }

    /// <summary>Bytes kept free on each device for activations and scratch when placing a graph (null: the placement default, 1 GiB).</summary>
    public long? ReservePerDevice { get; init; }

    /// <summary>Auxiliary model files by role, opened next to the primary one (e.g. a separately shipped VAE or vocoder).</summary>
    public IReadOnlyList<(string Role, string Path)> Auxiliary { get; init; } = [];

    /// <summary>Architecture-adapter settings.</summary>
    public AdapterOptions Adapter { get; init; } = new();

    /// <summary>Device features (<see cref="DeviceFeatures"/>) not to use when compiling.</summary>
    public IReadOnlySet<string> DisabledFeatures { get; init; } = new HashSet<string>();

    /// <summary>Opt-in: let kernels trade precision for speed (see <see cref="CompileOptions.AllowReducedPrecision"/>).</summary>
    public bool AllowReducedPrecision { get; init; }

    /// <summary>
    /// Precompiled-artifact routing shared with other loads, or <see langword="null"/> to consult the artifact sets stored in the model
    /// package (see <see cref="CompilationArtifacts"/>).
    /// </summary>
    public CompilationArtifacts? Artifacts { get; init; }

    /// <summary>Receives progress and compile-plan messages; null for silence.</summary>
    public Action<string>? Log { get; init; }

    /// <summary>Receives load and compile progress; null for none.</summary>
    public IProgress<LoadProgress>? Progress { get; init; }

    /// <summary>Cancels loading and compilation.</summary>
    public CancellationToken CancellationToken { get; init; }
}
