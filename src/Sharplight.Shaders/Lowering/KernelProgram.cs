using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Backend-neutral executable form of a graph segment: logical buffers, kernel launches in order
/// (with hazard info), and where segment inputs/outputs/states live. Vulkan/CUDA/Metal providers
/// only allocate buffers, create pipelines and replay the launches.
/// </summary>
public sealed class KernelProgram
{
    /// <summary>All buffers; <see cref="BufferDecl.Id"/> is the index.</summary>
    public required IReadOnlyList<BufferDecl> Buffers { get; init; }
    /// <summary>Launches in execution order; consecutive launches need a barrier only when their ranges conflict.</summary>
    public required IReadOnlyList<KernelLaunch> Launches { get; init; }
    /// <summary>Launches that reset all states (zero fill), run on ResetState.</summary>
    public required IReadOnlyList<KernelLaunch> ResetLaunches { get; init; }
    /// <summary>Where each segment input is expected (host-written input buffers).</summary>
    public required IReadOnlyDictionary<Value, TensorRef> InputRefs { get; init; }
    /// <summary>Where each segment output is produced (dense, host-readable buffers).</summary>
    public required IReadOnlyDictionary<Value, TensorRef> OutputRefs { get; init; }
    /// <summary>Host-visible buffer of 32-bit runtime scalars (see <see cref="ScalarSlots"/>), written before each execution.</summary>
    public required BufferDecl RuntimeBuffer { get; init; }
    /// <summary>Slot index in the runtime buffer for each runtime scalar.</summary>
    public required IReadOnlyDictionary<string, int> ScalarSlots { get; init; }
    /// <summary>Symbolic dimensions the program depends on; every execution must bind all of them.</summary>
    public required IReadOnlyList<string> Symbols { get; init; }
}
