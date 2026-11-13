using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Backend-neutral executable form of a graph segment: logical buffers, kernel launches in order
/// (with hazard info), and where segment inputs/outputs/states live. Vulkan/CUDA/Metal providers
/// only allocate buffers, create pipelines and replay the launches.
/// </summary>
public sealed class KernelProgram
{
    public required IReadOnlyList<BufferDecl> Buffers { get; init; }
    public required IReadOnlyList<KernelLaunch> Launches { get; init; }
    /// <summary>Launches that reset all states (zero fill), run on ResetState.</summary>
    public required IReadOnlyList<KernelLaunch> ResetLaunches { get; init; }
    public required IReadOnlyDictionary<Value, TensorRef> InputRefs { get; init; }
    public required IReadOnlyDictionary<Value, TensorRef> OutputRefs { get; init; }
    public required BufferDecl RuntimeBuffer { get; init; }
    /// <summary>Slot index in the runtime buffer for each runtime scalar.</summary>
    public required IReadOnlyDictionary<string, int> ScalarSlots { get; init; }
    public required IReadOnlyList<string> Symbols { get; init; }
}
