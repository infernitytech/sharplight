namespace Sharplight.Runtime;

/// <summary>
/// Reused host buffers of the split passes of one CPU expert segment (<see cref="CompiledModel"/>): the expert ids and slot weights of the
/// GPU's and the CPU's share, and the GPU's outputs.
/// </summary>
internal sealed class SplitBuffers(int idBytes, long outBytes)
{
    /// <summary>The ids of the GPU's share (the CPU's slots -1).</summary>
    public byte[] GpuIds { get; } = new byte[idBytes];
    /// <summary>The ids of the CPU's share (the GPU's slots -1).</summary>
    public byte[] CpuIds { get; } = new byte[idBytes];
    /// <summary>The slot weights of the GPU's share (the CPU's slots zero), when the segment needs them.</summary>
    public byte[] GpuWeights { get; } = new byte[idBytes];
    /// <summary>The slot weights of the CPU's share (the GPU's slots zero).</summary>
    public byte[] CpuWeights { get; } = new byte[idBytes];
    /// <summary>The GPU's outputs, added to the CPU's.</summary>
    public byte[] Out { get; } = new byte[outBytes];
}
