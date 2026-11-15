using Sharplight.Graph;

namespace Sharplight.Runtime;

/// <summary>
/// Optional capability of an <see cref="ICompiledSegment"/>: reading an input straight from an earlier segment's output in device
/// memory, so that a boundary value between two segments on the same device does not go through host memory.
/// <see cref="CompiledModel"/> then neither downloads the value (unless a host consumer or the caller needs it) nor uploads it.
/// </summary>
public interface IDeviceForwardingSegment
{
    /// <summary>
    /// Makes every later execution read <paramref name="input"/> from the output of the same value of <paramref name="producer"/>,
    /// which executes before this segment in every run. Called once, before the first execution.
    /// </summary>
    /// <returns>False when the producer's output is not reachable from this segment (another device or provider); the value is then
    /// passed through host memory.</returns>
    public bool TryForwardFrom(Value input, ICompiledSegment producer);
}
