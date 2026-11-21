namespace Sharplight.Runtime;

/// <summary>
/// A compiled segment whose device runs its work on host threads that sleep when idle (the CPU). <see cref="CompiledModel"/> calls
/// <see cref="Wake"/> before it runs the segment before this one on another device, so the threads are awake when this one starts.
/// </summary>
public interface IWakeableSegment : ICompiledSegment
{
    /// <summary>Has the device's threads wait actively for work for a short time; cheap, and harmless when no work follows.</summary>
    public void Wake();
}
