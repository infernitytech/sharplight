namespace Sharplight.Runtime;

/// <summary>Stored compilation caches, one per device target (<see cref="IComputeDevice.CompilationTarget"/>).</summary>
public interface ICompilationCacheStore
{
    public IEnumerable<string> Targets { get; }
    /// <summary>The read-only cache stored for <paramref name="target"/>, or null.</summary>
    public ICompilationCache? Find(string target);
}
