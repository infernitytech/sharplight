namespace Sharplight.Storage;

/// <summary>An <see cref="IProgress{T}"/> that calls back on the reporting thread (no posting, so reports stay in order).</summary>
internal sealed class SynchronousProgress<T>(Action<T> report) : IProgress<T>
{
    public void Report(T value) => report(value);
}
