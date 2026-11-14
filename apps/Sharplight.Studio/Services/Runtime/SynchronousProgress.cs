namespace Sharplight.Studio.Services.Runtime;

/// <summary>An <see cref="IProgress{T}"/> that calls its handler on the reporting thread (unlike <see cref="Progress{T}"/>).</summary>
internal sealed class SynchronousProgress<T>(Action<T> handler) : IProgress<T>
{
    /// <inheritdoc/>
    public void Report(T value) => handler(value);
}
