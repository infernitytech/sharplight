using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>Hands every report to <paramref name="handler"/> on the reporting thread (unlike <see cref="Progress{T}"/>, which posts).</summary>
/// <param name="handler">Receives the reports.</param>
internal sealed class ForwardingProgress(Action<LoadProgress> handler) : IProgress<LoadProgress>
{
    public void Report(LoadProgress value) => handler(value);
}
