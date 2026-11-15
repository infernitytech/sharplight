using Sharplight.Package;
using Sharplight.Studio.Services.Precompile;

namespace Sharplight.Studio.Tests.Fakes;

/// <summary>
/// Precompiles nothing: reports the scripted progress, then completes with <see cref="Report"/> (or waits for <see cref="Gate"/> when
/// set, so tests can look at the state while it runs). <see cref="Run"/> replaces the whole behavior (e.g. a real precompilation).
/// </summary>
internal sealed class FakePrecompiler : IPrecompileService
{
    private readonly HashSet<string> _busy = new(StringComparer.Ordinal);

    public List<string> Calls { get; } = [];

    public List<PrecompileProgress> Steps { get; } = [];

    public TaskCompletionSource? Gate { get; set; }

    public Func<string, IProgress<PrecompileProgress>, CancellationToken, Task<PackagePrecompileReport>>? Run { get; set; }

    public PackagePrecompileReport Report { get; set; } = new("", [], [], [], [], 0, 0, TimeSpan.Zero);

    public bool IsBusy(string path)
    {
        lock (_busy)
        {
            return _busy.Contains(path);
        }
    }

    public async Task<PackagePrecompileReport> PrecompileAsync(string path, IProgress<PrecompileProgress> progress, CancellationToken cancellationToken)
    {
        Calls.Add(path);
        lock (_busy)
        {
            _busy.Add(path);
        }

        try
        {
            if (Run is not null)
            {
                return await Run(path, progress, cancellationToken);
            }

            foreach (var step in Steps)
            {
                progress.Report(step);
            }

            if (Gate is not null)
            {
                await Gate.Task.WaitAsync(cancellationToken);
            }

            return Report with { PackagePath = path };
        }
        finally
        {
            lock (_busy)
            {
                _busy.Remove(path);
            }
        }
    }
}
