using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>Keeps every usage report.</summary>
public sealed class RecordingUsageObserver : ISharplightUsageObserver
{
    private readonly List<SharplightUsage> _usages = [];

    public IReadOnlyList<SharplightUsage> Usages
    {
        get
        {
            lock (_usages)
            {
                return [.. _usages];
            }
        }
    }

    public void OnRequestCompleted(SharplightUsage usage)
    {
        lock (_usages)
        {
            _usages.Add(usage);
        }
    }
}
