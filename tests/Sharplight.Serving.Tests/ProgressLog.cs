using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>Records progress reports synchronously (unlike <see cref="Progress{T}"/>, which posts them).</summary>
internal sealed class ProgressLog : IProgress<LoadProgress>
{
    private readonly List<LoadProgress> _reports = [];

    public IReadOnlyList<LoadProgress> Reports
    {
        get
        {
            lock (_reports)
            {
                return [.. _reports];
            }
        }
    }

    public void Report(LoadProgress value)
    {
        lock (_reports)
        {
            _reports.Add(value);
        }
    }

    /// <summary>Stages in order of their first report.</summary>
    public IReadOnlyList<LoadStage> Stages => [.. Reports.Select(r => r.Stage).Distinct()];

    /// <summary>The last report of <paramref name="stage"/>.</summary>
    public LoadProgress Last(LoadStage stage) => Reports.Last(r => r.Stage == stage);
}
