using System.Globalization;
using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// Folds the engine's <see cref="LoadProgress"/> reports (several stages may run at once, e.g. compiling kernels while weights
/// upload) into one <see cref="LoadStatus"/>: each stage has a weight in the overall fraction, finished stages count fully.
/// Thread-safe.
/// </summary>
internal sealed class LoadProgressTracker
{
    private static readonly (LoadStage Stage, double Weight)[] s_weights =
    [
        (LoadStage.Reading, 0.05), (LoadStage.Building, 0.05), (LoadStage.Placing, 0.05), (LoadStage.Compiling, 0.4), (LoadStage.Uploading, 0.45),
    ];

    private readonly Dictionary<LoadStage, LoadProgress> _latest = [];
    private readonly Lock _lock = new();

    /// <summary>Records a report and returns the combined status.</summary>
    public LoadStatus Report(LoadProgress progress)
    {
        lock (_lock)
        {
            if (progress.Stage == LoadStage.Ready)
            {
                return new LoadStatus(1, "Ready", IsDone: true);
            }

            _latest[progress.Stage] = progress;
            int furthest = s_weights.Select((entry, index) => (entry, index)).Where(pair => _latest.ContainsKey(pair.entry.Stage)).Max(pair => pair.index);
            double fraction = 0;
            for (int i = 0; i < s_weights.Length; i++)
            {
                var (stage, weight) = s_weights[i];
                if (_latest.TryGetValue(stage, out var report))
                {
                    fraction += weight * report.Fraction;
                }
                else if (i < furthest)
                {
                    fraction += weight;   // skipped or finished without a report
                }
            }

            var running = _latest.Values.Where(report => !report.IsComplete).OrderBy(report => report.Stage).Select(Describe).ToList();
            var text = running.Count > 0 ? string.Join(" · ", running) : Describe(progress);
            return new LoadStatus(Math.Clamp(fraction, 0, 0.99), text);
        }
    }

    /// <summary>One stage as text.</summary>
    internal static string Describe(LoadProgress progress) => progress.Stage switch
    {
        LoadStage.Reading => "Reading model",
        LoadStage.Building => "Building graphs",
        LoadStage.Placing => "Planning placement",
        LoadStage.Compiling => progress.Cached > 0
            ? string.Create(CultureInfo.InvariantCulture, $"Compiling kernels {progress.Completed} / {progress.Total} ({progress.Cached} cached)")
            : string.Create(CultureInfo.InvariantCulture, $"Compiling kernels {progress.Completed} / {progress.Total}"),
        LoadStage.Uploading => string.Create(CultureInfo.InvariantCulture, $"Uploading weights {progress.Completed / 1e9:F2} / {progress.Total / 1e9:F2} GB"),
        _ => "Ready",
    };
}
