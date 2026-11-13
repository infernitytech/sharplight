namespace Sharplight.OnnxRun;

/// <summary>Wall-clock times of the timed runs, in milliseconds.</summary>
internal sealed class RunTimings
{
    private readonly List<double> _milliseconds = [];

    public void Add(double milliseconds)
    {
        _milliseconds.Add(milliseconds);
    }

    /// <summary>One-line summary with the minimum, median and mean.</summary>
    public string Summarize(int batchSize)
    {
        var sorted = _milliseconds.Order().ToList();
        double median = sorted[sorted.Count / 2];
        return $"run: min {sorted[0]:F2} ms, median {median:F2} ms, mean {sorted.Average():F2} ms over {sorted.Count} runs (batch {batchSize})";
    }
}
