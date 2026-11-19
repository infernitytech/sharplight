namespace Sharplight.Llm.Serving;

/// <summary>
/// Online estimate of what a scheduler step costs: <c>time(rows) ≈ Fixed + PerRow · rows</c>, fitted by exponentially weighted least
/// squares over the steps the engine ran (recent steps weigh more, so the estimate follows the load and the device). Steps far slower
/// than predicted once the fit is established (kernels recorded for a new step shape, a device shared with another process) are
/// left out. Until enough steps were seen the prior applies. Used by the prefix cache to price an extra prefill pass against
/// recomputing tokens. Not thread-safe (the scheduler thread owns it).
/// </summary>
internal sealed class StepCostModel
{
    /// <summary>Weight of the previous statistics per new step (a memory of roughly 1 / (1 - Decay) steps).</summary>
    private const double Decay = 0.98;
    /// <summary>Steps before the fit replaces the prior.</summary>
    private const int Warmup = 16;
    /// <summary>A step slower than this many times the prediction (once warm) counts as an outlier.</summary>
    private const double OutlierFactor = 3;

    private double _w, _x, _y, _xx, _xy;
    private int _seen;
    private double _fixed, _perRow, _copy;
    private readonly bool _frozen;

    /// <param name="fixedMs">Prior cost of a step without rows (launch overhead, weights streamed once), in milliseconds.</param>
    /// <param name="perRowMs">Prior cost of one more row in a step, in milliseconds.</param>
    /// <param name="copyMs">Prior cost of copying one slot's recurrent state (a snapshot or its restore), in milliseconds.</param>
    /// <param name="frozen">Keep the priors whatever is observed (deterministic scheduling for tests).</param>
    public StepCostModel(double fixedMs = 2.0, double perRowMs = 0.01, double copyMs = 0.2, bool frozen = false)
    {
        _fixed = fixedMs;
        _perRow = perRowMs;
        _copy = copyMs;
        _frozen = frozen;
    }

    /// <summary>Estimated milliseconds of a step regardless of its rows.</summary>
    public double FixedMs => _fixed;
    /// <summary>Estimated milliseconds one more row adds to a step.</summary>
    public double PerRowMs => _perRow;
    /// <summary>Estimated milliseconds of copying one slot's recurrent state (taking or restoring a snapshot).</summary>
    public double CopyMs => _copy;
    /// <summary>Steps observed (outliers included).</summary>
    public int Seen => _seen;

    /// <summary>Estimated milliseconds of a step of <paramref name="rows"/> rows.</summary>
    public double StepMs(int rows) => _fixed + _perRow * rows;

    /// <summary>Adds a measured step.</summary>
    /// <param name="rows">Rows of the step.</param>
    /// <param name="ms">Its wall time in milliseconds.</param>
    public void Observe(int rows, double ms)
    {
        _seen++;
        if (_frozen || rows <= 0 || ms <= 0 || (_seen > Warmup && ms > OutlierFactor * StepMs(rows) + 1))
        {
            return;
        }

        _w = Decay * _w + 1;
        _x = Decay * _x + rows;
        _y = Decay * _y + ms;
        _xx = Decay * _xx + (double)rows * rows;
        _xy = Decay * _xy + rows * ms;
        if (_seen < Warmup)
        {
            return;
        }

        double meanX = _x / _w, meanY = _y / _w;
        double varX = _xx / _w - meanX * meanX;
        double slope = _perRow;
        if (varX > 1e-6 * Math.Max(1, meanX * meanX))
        {
            slope = (_xy / _w - meanX * meanY) / varX;
        }

        // Rows never make a step cheaper, and a step always costs something: keep both terms positive (the fit can tilt either way
        // while the steps seen span a narrow range of rows).
        slope = Math.Max(slope, 1e-6);
        _perRow = slope;
        _fixed = Math.Max(meanY - slope * meanX, 0.05 * meanY);
    }

    /// <summary>Adds a measured state copy (milliseconds).</summary>
    public void ObserveCopy(double ms)
    {
        if (!_frozen && ms >= 0)
        {
            _copy = 0.9 * _copy + 0.1 * ms;
        }
    }
}
