namespace Sharplight.Runtime;

/// <summary>
/// Chooses how many of a row's expert slots the GPU takes in the split passes of one CPU expert segment (<see cref="CompiledModel"/>), from
/// 0 (the CPU alone) to all of them, by the measured time of whole passes: it keeps a smoothed pass time per share, takes the fastest one
/// known, tries an unmeasured neighbor of it right away and a measured neighbor every <see cref="ProbeInterval"/> passes, so the share
/// follows the machine's load and finds the CPU-only pass where the GPU's copies are slower than the CPU's products (for instance large
/// experts over a slow bus).
/// </summary>
internal sealed class SplitBalance
{
    /// <summary>Passes between two tries of a measured neighbor of the fastest share.</summary>
    internal const int ProbeInterval = 8;

    /// <summary>Weight of a new pass in the smoothed time of its share.</summary>
    private const double Smoothing = 0.25;

    /// <summary>A pass counts at most this many times the smoothed time of its share (a descheduled pass must not discard a good share).</summary>
    private const double OutlierFactor = 1.5;

    private readonly double[] _ms;
    private readonly int[] _samples;
    private int _passes;
    private bool _probeUp;

    /// <param name="slots">The slots per row.</param>
    /// <param name="start">The share tried first.</param>
    public SplitBalance(int slots, int start)
    {
        _ms = new double[slots + 1];
        _ms.AsSpan().Fill(double.NaN);
        _samples = new int[slots + 1];
        Current = Math.Clamp(start, 0, slots);
    }

    /// <summary>The share of the next pass.</summary>
    public int Current { get; private set; }

    /// <summary>Records the time of a pass with share <paramref name="gpuSlots"/> and chooses the share of the next one.</summary>
    public void Record(int gpuSlots, double ms)
    {
        // The first pass of a share pays its warm-up (first copies and launches): the second replaces it.
        ref double smoothed = ref _ms[gpuSlots];
        smoothed = _samples[gpuSlots]++ < 2 ? ms : smoothed + Smoothing * (Math.Min(ms, OutlierFactor * smoothed) - smoothed);
        _passes++;
        int best = 0;
        for (int s = 1; s < _ms.Length; s++)
        {
            if (double.IsNaN(_ms[best]) || !double.IsNaN(_ms[s]) && _ms[s] < _ms[best])
            {
                best = s;
            }
        }

        int below = best - 1;
        int above = best + 1;
        if (below >= 0 && double.IsNaN(_ms[below]))
        {
            Current = below;
        }
        else if (above < _ms.Length && double.IsNaN(_ms[above]))
        {
            Current = above;
        }
        else if (_passes % ProbeInterval == 0)
        {
            _probeUp = !_probeUp;
            int neighbor = _probeUp ? above : below;
            Current = neighbor >= 0 && neighbor < _ms.Length ? neighbor : _probeUp ? below : above;
            Current = Math.Clamp(Current, 0, _ms.Length - 1);
        }
        else
        {
            Current = best;
        }
    }
}
