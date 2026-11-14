using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Hashed n-gram row ids (<c>NGramHash</c>): window f32 [T, N] (token + 1 per position back, 0 = none; a missing token cuts the window) → i32 [T, H].</summary>
internal sealed unsafe class NGramHashOp : CpuOp
{
    private readonly TRef _window;
    private readonly TRef _y;
    private readonly ulong[] _multipliers;
    private readonly ulong[] _sizes;
    private readonly ulong[] _offsets;
    private readonly int _headsPerOrder;
    private readonly long _stop;
    private readonly bool _stopCuts;
    public override string Name => "NGramHash";

    public NGramHashOp(TRef window, TRef y, long[] multipliers, long[] sizes, long[] offsets, int headsPerOrder, long stop, bool stopCuts = true)
    {
        _window = window;
        _y = y;
        _multipliers = [.. multipliers.Select(m => (ulong)m)];
        _sizes = [.. sizes.Select(s => (ulong)s)];
        _offsets = [.. offsets.Select(o => (ulong)o)];
        _headsPerOrder = headsPerOrder;
        _stop = stop;
        _stopCuts = stopCuts;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public override void Run(RunParameters p)
    {
        long t = _window.Dim(0, p);
        int n = _multipliers.Length;
        int heads = _sizes.Length;
        Span<long> tokens = stackalloc long[n];
        for (long i = 0; i < t; i++)
        {
            float* w = _window.F(i);
            bool cut = false;
            for (int j = 0; j < n; j++)
            {
                long token = (long)w[j * _window.Strides[^1]] - 1;
                cut = cut || token < 0 || (j > 0 && _stopCuts && token == _stop);
                tokens[j] = cut ? _stop : token;
            }

            int* y = (int*)_y.P + i * heads;
            for (int h = 0; h < heads; h++)
            {
                int order = h / _headsPerOrder + 2;
                ulong mixed = 0;
                for (int j = 0; j < order; j++)
                {
                    mixed ^= (ulong)tokens[j] * _multipliers[j];
                }

                y[h] = (int)(mixed % _sizes[h] + _offsets[h]);
            }
        }
    }
}
