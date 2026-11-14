using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Categorical sampling from softmax(x) per row by inverse CDF over <see cref="CounterRandom"/> uniforms.</summary>
internal sealed unsafe class MultinomialOp(TRef x, TRef y, long samples, long seed) : CpuOp
{
    private readonly ulong _key = CounterRandom.Key(seed);
    private readonly int _classes = (int)x.Dims[1];

    public override string Name => "Multinomial";

    public override void Run(RunParameters p)
    {
        long rows = x.Dim(0, p);
        var cdf = new double[_classes];
        for (long r = 0; r < rows; r++)
        {
            float* row = (float*)x.P + (r * _classes);
            float max = float.NegativeInfinity;
            for (int c = 0; c < _classes; c++)
            {
                max = Math.Max(max, row[c]);
            }

            double total = 0;
            for (int c = 0; c < _classes; c++)
            {
                total += Math.Exp(row[c] - max);
                cdf[c] = total;
            }

            int* o = (int*)y.P + (r * samples);
            for (long s = 0; s < samples; s++)
            {
                double u = CounterRandom.Uniform(_key, (r * samples) + s) * total;
                int index = Array.BinarySearch(cdf, u);
                index = index < 0 ? ~index : index + 1;
                o[s] = Math.Min(index, _classes - 1);
            }
        }
    }
}
