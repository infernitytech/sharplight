using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Fills a dense f32 output with uniform or normal values from <see cref="CounterRandom"/> (element i uses draw i).</summary>
internal sealed unsafe class RandomOp(TRef y, bool normal, float a, float b, long seed) : CpuOp
{
    private readonly ulong _key = CounterRandom.Key(seed);

    public override string Name => "Random";

    public override void Run(RunParameters p)
    {
        long n = y.Elements(p);
        float* o = (float*)y.P;
        for (long i = 0; i < n; i++)
        {
            o[i] = normal ? (float)(a + (b * CounterRandom.Normal(_key, i))) : a + ((b - a) * CounterRandom.Uniform(_key, i));
        }
    }
}
