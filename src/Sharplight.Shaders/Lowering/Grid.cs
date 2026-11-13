namespace Sharplight.Shaders.Lowering;

public readonly record struct Grid(uint X, uint Y = 1, uint Z = 1)
{
    public static uint Div(long n, long d) => (uint)Math.Max(1, (n + d - 1) / d);
}
