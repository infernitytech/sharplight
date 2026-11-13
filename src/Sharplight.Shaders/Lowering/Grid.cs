namespace Sharplight.Shaders.Lowering;

/// <summary>Number of workgroups of a dispatch.</summary>
/// <param name="X">Workgroups along x.</param>
/// <param name="Y">Workgroups along y.</param>
/// <param name="Z">Workgroups along z.</param>
public readonly record struct Grid(uint X, uint Y = 1, uint Z = 1)
{
    /// <summary>Ceiling division <paramref name="n"/> / <paramref name="d"/>, at least 1 (a dispatch always has a workgroup).</summary>
    public static uint Div(long n, long d) => (uint)Math.Max(1, (n + d - 1) / d);
}
