using System.Runtime.Intrinsics;

namespace Sharplight.Cpu.Tests;

/// <summary><see cref="VectorActivations"/> against double-precision references, at both vector widths.</summary>
public sealed class VectorActivationsTests
{
    public static TheoryData<string> Functions() => ["Tanh", "Softplus", "Mish"];

    [Theory, MemberData(nameof(Functions))]
    public void ActivationIsAccurate(string function)
    {
        var rnd = new Random(7);
        var inputs = new List<float> { 0f, -0f, 1e-8f, -1e-8f, 0.6249f, 0.625f, -0.625f, 19.99f, 20f, 20.01f, -20f, 44f, -44f, 90f, -90f, -103f };
        for (int i = 0; i < 100_000; i++)
        {
            inputs.Add((float)(rnd.NextDouble() * 60 - 30));
            inputs.Add((float)((rnd.NextDouble() - 0.5) * Math.Pow(10, rnd.Next(-6, 1))));
        }

        while (inputs.Count % 8 != 0)
        {
            inputs.Add(1f);
        }

        for (int i = 0; i < inputs.Count; i += 8)
        {
            var x = inputs.GetRange(i, 8).ToArray();
            var y256 = new float[8];
            var y128 = new float[8];
            Apply256(function, Vector256.Create(x)).CopyTo(y256);
            Apply128(function, Vector128.Create(x, 0)).CopyTo(y128, 0);
            Apply128(function, Vector128.Create(x, 4)).CopyTo(y128, 4);
            for (int j = 0; j < 8; j++)
            {
                double expected = Reference(function, x[j]);
                double ulp = Math.Max(MathF.BitIncrement(MathF.Abs((float)expected)) - MathF.Abs((float)expected), float.Epsilon);
                foreach (var actual in new[] { y256[j], y128[j] })
                {
                    Assert.True(Math.Abs(actual - expected) <= 4 * ulp + 1e-30, $"{function}({x[j]}) = {actual}, expected {expected}");
                }
            }
        }
    }

    [Theory, MemberData(nameof(Functions))]
    public void ActivationKeepsNaN(string function)
    {
        Assert.True(float.IsNaN(Apply256(function, Vector256.Create(float.NaN))[0]));
    }

    private static double Reference(string function, double x) => function switch
    {
        "Tanh" => Math.Tanh(x),
        "Softplus" => x > 20 ? x : Log1p(Math.Exp(x)),
        _ => x > 20 ? x : x * Math.Tanh(Log1p(Math.Exp(x))),
    };

    /// <summary>log(1 + u) in double, by its series where 1 + u would round away u's low digits.</summary>
    private static double Log1p(double u) => u < 1e-4 ? u - u * u / 2 + u * u * u / 3 : Math.Log(1 + u);

    private static Vector256<float> Apply256(string function, Vector256<float> x) => function switch
    {
        "Tanh" => VectorActivations.Tanh(x),
        "Softplus" => VectorActivations.Softplus(x),
        _ => VectorActivations.Mish(x),
    };

    private static Vector128<float> Apply128(string function, Vector128<float> x) => function switch
    {
        "Tanh" => VectorActivations.Tanh(x),
        "Softplus" => VectorActivations.Softplus(x),
        _ => VectorActivations.Mish(x),
    };
}
