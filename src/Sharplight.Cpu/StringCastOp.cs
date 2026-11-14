using System.Globalization;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Conversion between str and f32 / i32 tensors (see <c>StandardOps.StringCast</c>).</summary>
internal sealed unsafe class StringCastOp(TRef x, TRef y) : CpuOp
{
    public override string Name => "StringCast";

    public override void Run(RunParameters p)
    {
        long n = x.Elements(p);
        for (long i = 0; i < n; i++)
        {
            if (x.Type == DataType.Str)
            {
                double v = Parse(p.Strings[((int*)x.P)[i]]);
                if (y.Type == DataType.I32)
                {
                    ((int*)y.P)[i] = double.IsNaN(v) ? 0 : (int)Math.Clamp(v, int.MinValue, int.MaxValue);
                }
                else
                {
                    ((float*)y.P)[i] = (float)v;
                }
            }
            else
            {
                string text = x.Type == DataType.I32 ? ((int*)x.P)[i].ToString(CultureInfo.InvariantCulture) : Format(((float*)x.P)[i]);
                ((int*)y.P)[i] = p.Strings.Add(text);
            }
        }
    }

    private static double Parse(string s)
    {
        string t = s.Trim();
        return t.ToUpperInvariant() switch
        {
            "INF" or "+INF" or "INFINITY" or "+INFINITY" => double.PositiveInfinity,
            "-INF" or "-INFINITY" => double.NegativeInfinity,
            _ => double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : double.NaN,
        };
    }

    private static string Format(float v)
        => float.IsNaN(v) ? "NaN" : float.IsPositiveInfinity(v) ? "INF" : float.IsNegativeInfinity(v) ? "-INF" : v.ToString("R", CultureInfo.InvariantCulture);
}
