using System.Globalization;

namespace Sharplight.TimeSeries;

/// <summary>
/// The mixture of a masked-encoder model (<see cref="TimeSeriesGraphs.ComponentsKey"/>): maps the raw parameters of one step
/// (<see cref="TimeSeriesGraphs.DistributionOutput"/>) through uni2ts' domain maps and samples from the mixture (in the scaled space; the caller applies
/// the scaler's affine transform). Mapped layout: the component probabilities (softmax of the logits), then every component's parameters in config order.
/// </summary>
internal sealed class MixtureDistribution
{
    private const double Epsilon = 1.1920928955078125e-07;   // torch.finfo(float32).eps, the scales' lower bound
    private readonly string[] _kinds;
    private readonly double[] _fixedScales;
    private readonly int[] _offsets;

    public MixtureDistribution(string components)
    {
        _kinds = components.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        _fixedScales = new double[_kinds.Length];
        _offsets = new int[_kinds.Length];
        int offset = _kinds.Length;
        for (int c = 0; c < _kinds.Length; c++)
        {
            if (_kinds[c].StartsWith("normal_fixed:", StringComparison.Ordinal))
            {
                _fixedScales[c] = double.Parse(_kinds[c]["normal_fixed:".Length..], CultureInfo.InvariantCulture);
                _kinds[c] = "normal_fixed";
            }

            _offsets[c] = offset;
            offset += ParameterCount(_kinds[c]);
        }

        Columns = offset;
    }

    /// <summary>The raw (and mapped) values per step.</summary>
    public int Columns { get; }

    /// <summary>The component kinds (<c>student_t</c>, <c>normal</c>, <c>normal_fixed</c>, <c>negative_binomial</c>, <c>log_normal</c>).</summary>
    public IReadOnlyList<string> Kinds => _kinds;

    /// <summary>The offset of component <paramref name="component"/>'s parameters in the mapped layout.</summary>
    public int Offset(int component) => _offsets[component];

    /// <summary>The parameters of a component kind (a <c>normal_fixed:&lt;scale&gt;</c> entry has one).</summary>
    /// <exception cref="NotSupportedException">Unknown kind.</exception>
    public static int ParameterCount(string kind) => kind switch
    {
        "student_t" => 3,
        "normal" or "negative_binomial" or "log_normal" => 2,
        _ when kind.StartsWith("normal_fixed", StringComparison.Ordinal) => 1,
        _ => throw new NotSupportedException($"Unknown mixture component '{kind}'."),
    };

    /// <summary>Applies the domain maps to the raw parameters of one step.</summary>
    public double[] Map(ReadOnlySpan<float> raw)
    {
        var mapped = new double[Columns];
        int k = _kinds.Length;
        double max = double.NegativeInfinity;
        for (int c = 0; c < k; c++)
        {
            max = Math.Max(max, raw[c]);
        }

        double total = 0;
        for (int c = 0; c < k; c++)
        {
            mapped[c] = Math.Exp(raw[c] - max);
            total += mapped[c];
        }

        for (int c = 0; c < k; c++)
        {
            mapped[c] /= total;
            int o = _offsets[c];
            switch (_kinds[c])
            {
                case "student_t":
                    mapped[o] = 2 + Softplus(raw[o]);
                    mapped[o + 1] = raw[o + 1];
                    mapped[o + 2] = Math.Max(Softplus(raw[o + 2]), Epsilon);
                    break;
                case "normal":
                case "log_normal":
                    mapped[o] = raw[o];
                    mapped[o + 1] = Math.Max(Softplus(raw[o + 1]), Epsilon);
                    break;
                case "normal_fixed":
                    mapped[o] = raw[o];
                    break;
                case "negative_binomial":
                    mapped[o] = Softplus(raw[o]);
                    mapped[o + 1] = raw[o + 1];
                    break;
            }
        }

        return mapped;
    }

    /// <summary>One draw from the mixture with the mapped parameters (in the scaled space).</summary>
    public double Sample(ReadOnlySpan<double> mapped, VariateSampler sampler)
    {
        double u = sampler.Uniform();
        int c = 0;
        double running = mapped[0];
        while (c < _kinds.Length - 1 && u >= running)
        {
            c++;
            running += mapped[c];
        }

        int o = _offsets[c];
        switch (_kinds[c])
        {
            case "student_t":
                double df = mapped[o];
                double chi = 2 * sampler.Gamma(df / 2);
                return mapped[o + 1] + (mapped[o + 2] * sampler.Normal() * Math.Sqrt(df / Math.Max(chi, double.Epsilon)));
            case "normal":
                return mapped[o] + (mapped[o + 1] * sampler.Normal());
            case "normal_fixed":
                return mapped[o] + (_fixedScales[c] * sampler.Normal());
            case "negative_binomial":
                return sampler.Poisson(sampler.Gamma(mapped[o]) * Math.Exp(mapped[o + 1]));
            default:
                return Math.Exp(mapped[o] + (mapped[o + 1] * sampler.Normal()));
        }
    }

    // torch.nn.functional.softplus (beta 1, threshold 20).
    private static double Softplus(double x) => x > 20 ? x : double.LogP1(Math.Exp(x));
}
