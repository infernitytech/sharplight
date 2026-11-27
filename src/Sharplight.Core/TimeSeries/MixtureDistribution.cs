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

    /// <summary>
    /// The log density of the mixture with the mapped parameters at <paramref name="value"/> (in the scaled space), as uni2ts' <c>Mixture.log_prob</c>:
    /// the log-sum-exp over the components of the log weight plus the component's log density, a component outside its support (negative binomial: below 0,
    /// log-normal: not above 0) contributing nothing.
    /// </summary>
    public double LogProb(ReadOnlySpan<double> mapped, double value)
    {
        Span<double> terms = stackalloc double[_kinds.Length];
        double max = double.NegativeInfinity;
        for (int c = 0; c < _kinds.Length; c++)
        {
            // A component of weight 0 contributes nothing (as exp(log 0 + finite) in torch).
            double component = mapped[c] > 0 ? ComponentLogProb(c, mapped, value) : double.NegativeInfinity;
            terms[c] = double.IsNegativeInfinity(component) ? component : Math.Log(mapped[c]) + component;
            max = Math.Max(max, terms[c]);
        }

        if (double.IsNegativeInfinity(max) || double.IsNaN(max))
        {
            return max;
        }

        double sum = 0;
        foreach (double term in terms)
        {
            sum += Math.Exp(term - max);
        }

        return max + Math.Log(sum);
    }

    // torch.distributions' log_prob of component c (uni2ts' NegativeBinomial for that kind), -inf outside the support.
    private double ComponentLogProb(int c, ReadOnlySpan<double> mapped, double x)
    {
        int o = _offsets[c];
        switch (_kinds[c])
        {
            case "student_t":
            {
                double df = mapped[o];
                double scale = mapped[o + 2];
                double y = (x - mapped[o + 1]) / scale;
                double z = Math.Log(scale) + (0.5 * Math.Log(df)) + (0.5 * Math.Log(Math.PI)) + LogGamma(0.5 * df) - LogGamma(0.5 * (df + 1));
                return (-0.5 * (df + 1) * double.LogP1(y * y / df)) - z;
            }

            case "normal":
                return NormalLogProb(x, mapped[o], mapped[o + 1]);
            case "normal_fixed":
                return NormalLogProb(x, mapped[o], _fixedScales[c]);
            case "negative_binomial":
            {
                if (!(x >= 0))
                {
                    return double.NegativeInfinity;
                }

                double r = mapped[o];
                double logits = mapped[o + 1];
                double unnormalized = (r * LogSigmoid(-logits)) + (LogSigmoid(logits) * x);
                double normalization = LogGamma(1 + x) + LogGamma(r) - LogGamma(1 + x + r) + Math.Log(r + x);
                return unnormalized - normalization;
            }

            default:
                return x > 0 ? NormalLogProb(Math.Log(x), mapped[o], mapped[o + 1]) - Math.Log(x) : double.NegativeInfinity;
        }
    }

    private static double NormalLogProb(double x, double loc, double scale)
    {
        double d = x - loc;
        return (-(d * d) / (2 * scale * scale)) - Math.Log(scale) - (0.5 * Math.Log(2 * Math.PI));
    }

    // torch.nn.functional.logsigmoid: min(x, 0) - log1p(exp(-|x|)).
    private static double LogSigmoid(double x) => Math.Min(x, 0) - double.LogP1(Math.Exp(-Math.Abs(x)));

    private static readonly double[] s_lanczos =
    [
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905,
        -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
    ];

    /// <summary>log |Γ(x)| (Lanczos, g = 7, with the reflection formula below 0.5); about 1e-15 relative.</summary>
    internal static double LogGamma(double x)
    {
        if (x < 0.5)
        {
            return Math.Log(Math.PI / Math.Abs(Math.Sin(Math.PI * x))) - LogGamma(1 - x);
        }

        x -= 1;
        double a = s_lanczos[0];
        double t = x + 7.5;
        for (int i = 1; i < s_lanczos.Length; i++)
        {
            a += s_lanczos[i] / (x + i);
        }

        return (0.5 * Math.Log(2 * Math.PI)) + ((x + 0.5) * Math.Log(t)) - t + Math.Log(a);
    }

    // torch.nn.functional.softplus (beta 1, threshold 20).
    private static double Softplus(double x) => x > 20 ? x : double.LogP1(Math.Exp(x));
}
