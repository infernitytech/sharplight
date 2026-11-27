namespace Sharplight.TimeSeries;

/// <summary>Draws standard normal, gamma and Poisson variates from a seeded <see cref="Random"/> (the distributions of uni2ts' mixture components).</summary>
internal sealed class VariateSampler(Random random)
{
    private double _spare;
    private bool _hasSpare;

    /// <summary>A uniform draw in [0, 1).</summary>
    public double Uniform() => random.NextDouble();

    /// <summary>A standard normal draw (Marsaglia's polar method).</summary>
    public double Normal()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }

        double u;
        double v;
        double s;
        do
        {
            u = (2 * random.NextDouble()) - 1;
            v = (2 * random.NextDouble()) - 1;
            s = (u * u) + (v * v);
        }
        while (s >= 1 || s == 0);

        double factor = Math.Sqrt(-2 * Math.Log(s) / s);
        _spare = v * factor;
        _hasSpare = true;
        return u * factor;
    }

    /// <summary>A gamma draw of shape <paramref name="shape"/> and scale 1 (Marsaglia and Tsang; shapes below 1 boosted by U^(1/shape)).</summary>
    public double Gamma(double shape)
    {
        if (!(shape > 0))
        {
            return 0;
        }

        if (shape < 1)
        {
            return Gamma(shape + 1) * Math.Pow(1 - random.NextDouble(), 1 / shape);
        }

        double d = shape - (1.0 / 3);
        double c = 1 / Math.Sqrt(9 * d);
        while (true)
        {
            double x;
            double v;
            do
            {
                x = Normal();
                v = 1 + (c * x);
            }
            while (v <= 0);

            v = v * v * v;
            double u = 1 - random.NextDouble();
            if (u < 1 - (0.0331 * x * x * x * x) || Math.Log(u) < (0.5 * x * x) + (d * (1 - v + Math.Log(v))))
            {
                return d * v;
            }
        }
    }

    /// <summary>A Poisson draw of rate <paramref name="rate"/> (multiplication method below 10, Hörmann's PTRS transformed rejection above).</summary>
    public double Poisson(double rate)
    {
        if (!(rate > 0))
        {
            return 0;
        }

        if (rate < 10)
        {
            double limit = Math.Exp(-rate);
            double product = random.NextDouble();
            int k = 0;
            while (product > limit)
            {
                k++;
                product *= random.NextDouble();
            }

            return k;
        }

        double slam = Math.Sqrt(rate);
        double logRate = Math.Log(rate);
        double b = 0.931 + (2.53 * slam);
        double a = -0.059 + (0.02483 * b);
        double inverseAlpha = 1.1239 + (1.1328 / (b - 3.4));
        double vr = 0.9277 - (3.6224 / (b - 2));
        while (true)
        {
            double u = random.NextDouble() - 0.5;
            double v = random.NextDouble();
            double us = 0.5 - Math.Abs(u);
            double k = Math.Floor((((2 * a / us) + b) * u) + rate + 0.43);
            if (us >= 0.07 && v <= vr)
            {
                return k;
            }

            if (k < 0 || (us < 0.013 && v > us))
            {
                continue;
            }

            if (Math.Log(v) + Math.Log(inverseAlpha) - Math.Log((a / (us * us)) + b) <= -rate + (k * logRate) - LogFactorial(k))
            {
                return k;
            }
        }
    }

    // ln(k!) by Stirling's series (exact enough from k = 10 on), a product below.
    private static double LogFactorial(double k)
    {
        if (k < 10)
        {
            double product = 1;
            for (int i = 2; i <= (int)k; i++)
            {
                product *= i;
            }

            return Math.Log(product);
        }

        double n = k + 1;
        return ((n - 0.5) * Math.Log(n)) - n + (0.5 * Math.Log(2 * Math.PI)) + (1 / (12 * n)) - (1 / (360 * n * n * n));
    }
}
