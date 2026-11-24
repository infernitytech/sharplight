using System.Globalization;

/// <summary>One convolution: batch N, C input channels (over Parts concatenated inputs), H x W map, O outputs, K x K kernel.</summary>
internal sealed record ConvCase(string Label, long N, long C, long H, long W, long O, long K, long Stride, int Parts)
{
    public double GigaFlops
    {
        get
        {
            long ho = (H + 2 * (K / 2) - K) / Stride + 1;
            long wo = (W + 2 * (K / 2) - K) / Stride + 1;
            return 2.0 * N * O * ho * wo * C * K * K * 1e-9;
        }
    }

    public static ConvCase Parse(string spec)
    {
        var p = spec.Split(':');
        long L(int i) => long.Parse(p[i], CultureInfo.InvariantCulture);
        return new ConvCase(p[0], L(1), L(2), L(3), L(4), L(5), L(6), L(7), p.Length > 8 ? int.Parse(p[8], CultureInfo.InvariantCulture) : 1);
    }
}
