using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// S3Gen's HiFT vocoder (CosyVoice's neural-source-filter iSTFTNet) as one graph over the mel frames <c>[T, 80]</c>: the convolutional F0 predictor;
/// the harmonic source at the sample rate (F0 repeated to every sample, nine harmonics whose phase is the running sum of F0 · k / rate plus a
/// random start phase, gated by the voiced mask, mixed with Gaussian noise, merged by a linear layer and tanh); the source's STFT (a convolution
/// with the windowed DFT basis); the generator (input convolution, three transposed-convolution upsamplings each joined by the downsampled source
/// through a snake residual block, three snake residual blocks per stage averaged, output convolution) to log-magnitude and phase; and the
/// frequency-domain half of the inverse STFT (windowed frames, <c>[frames, n_fft]</c>) that the host overlap-adds. The random draws are inputs.
/// </summary>
internal static class ChatterboxVocoderGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "hift";

    /// <summary>The mel frames [<see cref="WellKnown.TokenSymbol"/>, 80].</summary>
    public const string MelInput = "mel";

    /// <summary>The start phase of each harmonic [1, harmonics] (the fundamental's is zero).</summary>
    public const string PhaseInput = "phase";

    /// <summary>Standard-normal noise per sample and harmonic [samples, harmonics].</summary>
    public const string NoiseInput = "noise";

    /// <summary>The windowed inverse-DFT frames [samples / hop + 1, n_fft].</summary>
    public const string FramesOutput = "frames";

    /// <summary>The predicted F0 per mel frame [T, 1].</summary>
    public const string F0Output = "f0";

    private const string P = "s3gen.mel2wav.";
    private const int SampleRate = 24000;
    private const float SineAmplitude = 0.1f;
    private const float NoiseStd = 0.003f;
    private const float VoicedThreshold = 10f;

    /// <summary>Samples per mel frame: the product of the upsampling rates times the STFT hop.</summary>
    /// <param name="rates">The upsampling rates.</param>
    /// <param name="hop">The STFT hop.</param>
    public static long SamplesPerFrame(long[] rates, long hop) => rates.Aggregate(1L, (a, r) => a * r) * hop;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="rates">The upsampling rates (8, 5, 3).</param>
    /// <param name="kernels">The upsampling kernels (16, 11, 7).</param>
    /// <param name="nFft">The STFT size (16).</param>
    /// <param name="hop">The STFT hop (4).</param>
    /// <param name="harmonics">Harmonics including the fundamental (9).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, long[] rates, long[] kernels, int nFft, int hop, int harmonics)
    {
        var b = new GraphBuilder(GraphName, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        long up = SamplesPerFrame(rates, hop);
        var mel = b.Input(MelInput, InputRole.Other, DataType.F32, new Shape(frames, 80));
        var phase = b.Input(PhaseInput, InputRole.Other, DataType.F32, new Shape(1, harmonics));
        var noise = b.Input(NoiseInput, InputRole.Other, DataType.F32, new Shape(frames * up, harmonics));

        var f0 = F0(b, mel);
        b.Output(F0Output, f0);
        var source = Source(b, f0, phase, noise, up, harmonics);

        int bins = (nFft / 2) + 1;
        var basis = b.Constant("hift.stft_basis", DataType.F32, new Shape(2 * bins, 1, nFft), KokoroBlocks.Bytes(StftBasis(nFft)));
        var spectrum = b.Emit1("Conv1d", [source, basis],
            Attributes.From(("pad_left", (long)(nFft / 2)), ("pad_right", (long)(nFft / 2)), ("stride", (long)hop), ("pad_mode", "reflect")));

        var x = KokoroBlocks.Conv(b, mel, P + "conv_pre", 3);
        long[] cumulative = [.. Enumerable.Range(0, rates.Length).Select(i => rates.Skip(i + 1).Aggregate(1L, (a, r) => a * r))];
        long[] sourceKernels = [7, 7, 11];
        for (int i = 0; i < rates.Length; i++)
        {
            using var _ = b.Group($"hift.stage.{i}");
            x = KokoroBlocks.LeakyRelu(b, x, 0.1f);
            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"{P}ups.{i}.weight"), b.Weight($"{P}ups.{i}.bias")],
                Attributes.From(("stride", rates[i]), ("crop_left", (kernels[i] - rates[i]) / 2), ("crop_right", (kernels[i] - rates[i]) / 2)));
            if (i == rates.Length - 1)
            {
                x = ReflectPadLeft(b, x, x.Shape[1].Fixed);
            }

            long stride = cumulative[i];
            var si = stride == 1
                ? KokoroBlocks.Conv(b, spectrum, $"{P}source_downs.{i}", 0)
                : KokoroBlocks.Conv(b, spectrum, $"{P}source_downs.{i}", stride / 2, stride: stride);
            si = SnakeResBlock(b, si, $"{P}source_resblocks.{i}", sourceKernels[i], [1, 3, 5]);
            x = b.Add(x, si);
            long[] resKernels = [3, 7, 11];
            Value? sum = null;
            for (int j = 0; j < resKernels.Length; j++)
            {
                var branch = SnakeResBlock(b, x, $"{P}resblocks.{(i * resKernels.Length) + j}", resKernels[j], [1, 3, 5]);
                sum = sum is null ? branch : b.Add(sum, branch);
            }

            x = b.Scale(sum!, 1f / resKernels.Length);
        }

        x = KokoroBlocks.LeakyRelu(b, x, 0.01f);
        x = KokoroBlocks.Conv(b, x, P + "conv_post", 3);
        var magnitude = b.Emit1("Clip", [b.Unary("Exp", b.Slice(x, 1, 0, bins))], Attributes.From(("min", float.NegativeInfinity), ("max", 100f)));
        var angle = b.Unary("Sin", b.Slice(x, 1, bins, bins));
        var real = b.Mul(magnitude, b.Unary("Cos", angle));
        var imag = b.Mul(magnitude, b.Unary("Sin", angle));
        var stacked = b.Concat(2, b.Reshape(real, 0, bins, 1), b.Reshape(imag, 0, bins, 1));
        var inverse = b.Emit1("Dft", [stacked], Attributes.From(("length", (long)nFft), ("out_bins", (long)nFft), ("inverse", true), ("real_output", true)));
        var window = b.Constant("hift.window", DataType.F32, new Shape(nFft), KokoroBlocks.Bytes(Window.Hann(nFft)));
        b.Output(FramesOutput, b.Mul(b.Reshape(inverse, 0, nFft), window));
        return b.Build();
    }

    // ConvRNNF0Predictor: five convolutions with ELU, a linear classifier, absolute value: [T, 1].
    private static Value F0(GraphBuilder b, Value mel)
    {
        var x = mel;
        for (int i = 0; i < 5; i++)
        {
            x = b.Unary("Elu", KokoroBlocks.Conv(b, x, $"{P}f0_predictor.condnet.{2 * i}", 1));
        }

        return b.Unary("Abs", b.Linear(x, P + "f0_predictor.classifier.weight", P + "f0_predictor.classifier.bias"), "hift.f0");
    }

    // SineGen + SourceModuleHnNSF at the sample rate: [samples, 1].
    private static Value Source(GraphBuilder b, Value f0, Value phase, Value noise, long up, int harmonics)
    {
        var f0Samples = b.Emit1("Interpolate1d", [f0], Attributes.From(("mode", "nearest"), ("up", up), ("down", 1L)));              // [samples, 1]
        var numbers = b.Constant("hift.harmonic_numbers", DataType.F32, new Shape(harmonics), KokoroBlocks.Bytes([.. Enumerable.Range(1, harmonics).Select(i => (float)i)]));
        var one = b.Constant("hift.one", DataType.F32, new Shape(1), KokoroBlocks.Bytes([1f]));
        var threshold = b.Constant("hift.voiced_threshold", DataType.F32, new Shape(1), KokoroBlocks.Bytes([VoicedThreshold]));
        var increments = b.Scale(b.Mul(f0Samples, numbers), 1f / SampleRate);                                                     // [samples, harmonics]
        var cycles = b.Binary("Mod", b.Emit1("CumSum", [increments], Attributes.From(("axis", 0L))), one);
        var sines = b.Scale(b.Unary("Sin", b.Add(b.Scale(cycles, 2f * MathF.PI), phase)), SineAmplitude);
        var voiced = b.Binary("Greater", f0Samples, threshold);                                                                    // [samples, 1]
        var noiseAmplitude = b.Scale(voiced, NoiseStd - (SineAmplitude / 3f), SineAmplitude / 3f);
        var waves = b.Add(b.Mul(sines, voiced), b.Mul(noiseAmplitude, noise));
        return b.Unary("Tanh", b.Linear(waves, P + "m_source.l_linear.weight", P + "m_source.l_linear.bias"));
    }

    // HiFT's residual block: three rounds of snake, dilated convolution, snake, convolution, each added to its input.
    private static Value SnakeResBlock(GraphBuilder b, Value x, string name, long kernel, long[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            var xt = SnakeActivation.Apply(b, x, b.Weight($"{name}.activations1.{i}.alpha"));
            xt = KokoroBlocks.Conv(b, xt, $"{name}.convs1.{i}", ((kernel * dilations[i]) - dilations[i]) / 2, dilations[i]);
            xt = SnakeActivation.Apply(b, xt, b.Weight($"{name}.activations2.{i}.alpha"));
            xt = KokoroBlocks.Conv(b, xt, $"{name}.convs2.{i}", (kernel - 1) / 2);
            x = b.Add(xt, x);
        }

        return x;
    }

    internal static Value ReflectPadLeft(GraphBuilder b, Value x, long channels)
    {
        var ones = b.Constant($"hift.identity_{channels}", DataType.F32, new Shape(channels, 1, 1), KokoroBlocks.Bytes([.. Enumerable.Repeat(1f, (int)channels)]));
        return b.Emit1("Conv1d", [x, ones], Attributes.From(("pad_left", 1L), ("groups", channels), ("pad_mode", "reflect")));
    }

    // torch.stft's windowed DFT as convolution weights: cosines then negated sines, [2 · bins, 1, n_fft].
    internal static float[] StftBasis(int nFft)
    {
        int bins = (nFft / 2) + 1;
        var window = Window.Hann(nFft);
        var basis = new float[2 * bins * nFft];
        for (int k = 0; k < bins; k++)
        {
            for (int t = 0; t < nFft; t++)
            {
                double angle = 2 * Math.PI * k * t / nFft;
                basis[(k * nFft) + t] = (float)(window[t] * Math.Cos(angle));
                basis[((bins + k) * nFft) + t] = k == 0 || 2 * k == nFft ? 0f : (float)(-window[t] * Math.Sin(angle));
            }
        }

        return basis;
    }
}
