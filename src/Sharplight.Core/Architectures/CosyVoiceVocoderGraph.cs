using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// CosyVoice 3's causal HiFT vocoder (<c>CausalHiFTGenerator</c>) as two graphs over the mel frames <c>[T, 80]</c>, for a whole utterance (the
/// reference's <c>finalize=True</c>): <see cref="F0GraphName"/>, the causal convolutional F0 predictor (the first convolution sees three frames ahead,
/// the others only the past; the reference runs it in float64 because the source's phase accumulates its rounding, so a given F0 can be fed instead);
/// and <see cref="GraphName"/> over the mel and the F0: the causal neural source (the harmonics' phase is accumulated at the frame rate, scaled to the sample rate and held for each frame's samples, so the
/// sines are constant within a frame; gated by the voiced mask and mixed with uniform noise from the reference's fixed noise table, here an input);
/// the source's STFT (a convolution with the windowed DFT basis); the generator with causal convolutions throughout (the input convolution sees four
/// frames ahead; nearest upsampling followed by a left-padded convolution; strided causal downsampling of the source; causal snake residual blocks);
/// and the frequency-domain half of the inverse STFT (windowed frames <c>[samples / hop + 1, n_fft]</c>) that the host overlap-adds.
/// </summary>
internal static class CosyVoiceVocoderGraph
{
    /// <summary>The generator graph's name.</summary>
    public const string GraphName = "cosyvoice_hift";

    /// <summary>The F0 predictor graph's name.</summary>
    public const string F0GraphName = "cosyvoice_hift_f0";

    /// <summary>The mel frames [<see cref="WellKnown.TokenSymbol"/>, 80].</summary>
    public const string MelInput = "mel";

    /// <summary>Uniform noise in [0, 1) per sample and harmonic [samples, harmonics].</summary>
    public const string NoiseInput = "noise";

    /// <summary>The windowed inverse-DFT frames [samples / hop + 1, n_fft].</summary>
    public const string FramesOutput = "frames";

    /// <summary>The F0 per mel frame [T, 1] (output of the F0 graph, input of the generator).</summary>
    public const string F0Output = "f0";

    /// <summary>The F0 per mel frame [T, 1] (generator input).</summary>
    public const string F0Input = "f0";

    private const string P = "hift.";
    private const float SineAmplitude = 0.1f;
    private const float NoiseStd = 0.003f;
    private const float VoicedThreshold = 10f;

    /// <summary>Builds the F0 predictor graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    public static ComputeGraph BuildF0(Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(F0GraphName, weights);
        var mel = b.Input(MelInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 80));
        b.Output(F0Output, F0(b, mel));
        return b.Build();
    }

    /// <summary>Builds the generator graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="rates">The upsampling rates (8, 5, 3).</param>
    /// <param name="kernels">The upsampling kernels (16, 11, 7).</param>
    /// <param name="nFft">The STFT size (16).</param>
    /// <param name="hop">The STFT hop (4).</param>
    /// <param name="harmonics">Harmonics including the fundamental (9).</param>
    /// <param name="lookRight">The frames the input convolution sees ahead (4).</param>
    /// <param name="sampleRate">The sample rate (24000).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, long[] rates, long[] kernels, int nFft, int hop, int harmonics, int lookRight, int sampleRate)
    {
        var b = new GraphBuilder(GraphName, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        long up = ChatterboxVocoderGraph.SamplesPerFrame(rates, hop);
        var mel = b.Input(MelInput, InputRole.Other, DataType.F32, new Shape(frames, 80));
        var f0 = b.Input(F0Input, InputRole.Other, DataType.F32, new Shape(frames, 1));
        var noise = b.Input(NoiseInput, InputRole.Other, DataType.F32, new Shape(frames * up, harmonics));
        var source = Source(b, f0, noise, up, harmonics, sampleRate);

        int bins = (nFft / 2) + 1;
        var basis = b.Constant("hift.stft_basis", DataType.F32, new Shape(2 * bins, 1, nFft), KokoroBlocks.Bytes(ChatterboxVocoderGraph.StftBasis(nFft)));
        var spectrum = b.Emit1("Conv1d", [source, basis],
            Attributes.From(("pad_left", (long)(nFft / 2)), ("pad_right", (long)(nFft / 2)), ("stride", (long)hop), ("pad_mode", "reflect")));

        var x = Causal(b, mel, P + "conv_pre", left: 0, right: lookRight);
        long[] cumulative = [.. Enumerable.Range(0, rates.Length).Select(i => rates.Skip(i + 1).Aggregate(1L, (a, r) => a * r))];
        long[] sourceKernels = [7, 7, 11];
        long[] resKernels = [3, 7, 11];
        for (int i = 0; i < rates.Length; i++)
        {
            using var _ = b.Group($"hift.stage.{i}");
            x = KokoroBlocks.LeakyRelu(b, x, 0.1f);
            x = b.Emit1("Interpolate1d", [x], Attributes.From(("mode", "nearest"), ("up", rates[i]), ("down", 1L)));
            x = Causal(b, x, $"{P}ups.{i}", left: kernels[i] - 1, right: 0);
            if (i == rates.Length - 1)
            {
                x = ChatterboxVocoderGraph.ReflectPadLeft(b, x, x.Shape[1].Fixed);
            }

            long stride = cumulative[i];
            var si = stride == 1
                ? Causal(b, spectrum, $"{P}source_downs.{i}", left: 0, right: 0)
                : Causal(b, spectrum, $"{P}source_downs.{i}", left: stride - 1, right: 0, stride: stride);
            si = CausalSnakeResBlock(b, si, $"{P}source_resblocks.{i}", sourceKernels[i], [1, 3, 5]);
            x = b.Add(x, si);
            Value? sum = null;
            for (int j = 0; j < resKernels.Length; j++)
            {
                var branch = CausalSnakeResBlock(b, x, $"{P}resblocks.{(i * resKernels.Length) + j}", resKernels[j], [1, 3, 5]);
                sum = sum is null ? branch : b.Add(sum, branch);
            }

            x = b.Scale(sum!, 1f / resKernels.Length);
        }

        x = KokoroBlocks.LeakyRelu(b, x, 0.01f);
        x = Causal(b, x, P + "conv_post", left: 6, right: 0);
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

    // CausalConvRNNF0Predictor: a convolution of kernel 4 seeing three frames ahead, four of kernel 3 over the past, ELU after each, the linear
    // classifier and the absolute value: [T, 1].
    private static Value F0(GraphBuilder b, Value mel)
    {
        var x = b.Unary("Elu", Causal(b, mel, $"{P}f0_predictor.condnet.0", left: 0, right: 3));
        for (int i = 1; i < 5; i++)
        {
            x = b.Unary("Elu", Causal(b, x, $"{P}f0_predictor.condnet.{2 * i}", left: 2, right: 0));
        }

        return b.Unary("Abs", b.Linear(x, P + "f0_predictor.classifier.weight", P + "f0_predictor.classifier.bias"), "hift.f0");
    }

    // The causal SineGen2 + SourceModuleHnNSF: per frame the increments (F0 · k / rate) mod 1, their running sum · 2 · π · samples per frame held for
    // the frame's samples, sin · 0.1 gated by the voiced mask, plus the noise amplitude (0.003 voiced, 0.1 / 3 unvoiced) times the uniform noise,
    // merged by the linear layer and tanh: [samples, 1].
    private static Value Source(GraphBuilder b, Value f0, Value noise, long up, int harmonics, int sampleRate)
    {
        var numbers = b.Constant("hift.harmonic_numbers", DataType.F32, new Shape(harmonics), KokoroBlocks.Bytes([.. Enumerable.Range(1, harmonics).Select(i => (float)i)]));
        var one = b.Constant("hift.one", DataType.F32, new Shape(1), KokoroBlocks.Bytes([1f]));
        var rate = b.Constant("hift.sample_rate", DataType.F32, new Shape(1), KokoroBlocks.Bytes([sampleRate]));
        var threshold = b.Constant("hift.voiced_threshold", DataType.F32, new Shape(1), KokoroBlocks.Bytes([VoicedThreshold]));
        var increments = b.Binary("Mod", b.Binary("Div", b.Mul(f0, numbers), rate), one);                                   // [T, harmonics]
        var phase = b.Scale(b.Scale(b.Scale(b.Emit1("CumSum", [increments], Attributes.From(("axis", 0L))), 2f), MathF.PI), up);
        var voiced = b.Binary("Greater", f0, threshold);                                                                    // [T, 1]
        var sines = b.Mul(b.Scale(b.Unary("Sin", phase), SineAmplitude), voiced);
        var hold = Attributes.From(("mode", "nearest"), ("up", up), ("down", 1L));
        var sineSamples = b.Emit1("Interpolate1d", [sines], hold);                                                          // [samples, harmonics]
        var voicedSamples = b.Emit1("Interpolate1d", [voiced], hold);                                                       // [samples, 1]
        var noiseAmplitude = b.Scale(voicedSamples, NoiseStd - (SineAmplitude / 3f), SineAmplitude / 3f);
        var waves = b.Add(sineSamples, b.Mul(noiseAmplitude, noise));
        return b.Unary("Tanh", b.Linear(waves, P + "m_source.l_linear.weight", P + "m_source.l_linear.bias"));
    }

    // HiFT's causal residual block: three rounds of snake, dilated causal convolution, snake, causal convolution, each added to its input.
    private static Value CausalSnakeResBlock(GraphBuilder b, Value x, string name, long kernel, long[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            var xt = SnakeActivation.Apply(b, x, b.Weight($"{name}.activations1.{i}.alpha"));
            xt = Causal(b, xt, $"{name}.convs1.{i}", left: (kernel - 1) * dilations[i], right: 0, dilation: dilations[i]);
            xt = SnakeActivation.Apply(b, xt, b.Weight($"{name}.activations2.{i}.alpha"));
            xt = Causal(b, xt, $"{name}.convs2.{i}", left: kernel - 1, right: 0);
            x = b.Add(xt, x);
        }

        return x;
    }

    // A convolution with zero padding on one side (CausalConv1d: left for the past, right for a look-ahead).
    private static Value Causal(GraphBuilder b, Value x, string name, long left, long right, long dilation = 1, long stride = 1)
    {
        Value[] inputs = b.HasWeight(name + ".bias") ? [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")] : [x, b.Weight(name + ".weight")];
        return b.Emit1("Conv1d", inputs, Attributes.From(("pad_left", left), ("pad_right", right), ("dilation", dilation), ("stride", stride)));
    }
}
