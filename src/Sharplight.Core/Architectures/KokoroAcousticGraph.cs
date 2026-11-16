using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The second Kokoro graph over the decoder frames (symbolic count <c>T</c>): the F0 and noise predictors, the decoder, and the iSTFTNet generator with its
/// harmonic-plus-noise excitation source, ending in the frequency-domain half of the inverse STFT. The random numbers of the excitation come in as an input
/// (<see cref="NoiseInput"/>) so a run is reproducible. Outputs the windowed inverse-DFT frames (<see cref="FramesOutput"/>), which the host overlap-adds
/// (<c>Sharplight.Audio.OverlapAdd</c>) into the waveform, and the predicted F0 curve (<see cref="F0Output"/>).
/// </summary>
internal static class KokoroAcousticGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "kokoro_audio";

    /// <summary>The input of the prosody features expanded to frames, f32 <c>[T, hidden + style]</c>.</summary>
    public const string ProsodyInput = "prosody";

    /// <summary>The input of the acoustic features expanded to frames, f32 <c>[T, hidden]</c>.</summary>
    public const string AcousticInput = "acoustic";

    /// <summary>The input of the decoder style, f32 <c>[1, style]</c>.</summary>
    public const string DecoderStyleInput = "decoder_style";

    /// <summary>The input of the predictor style, f32 <c>[1, style]</c>.</summary>
    public const string PredictorStyleInput = "predictor_style";

    /// <summary>The input of standard-normal noise for the excitation source, f32 <c>[T · samples per frame, harmonics]</c>.</summary>
    public const string NoiseInput = "noise";

    /// <summary>The output of the windowed inverse-DFT frames, f32 <c>[T · samples per frame / hop + 1, n_fft]</c>.</summary>
    public const string FramesOutput = "frames";

    /// <summary>The output of the F0 curve in Hz, f32 <c>[2T, 1]</c>.</summary>
    public const string F0Output = "f0";

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph Build(KokoroParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(GraphName, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var en = b.Input(ProsodyInput, InputRole.Other, DataType.F32, new Shape(frames, p.Hidden + p.StyleDim));
        var asr = b.Input(AcousticInput, InputRole.Other, DataType.F32, new Shape(frames, p.Hidden));
        var styleDecoder = b.Input(DecoderStyleInput, InputRole.Other, DataType.F32, new Shape(1, p.StyleDim));
        var stylePredictor = b.Input(PredictorStyleInput, InputRole.Other, DataType.F32, new Shape(1, p.StyleDim));
        var noise = b.Input(NoiseInput, InputRole.Other, DataType.F32, new Shape(frames * p.SamplesPerFrame, p.Harmonics));

        var (f0, n) = Prosody(b, p, en, stylePredictor);                                                                    // [2T, 1] each
        var x = Decoder(b, p, asr, f0, n, styleDecoder, noise);
        var (real, imag) = (x.Real, x.Imag);
        var spectrum = b.Concat(2, b.Reshape(real, 0, p.NFft / 2 + 1, 1), b.Reshape(imag, 0, p.NFft / 2 + 1, 1));
        var inverse = b.Emit1("Dft", [spectrum], Attributes.From(("length", (long)p.NFft), ("out_bins", (long)p.NFft), ("inverse", true), ("real_output", true)));
        var window = b.Constant("hann_window", DataType.F32, new Shape(p.NFft), KokoroBlocks.Bytes(Window.Hann(p.NFft)));
        b.Output(FramesOutput, b.Mul(b.Reshape(inverse, 0, p.NFft), window));
        b.Output(F0Output, f0);
        return b.Build();
    }

    /// <summary>The F0 and noise curves (twice the frame rate) from the prosody features.</summary>
    private static (Value F0, Value N) Prosody(GraphBuilder b, KokoroParameters p, Value en, Value style)
    {
        var shared = KokoroBlocks.BiLstm(b, en, "predictor.shared");
        long h = p.Hidden;
        Value Chain(string name)
        {
            var x = KokoroBlocks.AdainResBlock(b, shared, style, $"predictor.{name}.0", h, h, upsample: false);
            x = KokoroBlocks.AdainResBlock(b, x, style, $"predictor.{name}.1", h, h / 2, upsample: true);
            x = KokoroBlocks.AdainResBlock(b, x, style, $"predictor.{name}.2", h / 2, h / 2, upsample: false);
            return KokoroBlocks.Conv(b, x, $"predictor.{name}_proj", 0);
        }

        return (Chain("F0"), Chain("N"));
    }

    /// <summary>The decoder blocks and the generator; returns the real and imaginary parts of the output spectrum (<c>[frames, bins]</c> each).</summary>
    private static (Value Real, Value Imag) Decoder(GraphBuilder b, KokoroParameters p, Value asr, Value f0, Value n, Value style, Value noise)
    {
        var f0Frames = KokoroBlocks.Conv(b, f0, "decoder.F0_conv", 1, stride: 2);
        var nFrames = KokoroBlocks.Conv(b, n, "decoder.N_conv", 1, stride: 2);
        var x = KokoroBlocks.AdainResBlock(b, b.Concat(1, asr, f0Frames, nFrames), style, "decoder.encode", p.Hidden + 2, 1024, upsample: false);
        var residual = KokoroBlocks.Conv(b, asr, "decoder.asr_res.0", 0);
        for (int i = 0; i < 4; i++)
        {
            x = b.Concat(1, x, residual, f0Frames, nFrames);
            x = KokoroBlocks.AdainResBlock(b, x, style, $"decoder.decode.{i}", 1024 + 2 + 64, i == 3 ? 512 : 1024, upsample: i == 3);
        }

        return Generator(b, p, x, f0, style, noise);
    }

    private static (Value Real, Value Imag) Generator(GraphBuilder b, KokoroParameters p, Value x, Value f0, Value style, Value noise)
    {
        int bins = (p.NFft / 2) + 1;
        var harmonics = HarmonicSource(b, p, f0, noise);                                                                   // [samples, 1]
        var stft = b.Constant("stft_basis", DataType.F32, new Shape(2 * bins, 1, p.NFft), KokoroBlocks.Bytes(StftBasis(p.NFft)));
        var transformed = b.Emit1("Conv1d", [harmonics, stft],
            Attributes.From(("pad_left", (long)(p.NFft / 2)), ("pad_right", (long)(p.NFft / 2)), ("stride", (long)p.Hop), ("pad_mode", "reflect")));
        var re = b.Slice(transformed, 1, 0, bins);
        var im = b.Slice(transformed, 1, bins, bins);
        var spec = b.Unary("Sqrt", b.Add(b.Unary("Square", re), b.Unary("Square", im)));
        var har = b.Concat(1, spec, Angle(b, im, re));                                                                      // [frames, 2 * bins]

        long channels = p.InitialChannels;
        for (int i = 0; i < p.UpsampleRates.Length; i++)
        {
            using var _ = b.Group($"decoder.generator.stage.{i}");
            long rate = p.UpsampleRates[i];
            long kernel = p.UpsampleKernels[i];
            long outChannels = channels / 2;
            x = KokoroBlocks.LeakyRelu(b, x, 0.1f);
            Value source;
            if (i + 1 < p.UpsampleRates.Length)
            {
                long stride = p.UpsampleRates.Skip(i + 1).Aggregate(1L, (a, c) => a * c);
                source = KokoroBlocks.Conv(b, har, $"decoder.generator.noise_convs.{i}", (stride + 1) / 2, stride: stride);
                source = KokoroBlocks.SnakeResBlock(b, source, style, $"decoder.generator.noise_res.{i}", outChannels, 7, [1, 3, 5]);
            }
            else
            {
                source = KokoroBlocks.Conv(b, har, $"decoder.generator.noise_convs.{i}", 0);
                source = KokoroBlocks.SnakeResBlock(b, source, style, $"decoder.generator.noise_res.{i}", outChannels, 11, [1, 3, 5]);
            }

            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"decoder.generator.ups.{i}.weight"), b.Weight($"decoder.generator.ups.{i}.bias")],
                Attributes.From(("stride", rate), ("crop_left", (kernel - rate) / 2), ("crop_right", (kernel - rate) / 2)));
            if (i == p.UpsampleRates.Length - 1)
            {
                x = ReflectPadLeft(b, x, outChannels);
            }

            x = b.Add(x, source);
            Value? sum = null;
            for (int j = 0; j < p.ResBlockKernels.Length; j++)
            {
                var branch = KokoroBlocks.SnakeResBlock(b, x, style, $"decoder.generator.resblocks.{(i * p.ResBlockKernels.Length) + j}", outChannels, p.ResBlockKernels[j], p.ResBlockDilations[j]);
                sum = sum is null ? branch : b.Add(sum, branch);
            }

            x = b.Scale(sum!, 1f / p.ResBlockKernels.Length);
            channels = outChannels;
        }

        x = KokoroBlocks.LeakyRelu(b, x, 0.01f);
        x = KokoroBlocks.Conv(b, x, "decoder.generator.conv_post", 3);
        var magnitude = b.Unary("Exp", b.Slice(x, 1, 0, bins));
        var phase = b.Unary("Sin", b.Slice(x, 1, bins, bins));
        return (b.Mul(magnitude, b.Unary("Cos", phase)), b.Mul(magnitude, b.Unary("Sin", phase)));
    }

    /// <summary>
    /// The angle of the complex number <c>re + i·im</c> (<c>torch.angle</c>): an exactly zero imaginary part gives 0 or pi by the sign of the real part, whatever
    /// the sign of that zero (the DC and Nyquist bins of a real transform have an imaginary part of exactly zero).
    /// </summary>
    private static Value Angle(GraphBuilder b, Value im, Value re)
    {
        var zero = b.Constant("angle_zero", DataType.F32, new Shape(1), KokoroBlocks.Bytes([0f]));
        var pi = b.Constant("angle_pi", DataType.F32, new Shape(1), KokoroBlocks.Bytes([MathF.PI]));
        var flat = b.Binary("Equal", im, zero);
        var half = b.Emit1("Where", [b.Binary("Less", re, zero), pi, zero]);
        return b.Emit1("Where", [flat, half, b.Binary("Atan2", im, re)]);
    }

    /// <summary>One reflected sample prepended along time (<c>ReflectionPad1d((1, 0))</c>), as a depthwise identity convolution with reflect padding.</summary>
    private static Value ReflectPadLeft(GraphBuilder b, Value x, long channels)
    {
        var ones = b.Constant($"identity_{channels}", DataType.F32, new Shape(channels, 1, 1), KokoroBlocks.Bytes([.. Enumerable.Repeat(1f, (int)channels)]));
        return b.Emit1("Conv1d", [x, ones], Attributes.From(("pad_left", 1L), ("groups", channels), ("pad_mode", "reflect")));
    }

    /// <summary>
    /// The harmonic-plus-noise excitation: the F0 curve times the harmonic numbers gives per-frame phase increments; their running sum (the phase, interpolated up
    /// to the sample rate) drives the sines, which are gated by the voiced mask, mixed with noise and merged to one channel.
    /// </summary>
    private static Value HarmonicSource(GraphBuilder b, KokoroParameters p, Value f0, Value noise)
    {
        long up = p.SamplesPerFrame / 2;                                                                                   // samples per F0 frame
        var numbers = b.Constant("harmonic_numbers", DataType.F32, new Shape(p.Harmonics), KokoroBlocks.Bytes([.. Enumerable.Range(1, p.Harmonics).Select(i => (float)i)]));
        var one = b.Constant("one", DataType.F32, new Shape(1), KokoroBlocks.Bytes([1f]));
        var threshold = b.Constant("voiced_threshold", DataType.F32, new Shape(1), KokoroBlocks.Bytes([p.VoicedThreshold]));
        var radians = b.Binary("Mod", b.Scale(b.Mul(f0, numbers), 1f / p.SampleRate), one);                                // [2T, harmonics]
        var phase = b.Scale(b.Scale(b.Emit1("CumSum", [radians], Attributes.From(("axis", 0L))), 2f), MathF.PI);
        var sines = b.Scale(b.Unary("Sin", b.Emit1("Interpolate1d", [b.Scale(phase, up)], Attributes.From(("mode", "linear"), ("up", up), ("down", 1L)))), p.SineAmplitude);
        var voiced = b.Emit1("Interpolate1d", [b.Binary("Greater", f0, threshold)], Attributes.From(("mode", "nearest"), ("up", up), ("down", 1L)));   // [samples, 1]
        var noiseAmplitude = b.Scale(voiced, p.NoiseStd - (p.SineAmplitude / 3f), p.SineAmplitude / 3f);
        var waves = b.Add(b.Mul(sines, voiced), b.Mul(noiseAmplitude, noise));
        return b.Unary("Tanh", b.Linear(waves, "decoder.generator.m_source.l_linear.weight", "decoder.generator.m_source.l_linear.bias"));
    }

    /// <summary>The weights of a forward real DFT as a convolution (<c>torch.stft</c> with a periodic Hann window): <c>[2 · bins, 1, n_fft]</c>, cosines then negated sines.</summary>
    private static float[] StftBasis(int nFft)
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
                // The imaginary parts of the DC and Nyquist bins are exactly zero (a real FFT's output): their phase is 0 or pi, never a rounding artifact.
                basis[((bins + k) * nFft) + t] = k == 0 || 2 * k == nFft ? 0f : (float)(-window[t] * Math.Sin(angle));
            }
        }

        return basis;
    }
}
