using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// XTTS v2's waveform decoder (Coqui's HiFi-GAN generator conditioned on the speaker embedding) as a graph over the GPT latents already interpolated
/// to the output frame rate <c>[T, 1024]</c>: the input convolution plus the speaker projection, four transposed-convolution upsamplings (leaky ReLU
/// 0.1 before each, the speaker projected and added after each), three residual blocks per stage averaged (leaky ReLU, dilated convolution, leaky
/// ReLU, convolution, residual), leaky ReLU, the bias-free output convolution and tanh: <c>[T · 256, 1]</c> samples at 24 kHz.
/// </summary>
internal static class XttsVocoderGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "xtts_vocoder";

    /// <summary>The interpolated latents [<see cref="WellKnown.TokenSymbol"/>, channels].</summary>
    public const string LatentsInput = "latents";

    /// <summary>The speaker embedding [1, 512].</summary>
    public const string SpeakerInput = "speaker";

    /// <summary>The samples [T · samples per frame, 1].</summary>
    public const string AudioOutput = "audio";

    private const string P = "hifigan_decoder.waveform_decoder.";
    private const float Slope = 0.1f;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="channels">Latent channels (1024).</param>
    /// <param name="speakerWidth">Speaker embedding width (512).</param>
    /// <param name="rates">Upsampling rates (8, 8, 2, 2).</param>
    /// <param name="kernels">Upsampling kernels (16, 16, 4, 4).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int channels, int speakerWidth, long[] rates, long[] kernels)
    {
        var b = new GraphBuilder(GraphName, weights);
        var z = b.Input(LatentsInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), channels));
        var g = b.Input(SpeakerInput, InputRole.Other, DataType.F32, new Shape(1, speakerWidth));
        var x = KokoroBlocks.Conv(b, z, P + "conv_pre", 3);
        x = b.Add(x, Project(b, g, P + "cond_layer"));
        long[] resKernels = [3, 7, 11];
        for (int i = 0; i < rates.Length; i++)
        {
            using var _ = b.Group($"xtts.vocoder.stage.{i}");
            x = KokoroBlocks.LeakyRelu(b, x, Slope);
            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"{P}ups.{i}.weight"), b.Weight($"{P}ups.{i}.bias")],
                Attributes.From(("stride", rates[i]), ("crop_left", (kernels[i] - rates[i]) / 2), ("crop_right", (kernels[i] - rates[i]) / 2)));
            x = b.Add(x, Project(b, g, $"{P}conds.{i}"));
            Value? sum = null;
            for (int j = 0; j < resKernels.Length; j++)
            {
                var branch = ResBlock(b, x, $"{P}resblocks.{(i * resKernels.Length) + j}", resKernels[j], [1, 3, 5]);
                sum = sum is null ? branch : b.Add(sum, branch);
            }

            x = b.Scale(sum!, 1f / resKernels.Length);
        }

        x = KokoroBlocks.LeakyRelu(b, x, 0.01f);
        x = KokoroBlocks.Conv(b, x, P + "conv_post", 3, bias: false);
        b.Output(AudioOutput, b.Unary("Tanh", x));
        return b.Build();
    }

    // A 1×1 convolution of the speaker embedding: one row added to every frame.
    private static Value Project(GraphBuilder b, Value g, string module)
    {
        var weight = b.Weight(module + ".weight");
        var matrix = b.Reshape(weight, weight.Shape[0].Fixed, weight.Shape[1].Fixed);
        return b.Add(b.MatMul(g, matrix), b.Weight(module + ".bias"));
    }

    // ResBlock1: three rounds of leaky ReLU, dilated convolution, leaky ReLU, convolution, each added to its input.
    private static Value ResBlock(GraphBuilder b, Value x, string name, long kernel, long[] dilations)
    {
        for (int i = 0; i < dilations.Length; i++)
        {
            var xt = KokoroBlocks.LeakyRelu(b, x, Slope);
            xt = KokoroBlocks.Conv(b, xt, $"{name}.convs1.{i}", ((kernel * dilations[i]) - dilations[i]) / 2, dilations[i]);
            xt = KokoroBlocks.LeakyRelu(b, xt, Slope);
            xt = KokoroBlocks.Conv(b, xt, $"{name}.convs2.{i}", (kernel - 1) / 2);
            x = b.Add(xt, x);
        }

        return x;
    }
}
