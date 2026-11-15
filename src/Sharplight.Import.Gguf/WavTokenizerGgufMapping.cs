using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// llama.cpp <c>wavtokenizer-dec</c> (the decoder of the WavTokenizer speech codec, used by OuteTTS) → canonical <c>wavtokenizer</c>:
/// a codebook lookup, an input convolution, six PosNet blocks (two residual blocks, an attention block, two residual blocks and a final
/// group norm), a layer norm, twelve ConvNeXt blocks, a final layer norm and a linear head whose output holds the log-magnitudes and
/// phases of an inverse STFT. The file stores per-channel vectors as <c>[C, 1]</c> and pointwise convolutions as <c>[O, I, 1]</c>; the
/// canonical weights are the plain vectors and matrices. The inverse STFT geometry is not in the file: its size follows from the head
/// (<c>n_fft = output - 2</c>), the hop is the 24 kHz / 75 Hz frame rate of the published checkpoints.
/// </summary>
internal sealed partial class WavTokenizerGgufMapping : IGgufArchitectureMapping
{
    private const int SampleRate = 24000;
    private const int FrameRate = 75;

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["wavtokenizer-dec"];

    /// <inheritdoc/>
    public string Architecture => "wavtokenizer";

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int outputs = g.Int("embedding_length");
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "vocab_size"] = g.Int("vocab_size"),
            [a + "codebook_dim"] = g.Int("features_length"),
            [a + "hidden_size"] = g.Int("posnet.embedding_length"),
            [a + "posnet_layers"] = g.Int("posnet.block_count"),
            [a + "block_count"] = g.Int("convnext.block_count"),
            [a + "feed_forward_length"] = g.Int("feed_forward_length"),
            [a + "norm_eps"] = g.Float("attention.layer_norm_epsilon"),
            [a + "group_norm_eps"] = g.Float("attention.group_norm_epsilon"),
            [a + "group_norm_groups"] = g.Int("attention.group_norm_groups"),
            [a + "output_size"] = outputs,
            [a + "n_fft"] = outputs - 2,
            [a + "hop_length"] = SampleRate / FrameRate,
            [a + "sample_rate"] = SampleRate,
        };
        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        switch (name)
        {
            case "token_embd.weight":
                return "embedding.weight";
            case "conv1d.weight":
            case "conv1d.bias":
                return "input_conv." + name[7..];
            case "token_embd_norm.weight":
            case "token_embd_norm.bias":
                return "embedding_norm." + name[16..];
            case "output_norm.weight":
            case "output_norm.bias":
                return name;
            case "output.weight":
            case "output.bias":
                return "head." + name[7..];
        }

        var posnet = PosnetName().Match(name);
        if (posnet.Success)
        {
            string layer = posnet.Groups[1].Value;
            string part = posnet.Groups[2].Value switch
            {
                "attn_q" => "q",
                "attn_k" => "k",
                "attn_v" => "v",
                "attn_output" => "o",
                "attn_norm" => layer == "5" ? "norm" : "attn_norm",
                var other => other,
            };
            return $"posnet.{layer}.{part}.{posnet.Groups[3].Value}";
        }

        var block = ConvNextName().Match(name);
        return block.Success ? $"blocks.{block.Groups[1].Value}.{block.Groups[2].Value}" + (block.Groups[2].Value == "gamma" ? "" : "." + block.Groups[3].Value) : null;
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo weight, GgufMetadataView gguf)
    {
        var shape = weight.Shape;
        if (shape.Rank >= 2 && shape[-1].Fixed == 1 && (shape.Rank == 2 || weight.Name.StartsWith("posnet.", StringComparison.Ordinal) && shape.Rank == 3))
        {
            return weight.As(weight.Name, new Sharplight.Tensors.Shape(shape.Dims.Take(shape.Rank - 1)));
        }

        return weight;
    }

    [GeneratedRegex(@"^posnet\.(\d+)\.([a-z_0-9]+)\.(weight|bias)$")]
    private static partial Regex PosnetName();

    [GeneratedRegex(@"^convnext\.(\d+)\.([a-z0-9]+)\.?(weight|bias)?$")]
    private static partial Regex ConvNextName();
}
