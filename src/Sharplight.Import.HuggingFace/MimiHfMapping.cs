using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Mimi (<c>mimi</c>: MimiModel, the 12.5 Hz speech codec of Moshi: a causal SEANet encoder and decoder around 8-layer transformers, a downsampling convolution and a split
/// semantic + acoustic residual vector quantizer) → canonical <c>mimi</c> (documented on <c>MimiAdapter</c>). The convolution modules' <c>.conv.</c> infix is dropped and the codebooks are
/// computed from the checkpoint's EMA statistics (<c>embed_sum / max(cluster_usage, 1e-5)</c>).
/// </summary>
internal sealed class MimiHfMapping : IHfArchitectureMapping
{
    private static readonly Regex s_codebook = new(@"^quantizer\.(semantic|acoustic)_residual_vector_quantizer\.layers\.(\d+)\.codebook\.(embed_sum|cluster_usage|initialized)$", RegexOptions.CultureInvariant);

    /// <inheritdoc/>
    public string Architecture => "mimi";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["mimi"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("use_causal_conv") == false)
        {
            yield return "mimi: only causal convolutions are supported.";
        }

        if ((c.GetString("pad_mode") ?? "constant") != "constant")
        {
            yield return "mimi: only constant (zero) padding of the SEANet convolutions is supported.";
        }

        if (c.GetBool("use_conv_shortcut") == true)
        {
            yield return "mimi: convolutional residual shortcuts are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = c.Int("sampling_rate"),
            [a + "frame_rate"] = (float)c.Double("frame_rate"),
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "num_filters"] = c.Int("num_filters"),
            [a + "kernel_size"] = c.Int("kernel_size"),
            [a + "last_kernel_size"] = c.Int("last_kernel_size"),
            [a + "residual_kernel_size"] = c.Int("residual_kernel_size"),
            [a + "dilation_growth_rate"] = c.Int("dilation_growth_rate"),
            [a + "residual_layers"] = c.Int("num_residual_layers"),
            [a + "compress"] = c.Int("compress"),
            [a + "upsampling_ratios"] = c.Get("upsampling_ratios")!.Value.EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "codebook_size"] = c.Int("codebook_size"),
            [a + "codebook_dim"] = c.GetInt("vector_quantization_hidden_dimension") ?? c.Int("codebook_dim"),
            [a + "semantic_codebooks"] = c.GetInt("num_semantic_quantizers") ?? 1,
            [a + "codebook_count"] = c.GetInt("num_quantizers") ?? 32,
            [a + "default_codebooks"] = 8,
            [a + "transformer.layers"] = c.Int("num_hidden_layers"),
            [a + "transformer.heads"] = c.Int("num_attention_heads"),
            [a + "transformer.head_dim"] = c.GetInt("head_dim") ?? c.Int("hidden_size") / c.Int("num_attention_heads"),
            [a + "transformer.norm_eps"] = (float)c.Double("norm_eps"),
            [a + "transformer.sliding_window"] = c.Int("sliding_window"),
            [a + "transformer.activation"] = c.GetString("hidden_act") ?? "gelu",
            [a + "transformer.rope_base"] = (float)(c.GetDouble("rope_theta") ?? c.Get("rope_parameters")?.GetProperty("rope_theta").GetDouble() ?? 10000.0),
            [a + "upsample_groups"] = c.Int("upsample_groups"),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        MapInto(collector, tensors);
        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    /// <summary>Maps Mimi's tensors (names as in a <c>mimi</c> checkpoint) into <paramref name="collector"/>; other codecs that embed a Mimi encoder call it with the encoder's tensors.</summary>
    /// <param name="collector">Receives the canonical weights.</param>
    /// <param name="tensors">The Mimi tensors by their Mimi names.</param>
    internal static void MapInto(HfWeightCollector collector, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        foreach (var (name, tensor) in tensors)
        {
            if (s_codebook.Match(name) is { Success: true } book)
            {
                if (book.Groups[3].Value == "embed_sum")
                {
                    string stem = name[..^"embed_sum".Length];
                    var usage = tensors[stem + "cluster_usage"].ToWeightInfo();
                    var sum = tensor.ToWeightInfo();
                    string canonical = $"quantizer.{book.Groups[1].Value}.layers.{book.Groups[2].Value}.embed";
                    collector.Add(WeightInfo.Derived(canonical, DataType.F32, sum.Shape, () => Codebook(sum, usage)));
                }

                continue;
            }

            string target = name.Replace("_residual_vector_quantizer", "", StringComparison.Ordinal).Replace(".conv.", ".", StringComparison.Ordinal);
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), target));
        }
    }

    private static byte[] Codebook(WeightInfo sum, WeightInfo usage)
    {
        var values = HfTransforms.DecodeF32(sum);
        var counts = HfTransforms.DecodeF32(usage);
        long width = sum.Shape[1].Fixed;
        for (int row = 0; row < counts.Length; row++)
        {
            float divisor = MathF.Max(counts[row], 1e-5f);
            for (long i = 0; i < width; i++)
            {
                values[(row * width) + i] /= divisor;
            }
        }

        return HfTransforms.EncodeF32(values);
    }
}
