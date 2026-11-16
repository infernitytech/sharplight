using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face EnCodec (<c>encodec</c>: EncodecModel, the causal 24 kHz codec with weight-normalized convolutions) → canonical <c>encodec</c> (documented on
/// <c>EncodecAdapter</c>). Weight normalization is folded (<c>weight_g</c> and <c>weight_v</c> become <c>weight</c>), the two LSTM bias vectors of a layer are summed
/// into <c>bias_l{n}</c>, and the codebooks are the <c>quantizer.layers.{i}.codebook.embed</c> weights (the EMA training buffers are dropped).
/// </summary>
internal sealed class EncodecHfMapping : IHfArchitectureMapping
{
    private static readonly Regex s_norm = new(@"^(.+)\.conv\.weight_(g|v)$", RegexOptions.CultureInvariant);

    /// <inheritdoc/>
    public string Architecture => "encodec";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["encodec"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("use_causal_conv") == false)
        {
            yield return "encodec: only the causal 24 kHz variant is supported (the 48 kHz model uses non-causal convolutions with group normalization and chunking).";
        }

        if ((c.GetString("norm_type") ?? "weight_norm") != "weight_norm")
        {
            yield return "encodec: only weight normalization is supported.";
        }

        if ((c.GetString("pad_mode") ?? "reflect") != "reflect")
        {
            yield return "encodec: only reflect padding is supported.";
        }

        if (c.GetInt("audio_channels") is > 1)
        {
            yield return "encodec: only mono models are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        long[] ratios = [.. c.Get("upsampling_ratios")!.Value.EnumerateArray().Select(v => v.GetInt64())];
        int codebookSize = c.Int("codebook_size");
        int sampleRate = c.Int("sampling_rate");
        long hop = ratios.Aggregate(1L, (x, y) => x * y);
        double frameRate = Math.Ceiling(sampleRate / (double)hop);
        double[] bandwidths = c.Get("target_bandwidths") is { ValueKind: JsonValueKind.Array } list ? [.. list.EnumerateArray().Select(v => v.GetDouble())] : [];
        double bitsPerCodebook = Math.Log2(codebookSize) * frameRate;
        int count = bandwidths.Length == 0 ? 0 : (int)Math.Floor(bandwidths.Max() * 1000 / bitsPerCodebook);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = sampleRate,
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "num_filters"] = c.Int("num_filters"),
            [a + "kernel_size"] = c.Int("kernel_size"),
            [a + "last_kernel_size"] = c.Int("last_kernel_size"),
            [a + "residual_kernel_size"] = c.Int("residual_kernel_size"),
            [a + "dilation_growth_rate"] = c.Int("dilation_growth_rate"),
            [a + "residual_layers"] = c.Int("num_residual_layers"),
            [a + "compress"] = c.Int("compress"),
            [a + "lstm_layers"] = c.Int("num_lstm_layers"),
            [a + "upsampling_ratios"] = ratios,
            [a + "codebook_size"] = codebookSize,
            [a + "codebook_dim"] = c.Int("codebook_dim"),
            [a + "codebook_count"] = count,
            [a + "default_codebooks"] = (int)Math.Clamp(Math.Floor(6.0 * 1000 / bitsPerCodebook), 1, Math.Max(count, 1)),
            [a + "conv_shortcut"] = c.GetBool("use_conv_shortcut") ?? true,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        int count = (int)Math.Floor(c.Get("target_bandwidths")!.Value.EnumerateArray().Max(v => v.GetDouble()) * 1000 / (Math.Log2(c.Int("codebook_size")) * Math.Ceiling(c.Int("sampling_rate")
            / (double)c.Get("upsampling_ratios")!.Value.EnumerateArray().Aggregate(1L, (x, v) => x * v.GetInt64()))));
        foreach (var (name, tensor) in tensors)
        {
            if (name.EndsWith(".cluster_size", StringComparison.Ordinal) || name.EndsWith(".embed_avg", StringComparison.Ordinal) || name.EndsWith(".inited", StringComparison.Ordinal))
            {
                continue;
            }

            if (s_norm.Match(name) is { Success: true } norm)
            {
                if (norm.Groups[2].Value == "g")
                {
                    string stem = norm.Groups[1].Value;
                    collector.Add(HfWeightNormFolding.Fold(tensor.ToWeightInfo(), tensors[stem + ".conv.weight_v"].ToWeightInfo(), stem + ".weight"));
                }

                continue;
            }

            if (name.EndsWith(".conv.bias", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name.Replace(".conv.bias", ".bias", StringComparison.Ordinal)));
            }
            else if (Regex.Match(name, @"^(.+\.lstm)\.bias_ih_l(\d+)$") is { Success: true } lstm)
            {
                string n = lstm.Groups[2].Value;
                collector.Add(HfWeightNormFolding.Sum(tensor.ToWeightInfo(), tensors[$"{lstm.Groups[1].Value}.bias_hh_l{n}"].ToWeightInfo(), $"{lstm.Groups[1].Value}.bias_l{n}"));
            }
            else if (name.Contains(".lstm.bias_hh_l", StringComparison.Ordinal))
            {
                continue;
            }
            else if (name.Contains(".lstm.weight_", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name));
            }
            else if (Regex.IsMatch(name, @"^quantizer\.layers\.\d+\.codebook\.embed$"))
            {
                if (int.Parse(name.Split('.')[2], System.Globalization.CultureInfo.InvariantCulture) < count)
                {
                    collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name));
                }
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }
}
