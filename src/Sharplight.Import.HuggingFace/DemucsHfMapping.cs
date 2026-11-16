using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hybrid transformer Demucs (<c>htdemucs</c>: the <c>HTDemucs</c> module of the demucs package as published on the Hub with its constructor arguments as <c>config.json</c> and
/// the state dict as safetensors, e.g. KeighBee/demucs) → canonical <c>demucs</c> (documented on <c>DemucsAdapter</c>). Every 3-D convolution weight becomes a 4-D tensor with a unit
/// height (the graphs run all convolutions as 2-D), the fused attention projections are split into q, k and v. Only the segment-wise hybrid model without Wiener filtering
/// (<c>cac</c>, <c>wiener_iters</c> 0) with the cross-domain transformer, no multi-band wrapping and no group norm in the layers is supported.
/// </summary>
internal sealed partial class DemucsHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "demucs";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["htdemucs"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.GetBool("cac") == false)
        {
            yield return "htdemucs: only the complex-as-channels models are supported.";
        }

        if ((c.GetInt("wiener_iters") ?? 0) != 0 || (c.GetInt("end_iters") ?? 0) != 0)
        {
            yield return "htdemucs: Wiener filtering is not supported.";
        }

        if (c.Get("multi_freqs") is { ValueKind: System.Text.Json.JsonValueKind.Array } bands && bands.GetArrayLength() > 0)
        {
            yield return "htdemucs: multi-band wrapping is not supported.";
        }

        if (c.Int("norm_starts") < c.Int("depth"))
        {
            yield return "htdemucs: group norm in the encoder layers is not supported.";
        }

        if ((c.GetString("t_emb") ?? "sin") != "sin" || c.GetBool("t_sparse_self_attn") == true || c.GetBool("t_sparse_cross_attn") == true)
        {
            yield return "htdemucs: only sinusoidal positions and dense attention are supported.";
        }

        if (c.GetBool("t_norm_first") == false || c.GetBool("t_gelu") == false || c.GetBool("t_group_norm") == true || c.GetBool("t_norm_in_group") == true || c.GetBool("t_cross_first") == true
            || c.GetBool("t_norm_in") == false || c.GetBool("t_norm_out") == false || c.GetBool("t_layer_scale") == false)
        {
            yield return "htdemucs: only the layout of the published transformer (pre-norm, GELU, layer norms, layer scale, norm out) is supported.";
        }

        if ((c.GetInt("context_enc") ?? 0) != 0 || (c.GetInt("dconv_mode") ?? 3) != 3 || c.GetBool("use_train_segment") == false)
        {
            yield return "htdemucs: unsupported layer options.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int sampleRate = c.GetInt("samplerate") ?? 44100;
        double segment = c.GetDouble("segment") ?? 7.8;
        int channels = c.Int("channels");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sources"] = string.Join(',', c.GetStringArray("sources") ?? ["drums", "bass", "other", "vocals"]),
            [a + "sample_rate"] = sampleRate,
            [a + "audio_channels"] = c.GetInt("audio_channels") ?? 2,
            [a + "channels"] = channels,
            [a + "growth"] = c.GetInt("growth") ?? 2,
            [a + "depth"] = c.GetInt("depth") ?? 4,
            [a + "nfft"] = c.GetInt("nfft") ?? 4096,
            [a + "kernel_size"] = c.GetInt("kernel_size") ?? 8,
            [a + "stride"] = c.GetInt("stride") ?? 4,
            [a + "dconv_depth"] = c.GetInt("dconv_depth") ?? 2,
            [a + "dconv_compress"] = c.GetInt("dconv_comp") ?? 8,
            [a + "bottom_channels"] = c.GetInt("bottom_channels") ?? channels * 8,
            [a + "freq_emb"] = (float)(c.GetDouble("freq_emb") ?? 0.2),
            [a + "emb_scale"] = (float)(c.GetDouble("emb_scale") ?? 10),
            [a + "transformer.layer_count"] = c.GetInt("t_layers") ?? 5,
            [a + "transformer.head_count"] = c.GetInt("t_heads") ?? 8,
            [a + "transformer.hidden_size"] = (int)((c.GetInt("bottom_channels") ?? channels * 8) * (c.GetDouble("t_hidden_scale") ?? 4.0)),
            [a + "transformer.max_period"] = c.GetDouble("t_max_period") ?? 10000.0,
            [a + "transformer.position_weight"] = (float)(c.GetDouble("t_weight_pos_embed") ?? 1.0),
            [a + "segment_samples"] = (int)(segment * sampleRate),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (Projection().Match(name) is { Success: true } fused)
            {
                foreach (var part in Split(info, fused.Groups[1].Value, fused.Groups[2].Value == "weight"))
                {
                    collector.Add(part);
                }

                continue;
            }

            if (name.EndsWith("num_batches_tracked", StringComparison.Ordinal))
            {
                continue;
            }

            if (info.Shape.Rank == 3 && name.EndsWith(".weight", StringComparison.Ordinal))
            {
                collector.Add(info.As(name, new Shape(info.Shape[0].Fixed, info.Shape[1].Fixed, 1, info.Shape[2].Fixed)));
            }
            else
            {
                collector.Add(HfTransforms.Plain(info, name));
            }
        }

        return collector.Weights;
    }

    // The fused in_proj of a MultiheadAttention (rows q, k, v) as three projections.
    private static IEnumerable<WeightInfo> Split(WeightInfo info, string prefix, bool weight)
    {
        long rows = info.Shape[0].Fixed / 3;
        string[] parts = ["q", "k", "v"];
        for (int i = 0; i < 3; i++)
        {
            int part = i;
            var shape = weight ? new Shape(rows, info.Shape[1].Fixed) : new Shape(rows);
            yield return WeightInfo.Derived($"{prefix}.{parts[i]}.{(weight ? "weight" : "bias")}", DataType.F32, shape, () =>
            {
                var all = HfTransforms.DecodeF32(info);
                return HfTransforms.EncodeF32(all.AsSpan((int)(part * shape.Elements()), (int)shape.Elements()).ToArray());
            });
        }
    }

    [GeneratedRegex(@"^(crosstransformer\.layers(?:_t)?\.\d+\.(?:self|cross)_attn)\.in_proj_(weight|bias)$")]
    private static partial Regex Projection();
}
