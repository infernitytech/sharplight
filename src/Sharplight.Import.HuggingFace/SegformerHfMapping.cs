using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face SegFormer (<c>segformer</c>: SegformerForSemanticSegmentation, NVIDIA's mix-transformer b0-b5 on ADE20K / Cityscapes) → canonical <c>segformer</c> (documented on
/// <c>SegformerAdapter</c>). The decoder's batch norm is folded into the fusion convolution. The processor (a fixed square resize, mean and std) and the class names
/// (<c>id2label</c>) are stored as canonical metadata; checkpoints with <c>reduce_labels</c> (ADE20K) already have the background class dropped from the 150 classes.
/// </summary>
internal sealed class SegformerHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "segformer";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["segformer"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("architectures") is { ValueKind: JsonValueKind.Array } architectures
            && architectures.EnumerateArray().Any(a => a.GetString() == "SegformerForImageClassification"))
        {
            yield return "segformer: the image classification head is not supported (only SegformerForSemanticSegmentation).";
        }

        if (c.GetBool("reshape_last_stage") == false)
        {
            yield return "segformer: reshape_last_stage=false is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        var processor = HfImageProcessorConfig.Read(c.Directory, 512);
        int stages = c.Get("depths")!.Value.GetArrayLength();
        string Ints(string key) => string.Join(',', c.Get(key)!.Value.EnumerateArray().Select(v => v.GetInt32()));
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "image_size"] = HfImageProcessorConfig.InputSize(processor, 512),
            [a + "depths"] = Ints("depths"),
            [a + "hidden_sizes"] = Ints("hidden_sizes"),
            [a + "head_counts"] = Ints("num_attention_heads"),
            [a + "sr_ratios"] = Ints("sr_ratios"),
            [a + "patch_sizes"] = Ints("patch_sizes"),
            [a + "strides"] = Ints("strides"),
            [a + "mlp_ratios"] = c.Get("mlp_ratios") is { ValueKind: JsonValueKind.Array } ? Ints("mlp_ratios") : string.Join(',', Enumerable.Repeat(4, stages)),
            [a + "decoder_hidden_size"] = c.Int("decoder_hidden_size"),
            [a + "mlp.activation"] = (c.GetString("hidden_act") ?? "gelu") switch
            {
                "gelu" => "gelu",
                "gelu_new" or "gelu_pytorch_tanh" => "gelu_tanh",
                "relu" => "relu",
                "silu" or "swish" => "silu",
                var other => throw new NotSupportedException($"segformer: activation '{other}' is not supported."),
            },
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        if (c.Get("id2label") is { ValueKind: JsonValueKind.Object } map)
        {
            md[CanonicalKeys.ClassifierLabels] = map.EnumerateObject()
                .Select(p => (Index: int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture), Label: (p.Value.GetString() ?? "").Trim()))
                .OrderBy(p => p.Index).Select(p => p.Label).ToArray();
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.EndsWith("num_batches_tracked", StringComparison.Ordinal) || rawName.StartsWith("decode_head.batch_norm.", StringComparison.Ordinal))
            {
                continue;
            }

            string? name = Rename(rawName);
            if (name is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            if (rawName == "decode_head.linear_fuse.weight")
            {
                SafetensorsTensor T(string suffix) => tensors["decode_head.batch_norm" + suffix];
                var (w, b) = HfBatchNormFolding.Fold(tensor.ToWeightInfo(), T(".weight").ToWeightInfo(), T(".bias").ToWeightInfo(),
                    T(".running_mean").ToWeightInfo(), T(".running_var").ToWeightInfo(), 1e-5f, "decode.fuse");
                collector.Add(w);
                collector.Add(b);
                continue;
            }

            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), name));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static string? Rename(string name)
    {
        if (name == "decode_head.linear_fuse.weight")
        {
            return "decode.fuse.weight";
        }

        if (name is "decode_head.classifier.weight" or "decode_head.classifier.bias")
        {
            return "decode.classifier." + name["decode_head.classifier.".Length..];
        }

        if (Regex.Match(name, @"^decode_head\.linear_c\.(\d+)\.proj\.(weight|bias)$") is { Success: true } proj)
        {
            return $"decode.proj.{proj.Groups[1].Value}.{proj.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^segformer\.encoder\.patch_embeddings\.(\d+)\.(proj|layer_norm)\.(weight|bias)$") is { Success: true } embed)
        {
            return $"stages.{embed.Groups[1].Value}.{(embed.Groups[2].Value == "proj" ? "embed" : "embed_norm")}.{embed.Groups[3].Value}";
        }

        if (Regex.Match(name, @"^segformer\.encoder\.layer_norm\.(\d+)\.(weight|bias)$") is { Success: true } norm)
        {
            return $"stages.{norm.Groups[1].Value}.norm.{norm.Groups[2].Value}";
        }

        if (Regex.Match(name, @"^segformer\.encoder\.block\.(\d+)\.(\d+)\.(.+)\.(weight|bias)$") is { Success: true } block)
        {
            string module = block.Groups[3].Value switch
            {
                "layer_norm_1" => "norm1",
                "layer_norm_2" => "norm2",
                "attention.self.query" => "attn.q",
                "attention.self.key" => "attn.k",
                "attention.self.value" => "attn.v",
                "attention.self.sr" => "attn.sr",
                "attention.self.layer_norm" => "attn.sr_norm",
                "attention.output.dense" => "attn.o",
                "mlp.dense1" => "mlp.up",
                "mlp.dwconv.dwconv" => "mlp.dw",
                "mlp.dense2" => "mlp.down",
                _ => "",
            };
            return module.Length == 0 ? null : $"stages.{block.Groups[1].Value}.blocks.{block.Groups[2].Value}.{module}.{block.Groups[4].Value}";
        }

        return null;
    }
}
