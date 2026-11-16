using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face TrOCR (<c>vision-encoder-decoder</c> with a <c>vit</c> encoder and a <c>trocr</c> decoder: VisionEncoderDecoderModel) → canonical
/// <c>encoder_decoder</c> with <c>encoder.input</c> "image" (documented on <c>EncoderDecoderAdapter</c>). The ViT encoder is mapped by <see cref="VitHfMapping"/>
/// (weights under <c>vision.</c>, its stored image processor), the decoder like BART's post-norm decoder (learned positions without the offset of 2,
/// embedding LayerNorm, no final norm, tied output embedding). The tokenizer is read from <c>tokenizer.json</c> (checkpoints that only have vocab.json and
/// merges.txt need one, e.g. written by <c>tools/ocr-reference.py</c>).
/// </summary>
internal sealed class TrOcrHfMapping : IHfArchitectureMapping
{
    private const int PositionOffset = 2;

    private readonly VitHfMapping _vision = new();

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["vision-encoder-decoder"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        string encoder = c.Get("encoder") is { ValueKind: JsonValueKind.Object } e && e.TryGetProperty("model_type", out var et) ? et.GetString() ?? "" : "";
        string decoder = c.Get("decoder") is { ValueKind: JsonValueKind.Object } d && d.TryGetProperty("model_type", out var dt) ? dt.GetString() ?? "" : "";
        if (encoder != "vit" || decoder != "trocr")
        {
            yield return $"vision-encoder-decoder: only a vit encoder with a trocr decoder is supported (this one has '{encoder}' and '{decoder}').";
            yield break;
        }

        var decoderConfig = c.Get("decoder")!.Value;
        if (decoderConfig.TryGetProperty("use_learned_position_embeddings", out var learned) && learned.ValueKind == JsonValueKind.False)
        {
            yield return "trocr: sinusoidal position embeddings are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        using var vision = Nested(c, "encoder", "vit");
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in _vision.MapHyperParameters(vision, options))
        {
            if (key.StartsWith("vit.", StringComparison.Ordinal))
            {
                md[a + "vision." + key["vit.".Length..]] = value;
            }
            else if (key == CanonicalKeys.ImageProcessor)
            {
                md[key] = value;
            }
        }

        var decoder = c.Get("decoder")!.Value;
        int hidden = decoder.GetProperty("d_model").GetInt32();
        int heads = decoder.GetProperty("decoder_attention_heads").GetInt32();
        int visionHidden = vision.Int("hidden_size");
        string act = decoder.TryGetProperty("activation_function", out var activation) ? activation.GetString() ?? "gelu" : "gelu";
        int patch = vision.Int("patch_size");
        int image = (int)md[a + "vision.image_size"];
        md[a + "hidden_size"] = hidden;
        md[a + "encoder.hidden_size"] = visionHidden;
        md[a + "encoder.input"] = "image";
        md[a + "encoder.layer_count"] = 0;
        md[a + "encoder.rows"] = (image / patch) * (image / patch) + 1;
        md[a + "cross_attention.cached"] = true;
        md[a + "decoder.layer_count"] = decoder.GetProperty("decoder_layers").GetInt32();
        md[a + "attention.head_count"] = heads;
        md[a + "attention.head_dim"] = hidden / heads;
        md[a + "attention.scale"] = 1f / MathF.Sqrt(hidden / heads);
        md[a + "norm"] = "layer";
        md[a + "norm_eps"] = 1e-5f;
        md[a + "pre_norm"] = false;
        md[a + "embedding_scale"] = decoder.TryGetProperty("scale_embedding", out var scale) && scale.ValueKind == JsonValueKind.True ? MathF.Sqrt(hidden) : 1f;
        md[a + "mlp.activation"] = act switch
        {
            "gelu" => "gelu",
            "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
            "relu" => "relu",
            "silu" or "swish" => "silu",
            _ => throw new NotSupportedException($"trocr: activation '{act}' is not supported."),
        };
        md[a + "decoder_start_token"] = decoder.GetProperty("decoder_start_token_id").GetInt32();
        md[a + "end_token"] = decoder.GetProperty("eos_token_id").GetInt32();
        md[a + "context_length"] = decoder.GetProperty("max_position_embeddings").GetInt32();
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        using var vision = Nested(c, "encoder", "vit");
        var encoderTensors = tensors.Where(t => t.Key.StartsWith("encoder.", StringComparison.Ordinal)).ToDictionary(t => t.Key["encoder.".Length..], t => t.Value, StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (name, weight) in _vision.MapWeights(vision, encoderTensors, options))
        {
            collector.Add(weight.As("vision." + name));
        }

        const string Prefix = "decoder.model.decoder.";
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.StartsWith("encoder.", StringComparison.Ordinal))
            {
                continue;
            }

            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith(Prefix, StringComparison.Ordinal) ? rawName[Prefix.Length..] : rawName;
            switch (name)
            {
                case "embed_tokens.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "embed_positions.weight":
                    collector.Add(info.RowSlice("decoder.position_embedding.weight", PositionOffset, info.Shape[0].Fixed - PositionOffset));
                    continue;
                case "layernorm_embedding.weight" or "layernorm_embedding.bias":
                    collector.Add(HfTransforms.Plain(info, name.Replace("layernorm_embedding", "decoder.embedding_norm", StringComparison.Ordinal)));
                    continue;
                case "decoder.output_projection.weight" or "output_projection.weight":
                    collector.Add(info.As("output.weight"));
                    continue;
            }

            var m = System.Text.RegularExpressions.Regex.Match(name, @"^layers\.(\d+)\.(.+)\.(weight|bias)$");
            if (!m.Success || !BartHfMapping.LayerModules.TryGetValue(m.Groups[2].Value, out var module))
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            string canonical = $"decoder.layers.{m.Groups[1].Value}.{module}.{m.Groups[3].Value}";
            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // The nested encoder configuration as a standalone configuration of its model type.
    private static HfConfig Nested(HfConfig c, string key, string modelType)
    {
        var nested = JsonNode.Parse(c.Get(key)!.Value.GetRawText())!.AsObject();
        nested["model_type"] = modelType;
        nested.Remove("id2label");
        nested.Remove("label2id");
        return HfConfig.Parse(nested.ToJsonString(), c.Directory);
    }
}
