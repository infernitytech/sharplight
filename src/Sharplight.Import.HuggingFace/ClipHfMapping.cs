using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face dual-encoder image + text models (<c>clip</c>: CLIPModel; <c>siglip</c>: SiglipModel) → canonical <c>image_text</c> (documented on
/// <c>ImageTextAdapter</c>). The checkpoint's <c>vision_config</c> and <c>text_config</c> give the two towers (SigLIP checkpoints leave most keys at
/// transformers' defaults: 12 layers of 768, GELU-tanh, eps 1e-6, 64 text tokens); CLIP pools the end-of-text row of a causal text tower and projects
/// both towers, SigLIP pools the last row of a bidirectional text tower through a linear head and the image tower with an attention-pooling head.
/// </summary>
internal sealed partial class ClipHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "image_text";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["clip", "siglip"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.ModelType == "clip" && Tower(c, "vision_config").GetStringOr("hidden_act", "quick_gelu") is not ("quick_gelu" or "gelu"))
        {
            yield return "clip: only quick_gelu and gelu activations are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        bool siglip = c.ModelType == "siglip";
        var vision = Tower(c, "vision_config");
        var text = Tower(c, "text_config");
        string a = Architecture + ".";
        string defaultAct = siglip ? "gelu_pytorch_tanh" : "quick_gelu";
        int imageSize = vision.GetIntOr("image_size", 224);
        var processor = HfImageProcessorConfig.Read(c.Directory, imageSize);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "vision.layer_count"] = vision.GetIntOr("num_hidden_layers", 12),
            [a + "vision.hidden_size"] = vision.GetIntOr("hidden_size", 768),
            [a + "vision.ffn_hidden_size"] = vision.GetIntOr("intermediate_size", 3072),
            [a + "vision.attention.head_count"] = vision.GetIntOr("num_attention_heads", 12),
            [a + "vision.norm_eps"] = (float)vision.GetDoubleOr("layer_norm_eps", siglip ? 1e-6 : 1e-5),
            [a + "vision.mlp.activation"] = Activation(vision.GetStringOr("hidden_act", defaultAct)),
            [a + "vision.patch_size"] = vision.GetIntOr("patch_size", siglip ? 16 : 32),
            [a + "vision.image_size"] = HfImageProcessorConfig.InputSize(processor, imageSize),
            [a + "vision.pooling"] = siglip ? "attention_head" : "cls",
            [a + "text.layer_count"] = text.GetIntOr("num_hidden_layers", 12),
            [a + "text.hidden_size"] = text.GetIntOr("hidden_size", siglip ? 768 : 512),
            [a + "text.ffn_hidden_size"] = text.GetIntOr("intermediate_size", siglip ? 3072 : 2048),
            [a + "text.attention.head_count"] = text.GetIntOr("num_attention_heads", siglip ? 12 : 8),
            [a + "text.norm_eps"] = (float)text.GetDoubleOr("layer_norm_eps", siglip ? 1e-6 : 1e-5),
            [a + "text.mlp.activation"] = Activation(text.GetStringOr("hidden_act", defaultAct)),
            [a + "text.context_length"] = text.GetIntOr("max_position_embeddings", siglip ? 64 : 77),
            [a + "text.causal"] = !siglip,
            [a + "text.pool"] = siglip ? "last" : "eos",
            [a + "text.pad_token_id"] = text.GetIntOr("pad_token_id", siglip ? 1 : 49407),
            [a + "scoring"] = siglip ? "sigmoid" : "softmax",
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        bool siglip = c.ModelType == "siglip";
        var vision = Tower(c, "vision_config");
        int vh = vision.GetIntOr("hidden_size", 768);
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName;
            switch (name)
            {
                case "logit_scale":
                    collector.Add(HfTransforms.ToF32(info, "text.logit_scale", new Shape(1)));
                    continue;
                case "logit_bias":
                    collector.Add(HfTransforms.ToF32(info, "text.logit_bias", new Shape(1)));
                    continue;
                case "visual_projection.weight":
                    collector.Add(info.As("vision.projection.weight"));
                    continue;
                case "text_projection.weight":
                    collector.Add(info.As("text.projection.weight"));
                    continue;
                case "vision_model.embeddings.class_embedding":
                    collector.Add(HfTransforms.ToF32(info, "vision.class_embedding", new Shape(vh)));
                    continue;
                case "vision_model.embeddings.patch_embedding.weight":
                    collector.Add(info.As("vision.patch_embed.weight", new Shape(vh, info.Shape.Elements() / vh)));
                    continue;
                case "vision_model.embeddings.patch_embedding.bias":
                    collector.Add(HfTransforms.Plain(info, "vision.patch_embed.bias"));
                    continue;
                case "vision_model.embeddings.position_embedding.weight":
                    collector.Add(HfTransforms.ToF32(info, "vision.position_embedding.weight"));
                    continue;
                case "vision_model.pre_layrnorm.weight" or "vision_model.pre_layrnorm.bias":
                    collector.Add(HfTransforms.Plain(info, "vision.embedding_norm." + Last(name)));
                    continue;
                case "vision_model.post_layernorm.weight" or "vision_model.post_layernorm.bias":
                    collector.Add(HfTransforms.Plain(info, "vision.output_norm." + Last(name)));
                    continue;
                case "vision_model.head.probe":
                    collector.Add(HfTransforms.ToF32(info, "vision.head.probe", new Shape(vh)));
                    continue;
                case "vision_model.head.attention.in_proj_weight" or "vision_model.head.attention.in_proj_bias":
                    foreach (var part in SplitQkv(info, "vision.head.attention", vh))
                    {
                        collector.Add(part);
                    }

                    continue;
                case "vision_model.head.attention.out_proj.weight" or "vision_model.head.attention.out_proj.bias":
                    collector.Add(Linear(info, "vision.head.attention.o." + Last(name)));
                    continue;
                case "vision_model.head.layernorm.weight" or "vision_model.head.layernorm.bias":
                    collector.Add(HfTransforms.Plain(info, "vision.head.norm." + Last(name)));
                    continue;
                case "vision_model.head.mlp.fc1.weight" or "vision_model.head.mlp.fc1.bias":
                    collector.Add(Linear(info, "vision.head.mlp.up." + Last(name)));
                    continue;
                case "vision_model.head.mlp.fc2.weight" or "vision_model.head.mlp.fc2.bias":
                    collector.Add(Linear(info, "vision.head.mlp.down." + Last(name)));
                    continue;
                case "text_model.embeddings.token_embedding.weight":
                    collector.Add(info.As("text.token_embedding.weight"));
                    continue;
                case "text_model.embeddings.position_embedding.weight":
                    collector.Add(HfTransforms.ToF32(info, "text.position_embedding.weight"));
                    continue;
                case "text_model.final_layer_norm.weight" or "text_model.final_layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, "text.final_norm." + Last(name)));
                    continue;
                case "text_model.head.weight" or "text_model.head.bias":
                    collector.Add(Linear(info, "text.projection." + Last(name)));
                    continue;
                case var _ when name.EndsWith("embeddings.position_ids", StringComparison.Ordinal):
                    continue;
            }

            var m = LayerName().Match(name);
            if (m.Success && s_layerModules.TryGetValue(m.Groups[3].Value, out var module))
            {
                string tower = m.Groups[1].Value == "vision" ? "vision" : "text";
                collector.Add(Linear(info, $"{tower}.layers.{m.Groups[2].Value}.{module}.{m.Groups[4].Value}"));
                continue;
            }

            collector.AddUnmapped(rawName);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    [GeneratedRegex(@"^(vision|text)_model\.encoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    private static readonly Dictionary<string, string> s_layerModules = new(StringComparer.Ordinal)
    {
        ["layer_norm1"] = "input_norm",
        ["layer_norm2"] = "post_attention_norm",
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.out_proj"] = "attention.o",
        ["mlp.fc1"] = "mlp.up",
        ["mlp.fc2"] = "mlp.down",
    };

    private static string Last(string name) => name[(name.LastIndexOf('.') + 1)..];

    private static WeightInfo Linear(WeightInfo info, string name) => info.Shape.Rank == 1 ? HfTransforms.Plain(info, name) : info.As(name);

    // The attention head's fused in_proj (rows q, k, v) as the three projections.
    private static IEnumerable<WeightInfo> SplitQkv(WeightInfo info, string prefix, int hidden)
    {
        bool bias = info.Shape.Rank == 1;
        string kind = bias ? "bias" : "weight";
        string[] names = ["q", "k", "v"];
        for (int i = 0; i < 3; i++)
        {
            int part = i;
            var shape = bias ? new Shape(hidden) : new Shape(hidden, hidden);
            yield return WeightInfo.Derived($"{prefix}.{names[i]}.{kind}", DataType.F32, shape, () =>
            {
                var all = HfTransforms.DecodeF32(info);
                return HfTransforms.EncodeF32(all.AsSpan(part * (int)shape.Elements(), (int)shape.Elements()).ToArray());
            });
        }
    }

    private static string Activation(string hf) => hf switch
    {
        "gelu" => "gelu",
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
        "quick_gelu" => "quick_gelu",
        "relu" => "relu",
        "silu" or "swish" => "silu",
        _ => throw new NotSupportedException($"clip/siglip: activation '{hf}' is not supported."),
    };

    private static SubConfig Tower(HfConfig c, string key)
        => new(c.Root.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Object ? value : default);

    private readonly record struct SubConfig(JsonElement Value)
    {
        public int GetIntOr(string key, int fallback) => Has(key) ? (int)Value.GetProperty(key).GetDouble() : fallback;

        public double GetDoubleOr(string key, double fallback) => Has(key) ? Value.GetProperty(key).GetDouble() : fallback;

        public string GetStringOr(string key, string fallback) => Has(key) && Value.GetProperty(key).ValueKind == JsonValueKind.String ? Value.GetProperty(key).GetString()! : fallback;

        private bool Has(string key) => Value.ValueKind == JsonValueKind.Object && Value.TryGetProperty(key, out var v) && v.ValueKind != JsonValueKind.Null;
    }
}
