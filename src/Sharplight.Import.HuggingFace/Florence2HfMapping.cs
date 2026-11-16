using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Florence-2 (<c>florence2</c>: Florence2ForConditionalGeneration, the transformers port with a DaViT vision tower, the multi-modal projector and a BART
/// encoder-decoder) → canonical <c>encoder_decoder</c> with <c>encoder.input</c> "image_text" (documented on <c>EncoderDecoderAdapter</c> and <c>DaViTGraph</c>). The language
/// model is mapped by <see cref="BartHfMapping"/> (its tensors live under <c>model.language_model.</c>), the vision tower and projector keep their transformers names under
/// <c>vision.</c>. The task prompts of Florence-2's processor (<c>&lt;OCR&gt;</c>, <c>&lt;CAPTION&gt;</c>, <c>&lt;OD&gt;</c>, ...) are stored with the model as <c>encoder_decoder.prompt.tasks</c>, and
/// the generation settings (forced first token, no-repeat n-gram size) come from the text configuration.
/// </summary>
internal sealed class Florence2HfMapping : IHfArchitectureMapping
{
    private const string LanguagePrefix = "model.language_model.";
    private const int ImageTokens = 577;
    private const int MaxPromptTokens = 128;

    private static readonly (string Task, string Prompt)[] s_tasks =
    [
        ("<OCR>", "What is the text in the image?"),
        ("<OCR_WITH_REGION>", "What is the text in the image, with regions?"),
        ("<CAPTION>", "What does the image describe?"),
        ("<DETAILED_CAPTION>", "Describe in detail what is shown in the image."),
        ("<MORE_DETAILED_CAPTION>", "Describe with a paragraph what is shown in the image."),
        ("<OD>", "Locate the objects with category name in the image."),
        ("<DENSE_REGION_CAPTION>", "Locate the objects in the image, with their descriptions."),
        ("<REGION_PROPOSAL>", "Locate the region proposals in the image."),
        ("<CAPTION_TO_PHRASE_GROUNDING>", "Locate the phrases in the caption: {input}"),
        ("<REFERRING_EXPRESSION_SEGMENTATION>", "Locate {input} in the image with mask"),
        ("<REGION_TO_SEGMENTATION>", "What is the polygon mask of region {input}"),
        ("<OPEN_VOCABULARY_DETECTION>", "Locate {input} in the image."),
        ("<REGION_TO_CATEGORY>", "What is the region {input}?"),
        ("<REGION_TO_DESCRIPTION>", "What does the region {input} describe?"),
        ("<REGION_TO_OCR>", "What text is in the region {input}?"),
    ];

    private readonly BartHfMapping _language = new();

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["florence2"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (c.Get("vision_config") is not { ValueKind: JsonValueKind.Object } vision)
        {
            yield return "florence2: the configuration has no vision_config.";
            yield break;
        }

        if (vision.TryGetProperty("mlp_ratio", out var ratio) && ratio.GetDouble() != 4.0)
        {
            yield return "florence2: only an MLP ratio of 4 is supported.";
        }

        if (vision.TryGetProperty("image_pos_embed", out var position) && position.TryGetProperty("type", out var type) && type.GetString() != "learned_abs_2d")
        {
            yield return $"florence2: the image position embedding '{type.GetString()}' is not supported (learned_abs_2d).";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        using var text = Nested(c, "text_config", "bart");
        var md = new Dictionary<string, object>(_language.MapHyperParameters(text, options), StringComparer.Ordinal);
        var processor = HfImageProcessorConfig.Read(c.Directory, 768);
        var vision = c.Get("vision_config")!.Value;
        string Ints(string key) => string.Join(',', vision.GetProperty(key).EnumerateArray().Select(v => v.GetInt32()));
        md[CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString();
        md[a + "encoder.input"] = "image_text";
        md[a + "encoder.rows"] = ImageTokens + MaxPromptTokens;
        md[a + "cross_attention.cached"] = true;
        md[a + "vision.image_size"] = HfImageProcessorConfig.InputSize(processor, 768);
        md[a + "vision.depths"] = Ints("depths");
        md[a + "vision.embed_dims"] = Ints("embed_dim");
        md[a + "vision.head_counts"] = Ints("num_heads");
        md[a + "vision.group_counts"] = Ints("num_groups");
        md[a + "vision.patch_sizes"] = Ints("patch_size");
        md[a + "vision.strides"] = Ints("patch_stride");
        md[a + "vision.paddings"] = Ints("patch_padding");
        md[a + "vision.pre_norm"] = string.Join(',', vision.GetProperty("patch_prenorm").EnumerateArray().Select(v => v.GetBoolean() ? 1 : 0));
        md[a + "vision.window"] = vision.TryGetProperty("window_size", out var window) ? window.GetInt32() : 12;
        md[a + "vision.norm_eps"] = 1e-5f;
        md[a + "vision.mlp.activation"] = (vision.TryGetProperty("activation_function", out var act) ? act.GetString() : "gelu") switch
        {
            "gelu" => "gelu",
            "gelu_new" or "gelu_pytorch_tanh" => "gelu_tanh",
            var other => throw new NotSupportedException($"florence2: vision activation '{other}' is not supported."),
        };
        md[a + "prompt.max_tokens"] = MaxPromptTokens;
        md[a + "prompt.bos_token"] = text.GetInt("bos_token_id") ?? 0;
        md[a + "prompt.eos_token"] = text.GetInt("eos_token_id") ?? 2;
        md[a + "prompt.location_bins"] = 1000;
        md[a + "prompt.quad_tasks"] = "<OCR_WITH_REGION>";
        md[a + "prompt.tasks"] = new JsonObject(s_tasks.Select(t => KeyValuePair.Create(t.Task, (JsonNode?)JsonValue.Create(t.Prompt)))).ToJsonString();
        md[a + "forced_bos_token"] = text.GetInt("forced_bos_token_id") ?? -1;
        md[a + "no_repeat_ngram_size"] = text.GetInt("no_repeat_ngram_size") ?? 0;
        md[a + "end_token"] = text.GetInt("eos_token_id") ?? 2;
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        using var text = Nested(c, "text_config", "bart");
        var languageTensors = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            if (rawName.StartsWith(LanguagePrefix, StringComparison.Ordinal))
            {
                languageTensors[rawName[LanguagePrefix.Length..]] = tensor;
            }
            else if (rawName.StartsWith("model.vision_tower.", StringComparison.Ordinal))
            {
                collector.Add(Plain(tensor, "vision." + rawName["model.vision_tower.".Length..]));
            }
            else if (rawName.StartsWith("model.multi_modal_projector.", StringComparison.Ordinal))
            {
                string name = rawName["model.multi_modal_projector.".Length..];
                if (name.StartsWith("visual_temporal_embed.", StringComparison.Ordinal))
                {
                    continue;   // the cosine table is a constant of the graph
                }

                collector.Add(Plain(tensor, "vision.projector." + name));
            }
            else if (rawName == "lm_head.weight")
            {
                languageTensors[rawName] = tensor;
            }
            else
            {
                collector.AddUnmapped(rawName);
            }
        }

        foreach (var (name, weight) in _language.MapWeights(text, languageTensors, options))
        {
            collector.Add(weight);
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static WeightInfo Plain(SafetensorsTensor tensor, string name)
    {
        var info = tensor.ToWeightInfo();
        return info.Shape.Rank == 4 ? HfTransforms.ToF32(info, name) : info.Shape.Rank == 1 ? HfTransforms.Plain(info, name) : info.As(name);
    }

    // The nested configuration as a standalone configuration of its model type.
    private static HfConfig Nested(HfConfig c, string key, string modelType)
    {
        var nested = JsonNode.Parse(c.Get(key)!.Value.GetRawText())!.AsObject();
        nested["model_type"] = modelType;
        return HfConfig.Parse(nested.ToJsonString(), c.Directory);
    }
}
