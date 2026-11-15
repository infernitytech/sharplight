using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// DeepSeek-OCR (remote code, <c>deepseek_vl_v2</c> with <c>DeepseekOCRForCausalLM</c>), DeepSeek-OCR-2 (<c>deepseek_ocr2</c>; remote-code
/// <c>DeepseekOCR2ForCausalLM</c> checkpoints use <c>deepseek_vl_v2</c> too) and Unlimited-OCR (<c>unlimited_ocr</c>) → canonical <c>decoder</c>:
/// the language model of <see cref="DeepseekOcrTextHfMapping"/>. Checkpoints come with the remote-code names (<c>model.layers.*</c>, which
/// transformers also writes) or transformers' own (<c>model.language_model.*</c>). The SAM + CLIP image tower of DeepSeek-OCR and Unlimited-OCR
/// and DeepSeek-OCR-2's SAM + Qwen2 query encoder tower come from <see cref="DeepseekOcrVisionHfMapping"/>. DeepSeek-VL2 checkpoints (<c>deepseek_vl_v2</c> with latent
/// attention) are rejected. Unlimited-OCR's "reference" sliding window (each token sees the prompt plus its last <c>sliding_window</c> tokens) is
/// not modelled: attention is full, as in llama.cpp, which equals it until the generated text exceeds the window.
/// </summary>
internal sealed class DeepseekOcrHfMapping : IHfArchitectureMapping
{
    private const string LanguageModelPrefix = "model.language_model.";
    private readonly DeepseekOcrTextHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["deepseek_vl_v2", "deepseek_ocr2", "unlimited_ocr", "deepseek_ocr2_text", "unlimited_ocr_text"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HuggingFaceImportOptions options)
    {
        if (config.GetBool("use_mla") == true || config.GetInt("kv_lora_rank") is > 0)
        {
            throw new NotSupportedException("deepseek_vl_v2 with latent attention (DeepSeek-VL2) is not supported; only the DeepSeek-OCR models are.");
        }

        var md = _text.MapHyperParameters(config, options);
        if (!DeepseekOcrVisionHfMapping.HasTower(config))
        {
            return md;
        }

        var withVision = new Dictionary<string, object>(md);
        DeepseekOcrVisionHfMapping.AddVisionHyperParameters(config, withVision, Architecture, config.Int("hidden_size"));
        return withVision;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var text = tensors
            .Where(t => !IsTowerTensor(t.Key))
            .ToDictionary(
                t => t.Key.StartsWith(LanguageModelPrefix, StringComparison.Ordinal) ? "model." + t.Key[LanguageModelPrefix.Length..] : t.Key,
                t => t.Value,
                StringComparer.Ordinal);
        var weights = _text.MapWeights(config, text, options);
        if (!DeepseekOcrVisionHfMapping.HasTower(config))
        {
            return weights;
        }

        var all = new Dictionary<string, WeightInfo>(weights, StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (IsTowerTensor(name))
            {
                foreach (var w in DeepseekOcrVisionHfMapping.MapTowerTensor(name, tensor))
                {
                    all[w.Name] = w;
                }
            }
        }

        return all;
    }

    /// <summary>Whether a checkpoint tensor belongs to the image tower, its projector or its layout embeddings.</summary>
    internal static bool IsTowerTensor(string name)
        => name.StartsWith("model.sam_model.", StringComparison.Ordinal)
            || name.StartsWith("model.vision_model.", StringComparison.Ordinal)
            || name.StartsWith("model.qwen2_model.", StringComparison.Ordinal)
            || name.StartsWith("model.projector.", StringComparison.Ordinal)
            || name.StartsWith("model.vision_tower.", StringComparison.Ordinal)
            || name.StartsWith("model.multi_modal_projector.", StringComparison.Ordinal)
            || name is "model.image_newline" or "model.view_seperator" or "model.view_separator";
}
