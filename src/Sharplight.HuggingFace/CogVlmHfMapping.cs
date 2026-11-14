using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// CogVLM (remote-code <c>CogVLMForCausalLM</c>, whose config has no <c>model_type</c>; THUDM/cogvlm-chat-hf) → canonical <c>decoder</c>: the
/// Llama language model with visual experts of <see cref="CogVlmTextHfMapping"/> plus the EVA2-CLIP tower and GLU projector of
/// <see cref="CogVlmVisionHfMapping"/> when the checkpoint has a <c>vision_config</c>.
/// </summary>
internal sealed class CogVlmHfMapping : IHfArchitectureMapping
{
    private readonly CogVlmTextHfMapping _text = new();

    /// <inheritdoc/>
    public string Architecture => _text.Architecture;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["cogvlm"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig config, HfImportOptions options)
    {
        var md = new Dictionary<string, object>(_text.MapHyperParameters(config, options), StringComparer.Ordinal);
        CogVlmVisionHfMapping.AddVisionHyperParameters(config, md, Architecture, config.Int("hidden_size"));
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig config, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var text = tensors.Where(t => !CogVlmVisionHfMapping.IsVisionTensor(t.Key)).ToDictionary(t => t.Key, t => t.Value, StringComparer.Ordinal);
        var weights = new Dictionary<string, WeightInfo>(_text.MapWeights(config, text, options), StringComparer.Ordinal);
        if (QwenVisionHfMapping.VisionConfig(config) is null)
        {
            return weights;
        }

        foreach (var (name, tensor) in tensors)
        {
            if (CogVlmVisionHfMapping.IsVisionTensor(name) && CogVlmVisionHfMapping.MapVisionTensor(name, tensor) is { } w)
            {
                weights[w.Name] = w;
            }
        }

        if (tensors.TryGetValue("model.vision.patch_embedding.position_embedding.weight", out var positions))
        {
            var classPosition = CogVlmVisionHfMapping.ClassPosition(positions);
            weights[classPosition.Name] = classPosition;
        }

        return weights;
    }
}
