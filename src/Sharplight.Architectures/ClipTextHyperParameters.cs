using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The hyper-parameters of a CLIP text encoder (canonical keys <c>latent_diffusion.{component}.*</c>).</summary>
internal sealed record ClipTextHyperParameters
{
    /// <summary>Transformer layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>MLP activation: <c>quick_gelu</c> or <c>gelu</c>.</summary>
    public required string Activation { get; init; }

    /// <summary>Tokens per prompt.</summary>
    public required int ContextLength { get; init; }

    /// <summary>LayerNorm epsilon.</summary>
    public required float NormEps { get; init; }

    /// <summary>Layers from the end whose output is the conditioning (1 = the last).</summary>
    public required int ClipSkip { get; init; }

    /// <summary>Whether the final LayerNorm is applied to the conditioning.</summary>
    public required bool FinalNorm { get; init; }

    /// <summary>Reads the keys under <c>{prefix}{component}.</c>.</summary>
    /// <param name="source">The canonical source.</param>
    /// <param name="prefix">Architecture prefix (<c>latent_diffusion.</c>).</param>
    /// <param name="component">Encoder key (<c>text_encoder</c>).</param>
    public static ClipTextHyperParameters Read(ModelSource source, string prefix, string component)
    {
        string p = $"{prefix}{component}.";
        return new ClipTextHyperParameters
        {
            LayerCount = source.Get<int>(p + "layer_count"),
            HiddenSize = source.Get<int>(p + "hidden_size"),
            HeadCount = source.Get<int>(p + "head_count"),
            Activation = source.Get(p + "activation", "quick_gelu"),
            ContextLength = source.Get(p + "context_length", 77),
            NormEps = source.Get(p + "norm_eps", 1e-5f),
            ClipSkip = source.Get(p + "clip_skip", 1),
            FinalNorm = source.Get(p + "final_norm", true),
        };
    }
}
