namespace Sharplight.Import.Diffusers;

/// <summary>Hyper-parameters of a CLIP text encoder, written as the canonical <c>latent_diffusion.{component}.*</c> keys.</summary>
internal sealed record ClipTextConfig
{
    /// <summary>Transformer layers.</summary>
    public required int LayerCount { get; init; }
    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }
    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }
    /// <summary>MLP width.</summary>
    public required int FfnHiddenSize { get; init; }
    /// <summary>MLP activation: "quick_gelu" (OpenAI CLIP) or "gelu" (OpenCLIP).</summary>
    public required string Activation { get; init; }
    /// <summary>Tokens per sequence (77).</summary>
    public int ContextLength { get; init; } = 77;
    /// <summary>LayerNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;
    /// <summary>
    /// How many layers from the end the conditioning is taken (1: the last layer; 2: the penultimate one, the SD 2.x / SDXL
    /// default).
    /// </summary>
    public int ClipSkip { get; init; } = 1;
    /// <summary>Whether the final LayerNorm is applied to the conditioning (SD 1.x / 2.x: yes; SDXL: no).</summary>
    public bool FinalNorm { get; init; } = true;

    /// <summary>Writes the keys under <c>{prefix}{component}.</c>.</summary>
    public void WriteTo(IDictionary<string, object> metadata, string prefix, string component)
    {
        string p = $"{prefix}{component}.";
        metadata[p + "layer_count"] = LayerCount;
        metadata[p + "hidden_size"] = HiddenSize;
        metadata[p + "head_count"] = HeadCount;
        metadata[p + "ffn_hidden_size"] = FfnHiddenSize;
        metadata[p + "activation"] = Activation;
        metadata[p + "context_length"] = ContextLength;
        metadata[p + "norm_eps"] = NormEps;
        metadata[p + "clip_skip"] = ClipSkip;
        metadata[p + "final_norm"] = FinalNorm;
    }
}
