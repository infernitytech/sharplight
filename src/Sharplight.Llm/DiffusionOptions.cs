namespace Sharplight.Llm;

/// <summary>
/// Settings of masked-diffusion decoding (<see cref="DiffusionGenerator"/>): the generated span of <see cref="GenerationOptions.MaxNewTokens"/>
/// tokens starts as mask tokens, and every step runs the model over the whole sequence and reveals some of the masked positions with
/// their most likely token.
/// </summary>
public sealed record DiffusionOptions
{
    /// <summary>Denoising steps in total; 0 (the default): one per generated token.</summary>
    public int Steps { get; init; }

    /// <summary>
    /// Block length (LLaDA's semi-autoregressive schedule): the span is revealed block after block, each block over an equal share of the steps
    /// and an equal share of its tokens per step. 0 (the default): the whole span at once on Dream's timestep schedule (see <see cref="Eps"/>).
    /// The span length must be a multiple of the block length, and the steps a multiple of the block count.
    /// </summary>
    public int BlockLength { get; init; }

    /// <summary>
    /// Final time of the timestep schedule (Dream): step s of n reveals the fraction 1 − t(s+1) / t(s) of the masked positions, with
    /// t(s) = 1 − s / n · (1 − eps); the last step reveals the rest.
    /// </summary>
    public float Eps { get; init; } = 1e-3f;

    /// <summary>How the positions revealed in a step are chosen.</summary>
    public DiffusionUnmasking Unmasking { get; init; } = DiffusionUnmasking.Confidence;
}
