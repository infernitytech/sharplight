namespace Sharplight.Architectures;

/// <summary>How a mixture-of-experts router turns its logits into expert scores.</summary>
internal enum MoeScoreFunction
{
    /// <summary>Softmax over all experts (Mixtral, Qwen MoE, OLMoE).</summary>
    Softmax,

    /// <summary>Independent sigmoid per expert (DeepSeek-V3 style aux-loss-free routing, Flint).</summary>
    Sigmoid,
}
