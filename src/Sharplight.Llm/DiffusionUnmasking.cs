namespace Sharplight.Llm;

/// <summary>How a masked-diffusion step ranks the masked positions it may reveal (the most confident ones are revealed first).</summary>
public enum DiffusionUnmasking
{
    /// <summary>Probability of the predicted token (LLaDA's low-confidence remasking, Dream's <c>maskgit_plus</c>).</summary>
    Confidence,
    /// <summary>Difference between the probabilities of the two most likely tokens (Dream's <c>topk_margin</c>).</summary>
    Margin,
    /// <summary>Negative entropy of the predicted distribution (Dream's <c>entropy</c>): the sharpest distributions first.</summary>
    Entropy,
}
