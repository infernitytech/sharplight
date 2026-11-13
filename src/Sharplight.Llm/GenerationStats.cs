namespace Sharplight.Llm;

/// <summary>Token counts and timings of a finished <see cref="Generator"/> run.</summary>
/// <param name="PromptTokens">Rows processed during prefill (image rows included).</param>
/// <param name="GeneratedTokens">Tokens sampled, including a terminating end-of-generation token.</param>
/// <param name="PrefillTime">Time to process the prompt.</param>
/// <param name="DecodeTime">Time spent generating after prefill.</param>
public sealed record GenerationStats(int PromptTokens, int GeneratedTokens, TimeSpan PrefillTime, TimeSpan DecodeTime)
{
    /// <summary>Prompt processing speed.</summary>
    public double PrefillTokensPerSecond => PromptTokens / Math.Max(1e-9, PrefillTime.TotalSeconds);
    /// <summary>Generation speed (the first token belongs to prefill and is not counted).</summary>
    public double DecodeTokensPerSecond => Math.Max(0, GeneratedTokens - 1) / Math.Max(1e-9, DecodeTime.TotalSeconds);
}
