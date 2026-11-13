namespace Sharplight.Llm;

public sealed record GenerationStats(int PromptTokens, int GeneratedTokens, TimeSpan PrefillTime, TimeSpan DecodeTime)
{
    public double PrefillTokensPerSecond => PromptTokens / Math.Max(1e-9, PrefillTime.TotalSeconds);
    public double DecodeTokensPerSecond => Math.Max(0, GeneratedTokens - 1) / Math.Max(1e-9, DecodeTime.TotalSeconds);
}
