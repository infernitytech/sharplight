namespace Sharplight.Llm;

public sealed record GenerationOptions
{
    public int MaxNewTokens { get; init; } = 256;
    public SamplingOptions Sampling { get; init; } = new();
    /// <summary>Stop on end-of-generation tokens.</summary>
    public bool StopAtEog { get; init; } = true;
    public IReadOnlyList<string> StopStrings { get; init; } = [];
}
