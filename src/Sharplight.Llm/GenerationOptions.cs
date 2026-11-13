namespace Sharplight.Llm;

/// <summary>Limits, sampling and stop conditions of one <see cref="Generator"/> run.</summary>
public sealed record GenerationOptions
{
    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxNewTokens { get; init; } = 256;
    /// <summary>Token sampling configuration.</summary>
    public SamplingOptions Sampling { get; init; } = new();
    /// <summary>Stop on end-of-generation tokens.</summary>
    public bool StopAtEog { get; init; } = true;
    /// <summary>Generation ends after the output text ends with any of these strings (the string itself is still emitted).</summary>
    public IReadOnlyList<string> StopStrings { get; init; } = [];
}
