namespace Sharplight.ModelFeed;

/// <summary>Capability names of <see cref="PackageModel.Capabilities"/>.</summary>
public static class PackageCapabilities
{
    /// <summary>Image input.</summary>
    public const string Vision = "vision";

    /// <summary>Tool calls.</summary>
    public const string Tools = "tools";

    /// <summary>Reasoning (thinking) output.</summary>
    public const string Reasoning = "reasoning";

    /// <summary>Structured (JSON schema) output.</summary>
    public const string StructuredOutput = "structured-output";

    /// <summary>Usable as a speculative draft model.</summary>
    public const string SpeculativeDraft = "speculative-draft";
}
