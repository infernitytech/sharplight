namespace Sharplight.ModelFeed;

/// <summary>Input and output modalities of a model.</summary>
public sealed record PackageModalities
{
    /// <summary>Input modalities (<c>text</c>, <c>image</c>, <c>audio</c>, ...).</summary>
    public IReadOnlyList<string> Input { get; init; } = [];

    /// <summary>Output modalities.</summary>
    public IReadOnlyList<string> Output { get; init; } = [];
}
