namespace Sharplight.ModelFeed;

/// <summary>Display data of a package.</summary>
public sealed record PackageDisplay
{
    /// <summary>Model name (<c>Qwen3 0.6B</c>).</summary>
    public string? Name { get; init; }

    /// <summary>Variant label (<c>Q8_0</c>).</summary>
    public string? Variant { get; init; }

    /// <summary>Model family.</summary>
    public string? Family { get; init; }

    /// <summary>One-paragraph summary.</summary>
    public string? Summary { get; init; }

    /// <summary>Icon URL.</summary>
    public string? Icon { get; init; }

    /// <summary>Tags (<c>reasoning</c>, <c>tools</c>, ...).</summary>
    public IReadOnlyList<string> Tags { get; init; } = [];
}
