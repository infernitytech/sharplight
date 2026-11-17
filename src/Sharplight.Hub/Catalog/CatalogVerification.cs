namespace Sharplight.Hub.Catalog;

/// <summary>How and when an entry was checked.</summary>
public sealed record CatalogVerification
{
    /// <summary>The reference implementation (<c>llama.cpp</c>, <c>transformers</c>, <c>diffusers</c>, ...), or <see langword="null"/>.</summary>
    public string? Reference { get; init; }

    /// <summary>The result in one sentence (identical greedy ids, PSNR, relative error...).</summary>
    public string Summary { get; init; } = "";

    /// <summary>Execution providers it was run on (<c>CPU</c>, <c>Vulkan</c>).</summary>
    public IReadOnlyList<string> Providers { get; init; } = [];

    /// <summary>The date of the check, <c>yyyy-MM-dd</c>, or empty when not recorded.</summary>
    public string Date { get; init; } = "";

    /// <summary>The tests that check it (test class or method names).</summary>
    public IReadOnlyList<string> Tests { get; init; } = [];
}
