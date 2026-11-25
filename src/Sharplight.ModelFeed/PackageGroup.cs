namespace Sharplight.ModelFeed;

/// <summary>
/// The model a package is a variant of (<c>group</c> in manifests and catalog pages, docs/model-feed.md "Model groups"): every package
/// with the same <see cref="Id"/> is one quantization or format of the same model, so a host lists the model once and offers its
/// variants side by side. The package id is <c>&lt;<see cref="Id"/>&gt;.&lt;<see cref="Variant"/>&gt;</c>.
/// </summary>
public sealed record PackageGroup
{
    /// <summary>Model group id: the verified-catalog entry id (<c>qwen3.5-122b-a10b</c>); lower case <c>[a-z0-9.-]</c>.</summary>
    public required string Id { get; init; }

    /// <summary>Model name shared by every variant (<c>Qwen3.5 122B-A10B</c>).</summary>
    public string? Name { get; init; }

    /// <summary>Variant id within the group: the catalog variant id (<c>ud-q4-k-xl-gguf</c>).</summary>
    public required string Variant { get; init; }

    /// <summary>Variant label (<c>UD-Q4_K_XL GGUF</c>).</summary>
    public string? Label { get; init; }

    /// <summary>Upstream file format the package was converted from (<c>gguf</c>, <c>safetensors</c>, ...).</summary>
    public string? Format { get; init; }

    /// <summary>Weight quantization of the variant (<c>UD-Q4_K_XL</c>, <c>FP8</c>, <c>MXFP4</c>).</summary>
    public string? Quantization { get; init; }

    /// <summary>Average stored bits per weight element of the package (weight bytes · 8 / parameters), or <see langword="null"/> when unknown.</summary>
    public double? BitsPerWeight { get; init; }

    /// <summary>The variant a host offers first for this model (at most one per group).</summary>
    public bool Recommended { get; init; }

    /// <summary>Quality relative to <see cref="Reference"/> (<see cref="PackageQualityTiers"/>), or <see langword="null"/>.</summary>
    public string? Quality { get; init; }

    /// <summary>One sentence on the quality relative to <see cref="Reference"/>, or <see langword="null"/>.</summary>
    public string? QualityNote { get; init; }

    /// <summary>The reference precision quality is measured against (<c>BF16</c>, or <c>MXFP4</c> for a model released in it).</summary>
    public string? Reference { get; init; }

    /// <summary>Bytes of the package file (<c>.slm</c>).</summary>
    public long Size { get; init; }

    /// <summary>Bytes of the upstream files the package was built from (with companions).</summary>
    public long SourceSize { get; init; }
}
