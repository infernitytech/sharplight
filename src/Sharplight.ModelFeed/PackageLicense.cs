namespace Sharplight.ModelFeed;

/// <summary>License of a model, from its model card at the pinned revision.</summary>
public sealed record PackageLicense
{
    /// <summary>SPDX identifier (lower case, as on the Hub), or a custom id.</summary>
    public string? Spdx { get; init; }

    /// <summary>License name.</summary>
    public string? Name { get; init; }

    /// <summary>License text URL.</summary>
    public string? Url { get; init; }

    /// <summary>Whether commercial use is allowed, when known.</summary>
    public bool? CommercialUse { get; init; }

    /// <summary>Whether the terms must be accepted before install.</summary>
    public bool Gated { get; init; }

    /// <summary>Required attribution text.</summary>
    public string? Attribution { get; init; }
}
