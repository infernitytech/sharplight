namespace Sharplight.Models;

/// <summary>How far Sharplight supports an architecture or a weight format, as an importer reports it before a model is opened.</summary>
public enum SupportLevel
{
    /// <summary>No importer mapping or kernel handles it; importing or loading fails.</summary>
    Unsupported,

    /// <summary>Imports and runs, with a documented restriction (e.g. a feature that runs only in a reduced form).</summary>
    Partial,

    /// <summary>Imports and runs.</summary>
    Supported,
}
