namespace Sharplight.Hub.Catalog;

/// <summary>
/// A file set a variant needs besides its own files (a text encoder, VAE, vocoder, vision projector, second transformer...), built
/// into the same package as the auxiliary source of <see cref="Role"/>.
/// </summary>
public sealed record CatalogCompanion : CatalogFileSet
{
    /// <summary>Unique within the entry; variants refer to it.</summary>
    public required string Id { get; init; }

    /// <summary>The auxiliary role (<c>Sharplight.Vision.AuxiliaryRoles</c>: <c>text-encoder</c>, <c>vae</c>, <c>vocoder</c>, <c>mmproj</c>, ...).</summary>
    public required string Role { get; init; }

    /// <summary>What the companion is, e.g. <c>T5-XXL encoder (Q8_0)</c>.</summary>
    public required string Label { get; init; }
}
