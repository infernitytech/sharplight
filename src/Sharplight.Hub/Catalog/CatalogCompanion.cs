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

    /// <summary>
    /// Further roles the same files fill (a pipeline folder that serves as the text encoder and the VAE): downloaded once, passed in
    /// every role.
    /// </summary>
    public IReadOnlyList<string> AdditionalRoles { get; init; } = [];

    /// <summary><see cref="Role"/> followed by <see cref="AdditionalRoles"/>.</summary>
    public IEnumerable<string> AllRoles => [Role, .. AdditionalRoles];

    /// <summary>
    /// Whether the files go beside the variant's own files instead of being passed in <see cref="Role"/>: for a release whose files
    /// are split over repositories and whose importer expects them in one folder (SeedVR2 7B's DiT and VAE with the text embeddings of
    /// the 3B repository). <see cref="Role"/> still names what the files are (older hosts, which do not know this property, pass them
    /// in that role).
    /// </summary>
    public bool IntoVariantFolder { get; init; }

    /// <summary>What the companion is, e.g. <c>T5-XXL encoder (Q8_0)</c>.</summary>
    public required string Label { get; init; }
}
