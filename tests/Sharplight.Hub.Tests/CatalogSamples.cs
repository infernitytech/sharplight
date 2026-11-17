using Sharplight.Hub.Catalog;

namespace Sharplight.Hub.Tests;

/// <summary>Small catalogs for the catalog tests.</summary>
internal static class CatalogSamples
{
    public const string Commit = "0123456789abcdef0123456789abcdef01234567";
    public const string Sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    public static CatalogFile File(string path, long size = 1000) => new() { Path = path, Size = size, Sha256 = Sha };

    public static CatalogEntry Entry(string id = "tiny-model", params CatalogVariant[] variants) => new()
    {
        Id = id,
        Name = "Tiny model",
        Kind = "text-generation",
        Variants = variants.Length == 0 ? [Variant()] : variants,
        Verification = new CatalogVerification { Summary = "identical to the reference", Providers = ["CPU"], Date = "2026-11-16" },
    };

    public static CatalogVariant Variant(string id = "q8-0-gguf", string revision = Commit, params string[] companions) => new()
    {
        Id = id,
        Label = "Q8_0 GGUF",
        Format = CatalogFormats.Gguf,
        Repository = "owner/Model-GGUF",
        Revision = revision,
        Primary = "Model-Q8_0.gguf",
        Files = [File("Model-Q8_0.gguf")],
        Companions = companions,
    };

    public static VerifiedCatalog Catalog(params CatalogEntry[] entries) => new() { Updated = "2026-11-16", Entries = entries };
}
