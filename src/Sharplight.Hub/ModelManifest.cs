using System.Text.Json.Serialization;

namespace Sharplight.Hub;

/// <summary>
/// The record of a model in a <see cref="ModelStore"/> (<c>manifest.json</c> next to the package): where it came from (repository,
/// pinned commit, source files), what was built (package, precompiled artifact sets) and when. A model counts as stored only
/// once its manifest exists, which is written last.
/// </summary>
public sealed class ModelManifest
{
    /// <summary>Version of the manifest layout written by this library.</summary>
    public const int CurrentSchema = 1;

    /// <summary>Layout version of this manifest.</summary>
    public int Schema { get; init; } = CurrentSchema;

    /// <summary>The repository id, <c>owner/name</c>.</summary>
    public required string Repository { get; init; }

    /// <summary>The variant tag (e.g. <c>Q4_K_M</c>); empty for a repository with a single variant.</summary>
    public string Tag { get; init; } = "";

    /// <summary>Short name: the repository name in lower case without a format suffix (<c>qwen3-0.6b</c> for <c>Qwen/Qwen3-0.6B-GGUF</c>).</summary>
    public required string Alias { get; init; }

    /// <summary>The revision that was pulled (branch, tag or commit).</summary>
    public string Revision { get; init; } = "main";

    /// <summary>The commit the revision resolved to; the source files are from this commit.</summary>
    public required string Commit { get; init; }

    /// <summary>The source format.</summary>
    public ModelFormat Format { get; init; }

    /// <summary>Name of the model as built (from its metadata).</summary>
    public string? ModelName { get; init; }

    /// <summary>Architecture of the built model (e.g. <c>qwen3</c>).</summary>
    public string? Architecture { get; init; }

    /// <summary>File name of the package in the model directory.</summary>
    public string Package { get; init; } = ModelStore.PackageFileName;

    /// <summary>Size of the package in bytes.</summary>
    public long PackageBytes { get; init; }

    /// <summary>The files the package was converted from.</summary>
    public IReadOnlyList<ManifestFile> SourceFiles { get; init; } = [];

    /// <summary>Whether the source files were kept (in <c>source/</c> next to the package).</summary>
    public bool SourceKept { get; init; }

    /// <summary>Descriptors of the precompiled artifact sets stored in the package.</summary>
    public IReadOnlyList<string> Precompiled { get; init; } = [];

    /// <summary>When the model was pulled.</summary>
    public DateTimeOffset PulledAt { get; init; }

    /// <summary>The model's directory in the store (set when the manifest is read; not stored).</summary>
    [JsonIgnore]
    public string Directory { get; set; } = "";

    /// <summary><c>owner/name:tag</c>, or <c>owner/name</c> for a single-variant repository.</summary>
    [JsonIgnore]
    public string Name => Tag.Length == 0 ? Repository : $"{Repository}:{Tag}";

    /// <summary><c>alias:tag</c> in lower case, or the alias for a single-variant repository.</summary>
    [JsonIgnore]
    public string ShortName => Tag.Length == 0 ? Alias : $"{Alias}:{Tag.ToLowerInvariant()}";

    /// <summary>Full path of the package.</summary>
    [JsonIgnore]
    public string PackagePath => Path.Combine(Directory, Package);

    /// <summary>Total size of the source files.</summary>
    [JsonIgnore]
    public long SourceBytes => SourceFiles.Sum(file => file.Size);

    /// <summary>The alias for a repository: its name in lower case without a <c>-GGUF</c> / <c>-ONNX</c> suffix.</summary>
    /// <param name="repository">The repository id, <c>owner/name</c>.</param>
    public static string AliasFor(string repository)
    {
        var name = RepoPath.Name(repository).ToLowerInvariant();
        foreach (var suffix in new[] { "-gguf", "_gguf", ".gguf", "-onnx", "_onnx" })
        {
            if (name.EndsWith(suffix, StringComparison.Ordinal) && name.Length > suffix.Length)
            {
                return name[..^suffix.Length];
            }
        }

        return name;
    }
}
