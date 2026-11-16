using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Storage;

namespace Sharplight.Hub;

/// <summary>
/// A directory of pulled models: <c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;tag&gt;/</c> holds <c>model.slm</c> and
/// <c>manifest.json</c> (plus <c>source/</c> while a pull is running, or when the sources are kept). Models are named
/// <c>owner/name:tag</c> or by their short name <c>alias:tag</c> (see <see cref="ModelManifest"/>).
/// </summary>
public sealed partial class ModelStore
{
    /// <summary>Environment variable that overrides the default store location.</summary>
    public const string EnvironmentVariable = "SHARPLIGHT_MODELS";

    /// <summary>File name of the package in a model directory.</summary>
    public const string PackageFileName = "model.slm";

    /// <summary>File name of the manifest in a model directory.</summary>
    public const string ManifestFileName = "manifest.json";

    internal const string SourceDirectoryName = "source";
    internal const string PendingPackageFileName = "model.slm.pending";
    internal const string LockFileName = ".pull.lock";
    internal const string CommitFileName = ".commit";
    private const string LegacyDirectoryName = "models";
    private const string LegacyApplicationDirectoryName = "data";
    private const string DefaultTagDirectory = "default";
    private const string TemporarySuffix = ".tmp";
    private const char TagSeparator = ':';

    /// <summary>Opens (or, on first write, creates) the store at <paramref name="root"/>.</summary>
    /// <param name="root">The store directory.</param>
    public ModelStore(string root)
    {
        Root = Path.GetFullPath(root);
    }

    /// <summary>The store directory.</summary>
    public string Root { get; }

    /// <summary>
    /// The store directory to use (see <see cref="Resolve(string?)"/>): <paramref name="explicitRoot"/> when given, else
    /// <c>SHARPLIGHT_MODELS</c>, else the shared settings file, else the models folder of the storage layout.
    /// </summary>
    /// <param name="explicitRoot">A directory given on the command line, or <see langword="null"/>.</param>
    public static string ResolveRoot(string? explicitRoot)
    {
        return Resolve(explicitRoot).Root;
    }

    /// <summary>
    /// Resolves the store of a command-line tool with this process's storage layout (<see cref="StorageLayout.Resolve"/>): see
    /// <see cref="Resolve(string?, ModelStoreSettings?, StorageLayout)"/>, with the shared settings file of the layout
    /// (<see cref="ModelStoreSettings.FileFor"/>).
    /// </summary>
    /// <param name="commandLine">A directory given on the command line, or <see langword="null"/>.</param>
    public static ModelStoreResolution Resolve(string? commandLine)
    {
        var layout = StorageLayout.Resolve(StorageEnvironment.Process);
        return Resolve(commandLine, ModelStoreSettings.Load(ModelStoreSettings.FileFor(layout)), layout);
    }

    /// <summary>
    /// Resolves the store: <paramref name="commandLine"/> (<c>--models-dir</c>) when given, else <c>SHARPLIGHT_MODELS</c>, else
    /// <paramref name="settings"/>, else the models folder of <paramref name="layout"/> (<see cref="StorageLayout.ModelsDirectory"/>).
    /// </summary>
    /// <param name="commandLine">A directory given on the command line, or <see langword="null"/>.</param>
    /// <param name="settings">The store setting, or <see langword="null"/> when nothing was chosen.</param>
    /// <param name="layout">The storage layout in effect.</param>
    public static ModelStoreResolution Resolve(string? commandLine, ModelStoreSettings? settings, StorageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(layout);
        return Resolve(commandLine, Environment.GetEnvironmentVariable(EnvironmentVariable), settings, layout, SystemRoot);
    }

    /// <summary>The model store of the system folders (<see cref="StorageMode.System"/>), shared by every Sharplight tool of the user.</summary>
    public static string SystemRoot => StorageLayout.Create(StorageMode.System, null, StorageEnvironment.Process).ModelsDirectory;

    /// <summary>
    /// A store that still holds models while the store in effect (<paramref name="resolution"/>) has none: the system store, the
    /// portable store of this package, or where earlier versions kept them (the <c>data</c> or <c>models</c> folder next to the
    /// package). Used to offer moving them over.
    /// </summary>
    /// <param name="resolution">The store in effect.</param>
    /// <param name="layout">The storage layout in effect.</param>
    /// <returns>The store with models, or <see langword="null"/> when the store in effect has models or no other store has any.</returns>
    public static ModelStore? FindPreviousStore(ModelStoreResolution resolution, StorageLayout layout)
    {
        ArgumentNullException.ThrowIfNull(resolution);
        ArgumentNullException.ThrowIfNull(layout);
        return FindPreviousStore(resolution.Root, PreviousStoreCandidates(layout, SystemRoot));
    }

    /// <summary>Where <see cref="FindPreviousStore(ModelStoreResolution, StorageLayout)"/> looks, in order.</summary>
    internal static IReadOnlyList<string> PreviousStoreCandidates(StorageLayout layout, string systemRoot) =>
    [
        systemRoot,
        Path.Combine(layout.PortableRoot, LegacyDirectoryName),
        Path.Combine(layout.PackageDirectory, LegacyApplicationDirectoryName),
        Path.Combine(layout.PackageDirectory, LegacyDirectoryName),
    ];

    internal static ModelStoreResolution Resolve(string? commandLine, string? fromEnvironment, ModelStoreSettings? settings, StorageLayout layout,
        string systemRoot)
    {
        if (!string.IsNullOrWhiteSpace(commandLine))
        {
            return new ModelStoreResolution(commandLine, ModelStoreSource.CommandLine, ModelStoreLocation.Custom);
        }

        if (!string.IsNullOrWhiteSpace(fromEnvironment))
        {
            return new ModelStoreResolution(fromEnvironment, ModelStoreSource.EnvironmentVariable, ModelStoreLocation.Custom);
        }

        if (settings is not null)
        {
            switch (settings.Location)
            {
                case ModelStoreLocation.User:
                    return new ModelStoreResolution(systemRoot, ModelStoreSource.Settings, ModelStoreLocation.User);
                case ModelStoreLocation.Custom when !string.IsNullOrWhiteSpace(settings.CustomPath):
                    return new ModelStoreResolution(settings.CustomPath.Trim(), ModelStoreSource.Settings, ModelStoreLocation.Custom);
                case ModelStoreLocation.Application:
                    return new ModelStoreResolution(layout.ModelsDirectory, ModelStoreSource.Settings, ModelStoreLocation.Application, layout.Mode, layout.Notice);
                default:
                    break;
            }
        }

        return new ModelStoreResolution(layout.ModelsDirectory, ModelStoreSource.Default, ModelStoreLocation.Application, layout.Mode, layout.Notice);
    }

    internal static ModelStore? FindPreviousStore(string root, IEnumerable<string> candidates)
    {
        var current = new ModelStore(root);
        if (current.List().Count > 0)
        {
            return null;
        }

        var comparison = OperatingSystem.IsWindows() || OperatingSystem.IsMacOS() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
        foreach (var candidate in candidates)
        {
            var store = new ModelStore(candidate);
            if (!string.Equals(store.Root.TrimEnd(Path.DirectorySeparatorChar), current.Root.TrimEnd(Path.DirectorySeparatorChar), comparison)
                && store.List().Count > 0)
            {
                return store;
            }
        }

        return null;
    }

    /// <summary>Every stored model, ordered by name. Directories without a manifest (interrupted pulls) are skipped.</summary>
    public IReadOnlyList<ModelManifest> List()
    {
        if (!Directory.Exists(Root))
        {
            return [];
        }

        return [.. Directory.EnumerateFiles(Root, ManifestFileName, new EnumerationOptions { RecurseSubdirectories = true, MaxRecursionDepth = 3 })
            .Select(path => ReadManifest(Path.GetDirectoryName(path)!))
            .OfType<ModelManifest>()
            .OrderBy(manifest => manifest.Name, StringComparer.OrdinalIgnoreCase)];
    }

    /// <summary>
    /// Finds a stored model by <c>owner/name[:tag]</c> or <c>alias[:tag]</c> (case-insensitive). Without a tag, a model
    /// stored in a single variant is found.
    /// </summary>
    /// <param name="name">The name as typed.</param>
    /// <returns>The model, or <see langword="null"/> when none matches.</returns>
    /// <exception cref="HubException">Several stored models match (the message lists them).</exception>
    public ModelManifest? Find(string name)
    {
        var all = List();
        List<ModelManifest> matches;
        if (ModelReference.TryParse(name, out var reference))
        {
            matches = [.. all.Where(model => Same(model.Repository, reference.Repository) && TagMatches(model, reference.Tag))];
        }
        else
        {
            int separator = name.LastIndexOf(TagSeparator);
            var alias = separator >= 0 ? name[..separator] : name;
            var tag = separator >= 0 ? name[(separator + 1)..] : null;
            matches = [.. all.Where(model => Same(model.Alias, alias) && TagMatches(model, tag))];
        }

        return matches.Count switch
        {
            0 => null,
            1 => matches[0],
            _ => throw new HubException($"'{name}' matches several stored models: {string.Join(", ", matches.Select(model => model.Name))}. Add the tag."),
        };
    }

    /// <summary>Deletes a stored model: its manifest first (so it is never half registered), then its directory.</summary>
    /// <param name="manifest">A model from <see cref="List"/> or <see cref="Find"/>.</param>
    public void Remove(ModelManifest manifest)
    {
        var directory = manifest.Directory;
        if (!IsInside(directory))
        {
            throw new HubException($"'{directory}' is not inside the model store {Root}.");
        }

        File.Delete(Path.Combine(directory, ManifestFileName));
        Directory.Delete(directory, recursive: true);
        PruneEmptyParents(directory);
    }

    /// <summary>The directory of a model variant: <c>&lt;root&gt;/&lt;owner&gt;/&lt;name&gt;/&lt;tag&gt;</c>.</summary>
    /// <param name="repository">The repository id, <c>owner/name</c>.</param>
    /// <param name="tag">The variant tag; empty for a single-variant repository.</param>
    public string GetModelDirectory(string repository, string tag)
    {
        var parts = repository.Split('/');
        var tagDirectory = UnsafeFileNameCharacters().Replace(tag, "_");
        if (tagDirectory.Trim('.').Length == 0)
        {
            tagDirectory = DefaultTagDirectory;
        }

        return Path.Combine(Root, parts[0], parts[1], tagDirectory);
    }

    /// <summary>Reads the manifest in <paramref name="directory"/>, or null when there is none (or it is unreadable).</summary>
    internal static ModelManifest? ReadManifest(string directory)
    {
        var path = Path.Combine(directory, ManifestFileName);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            var manifest = JsonSerializer.Deserialize(File.ReadAllText(path), StoreJsonContext.Default.ModelManifest);
            if (manifest is not null)
            {
                manifest.Directory = directory;
            }

            return manifest;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>Writes the manifest into its <see cref="ModelManifest.Directory"/> atomically (temporary file, then rename).</summary>
    internal static void WriteManifest(ModelManifest manifest)
    {
        var path = Path.Combine(manifest.Directory, ManifestFileName);
        var temporary = path + TemporarySuffix;
        File.WriteAllText(temporary, JsonSerializer.Serialize(manifest, StoreJsonContext.Default.ModelManifest));
        File.Move(temporary, path, overwrite: true);
    }

    /// <summary>Takes the pull lock of a model directory; released when disposed (or when the process ends).</summary>
    /// <exception cref="HubException">Another pull of the same model is running.</exception>
    internal static IDisposable Lock(string directory, string name)
    {
        Directory.CreateDirectory(directory);
        try
        {
            return new FileStream(Path.Combine(directory, LockFileName), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None, 1, FileOptions.DeleteOnClose);
        }
        catch (IOException error)
        {
            throw new HubException($"Another pull of {name} is running.", inner: error);
        }
    }

    private static bool Same(string left, string right) => string.Equals(left, right, StringComparison.OrdinalIgnoreCase);

    private static bool TagMatches(ModelManifest model, string? tag)
    {
        if (tag is null || Same(model.Tag, tag))
        {
            return true;
        }

        if (model.Tag.Length == 0 && Same(tag, ModelFileSelector.DefaultTag))
        {
            return true;
        }

        // A file name pulled by name (Qwen3-0.6B-Q8_0.gguf) is stored under its quantization tag.
        return model.SourceFiles.Any(file => Same(RepoPath.Name(file.Path), tag) || Same(Path.GetFileNameWithoutExtension(file.Path), tag));
    }

    private bool IsInside(string directory)
    {
        var full = Path.GetFullPath(directory);
        return full.StartsWith(Root + Path.DirectorySeparatorChar, StringComparison.Ordinal);
    }

    internal void PruneEmptyParents(string directory)
    {
        var parent = Path.GetDirectoryName(directory);
        while (parent is not null && IsInside(parent) && Directory.Exists(parent) && !Directory.EnumerateFileSystemEntries(parent).Any())
        {
            Directory.Delete(parent);
            parent = Path.GetDirectoryName(parent);
        }
    }

    [GeneratedRegex(@"[^A-Za-z0-9._-]")]
    private static partial Regex UnsafeFileNameCharacters();
}
