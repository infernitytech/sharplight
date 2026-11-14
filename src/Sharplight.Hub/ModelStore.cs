using System.Text.Json;
using System.Text.RegularExpressions;

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
    /// <summary>Name of the store folder in the application directory (see <see cref="ApplicationRoot"/>).</summary>
    public const string ApplicationDirectoryName = "data";

    private const string DefaultDirectoryName = "models";
    private const string UserDirectoryName = "sharplight";
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
    /// <c>SHARPLIGHT_MODELS</c>, else the shared settings file, else the <c>data</c> folder in the application directory.
    /// </summary>
    /// <param name="explicitRoot">A directory given on the command line, or <see langword="null"/>.</param>
    public static string ResolveRoot(string? explicitRoot)
    {
        return Resolve(explicitRoot).Root;
    }

    /// <summary>
    /// Resolves the store of a command-line tool: <paramref name="commandLine"/> (<c>--models-dir</c>) when given, else
    /// <c>SHARPLIGHT_MODELS</c>, else the shared settings file (<see cref="ModelStoreSettings.DefaultFile"/>), else the
    /// application folder (<see cref="ApplicationRoot"/>). The application folder falls back to the per-user store
    /// (<see cref="UserRoot"/>) when it is not writable.
    /// </summary>
    /// <param name="commandLine">A directory given on the command line, or <see langword="null"/>.</param>
    public static ModelStoreResolution Resolve(string? commandLine)
    {
        return Resolve(commandLine, ModelStoreSettings.Load(ModelStoreSettings.DefaultFile));
    }

    /// <summary>
    /// Resolves the store of an application with its own settings: <paramref name="commandLine"/> when given, else
    /// <c>SHARPLIGHT_MODELS</c>, else <paramref name="settings"/>, else the application folder (<see cref="ApplicationRoot"/>).
    /// The application folder falls back to the per-user store (<see cref="UserRoot"/>) when it is not writable.
    /// </summary>
    /// <param name="commandLine">A directory given on the command line, or <see langword="null"/>.</param>
    /// <param name="settings">The application's store setting, or <see langword="null"/> when nothing was chosen.</param>
    public static ModelStoreResolution Resolve(string? commandLine, ModelStoreSettings? settings)
    {
        return Resolve(commandLine, Environment.GetEnvironmentVariable(EnvironmentVariable), settings, AppContext.BaseDirectory, UserRoot, CanWrite);
    }

    /// <summary>
    /// The default store: the <c>data</c> folder in the application directory (<see cref="AppContext.BaseDirectory"/>), so an
    /// unpacked application keeps its models next to itself.
    /// </summary>
    public static string ApplicationRoot => Path.Combine(AppContext.BaseDirectory, ApplicationDirectoryName);

    /// <summary>
    /// The per-user store, shared by every Sharplight executable of the user: <c>sharplight/models</c> in the local application
    /// data directory (<c>~/.local/share</c> or <c>XDG_DATA_HOME</c> on Linux, <c>%LOCALAPPDATA%</c> on Windows,
    /// <c>~/Library/Application Support</c> on macOS).
    /// </summary>
    public static string UserRoot => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData, Environment.SpecialFolderOption.DoNotVerify),
        UserDirectoryName, DefaultDirectoryName);

    /// <summary>
    /// A store that still holds models while the store in effect (<paramref name="resolution"/>) has none: the per-user store or a
    /// <c>models</c> folder next to the executable (where earlier versions kept them). Used to offer moving them over.
    /// </summary>
    /// <param name="resolution">The store in effect.</param>
    /// <returns>The store with models, or <see langword="null"/> when the store in effect has models or no other store has any.</returns>
    public static ModelStore? FindPreviousStore(ModelStoreResolution resolution)
    {
        return FindPreviousStore(resolution.Root, [UserRoot, Path.Combine(AppContext.BaseDirectory, DefaultDirectoryName)]);
    }

    internal static ModelStoreResolution Resolve(string? commandLine, string? fromEnvironment, ModelStoreSettings? settings, string applicationDirectory,
        string userRoot, Func<string, bool> canWrite)
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
                    return new ModelStoreResolution(userRoot, ModelStoreSource.Settings, ModelStoreLocation.User);
                case ModelStoreLocation.Custom when !string.IsNullOrWhiteSpace(settings.CustomPath):
                    return new ModelStoreResolution(settings.CustomPath.Trim(), ModelStoreSource.Settings, ModelStoreLocation.Custom);
                case ModelStoreLocation.Application:
                    return ResolveApplication(ModelStoreSource.Settings, applicationDirectory, userRoot, canWrite);
                default:
                    break;
            }
        }

        return ResolveApplication(ModelStoreSource.Default, applicationDirectory, userRoot, canWrite);
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

    /// <summary>
    /// True when models can be stored in <paramref name="directory"/>: it is not inside a system-wide install (<c>/usr</c>,
    /// Program Files) or a macOS app bundle, it can be created, and a file can be written into it. Creates the directory.
    /// </summary>
    internal static bool CanWrite(string directory)
    {
        if (IsInstallLocation(Path.GetFullPath(directory)))
        {
            return false;
        }

        try
        {
            Directory.CreateDirectory(directory);
            var probe = Path.Combine(directory, $".write-probe-{Environment.ProcessId}-{Guid.NewGuid():N}");
            using (new FileStream(probe, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1, FileOptions.DeleteOnClose))
            {
            }

            return true;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// True for directories that belong to an installation rather than to the user, even when the current process could write to
    /// them: <c>/usr</c> on Unix, Program Files on Windows, and the inside of a macOS <c>.app</c> bundle (writing there breaks its
    /// signature).
    /// </summary>
    internal static bool IsInstallLocation(string fullPath)
    {
        if (OperatingSystem.IsWindows())
        {
            foreach (var folder in new[] { Environment.SpecialFolder.ProgramFiles, Environment.SpecialFolder.ProgramFilesX86 })
            {
                var programFiles = Environment.GetFolderPath(folder, Environment.SpecialFolderOption.DoNotVerify);
                if (programFiles.Length > 0 && fullPath.StartsWith(programFiles.TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }

            return false;
        }

        if (OperatingSystem.IsMacOS() && fullPath.Contains(".app/Contents/", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        return fullPath.StartsWith("/usr/", StringComparison.Ordinal);
    }

    private static ModelStoreResolution ResolveApplication(ModelStoreSource source, string applicationDirectory, string userRoot, Func<string, bool> canWrite)
    {
        var root = Path.Combine(applicationDirectory, ApplicationDirectoryName);
        return canWrite(root)
            ? new ModelStoreResolution(root, source, ModelStoreLocation.Application)
            : new ModelStoreResolution(userRoot, source, ModelStoreLocation.User, ApplicationFolderNotWritable: true);
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
