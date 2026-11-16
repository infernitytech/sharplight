namespace Sharplight.Storage;

/// <summary>
/// Where Sharplight keeps every file it writes: configuration, application data, the model store, caches (compiled kernels, shader
/// sources) and logs. Three modes (<see cref="StorageMode"/>): <b>portable</b> keeps everything in a <c>data</c> folder next to the
/// package, <b>system</b> follows the platform's per-user folders, <b>custom</b> uses one folder the user chose. Studio, the
/// command-line tools and the engine's artifact store resolve the same layout (<see cref="Resolve"/>); hosts with a service container
/// register it once (<c>services.AddSharplightStorage()</c>) and take it from there.
/// </summary>
/// <remarks>
/// Resolution order: <c>SHARPLIGHT_HOME</c> (a custom root for everything), then the portable marker file
/// (<see cref="PortableMarkerFileName"/>) next to the package, then the pointer file in the system configuration folder
/// (<see cref="StoragePointer"/>), else the system folders. A portable or custom root that cannot be written falls back to the system
/// folders and says so (<see cref="Notice"/>). Narrower overrides apply on top: <c>SHARPLIGHT_MODELS</c> / <c>--models-dir</c> for the
/// model store, <c>SHARPLIGHT_COMPILATION_CACHE</c> for the artifact store.
/// </remarks>
public sealed class StorageLayout
{
    /// <summary>Environment variable that makes one folder the custom root of everything.</summary>
    public const string HomeVariable = "SHARPLIGHT_HOME";

    /// <summary>File next to the package that makes it portable (empty, or a JSON object).</summary>
    public const string PortableMarkerFileName = "sharplight.portable";

    /// <summary>Name of the portable root folder next to the package.</summary>
    public const string PortableDirectoryName = "data";

    internal const string ProductDirectoryName = "sharplight";
    internal const string ConfigDirectoryName = "config";
    internal const string CacheDirectoryName = "cache";
    internal const string LogsDirectoryName = "logs";
    private const string ModelsDirectoryName = "models";

    private StorageLayout(StorageMode mode, StorageModeSource source, string? root, StorageEnvironment environment, string? notice)
    {
        Mode = mode;
        Source = source;
        Root = root is null ? null : PackageLocation.Normalize(root);
        Platform = environment.Platform;
        PackageDirectory = PackageLocation.FindPackageDirectory(environment);
        var system = SystemStorageFolders.For(environment);
        PointerFile = Path.Combine(system.Config, StoragePointer.FileName);
        Notice = notice;
        if (Root is null)
        {
            (ConfigDirectory, DataDirectory, CacheDirectory, LogsDirectory) = system;
        }
        else
        {
            ConfigDirectory = Path.Combine(Root, ConfigDirectoryName);
            DataDirectory = Root;
            CacheDirectory = Path.Combine(Root, CacheDirectoryName);
            LogsDirectory = Path.Combine(Root, LogsDirectoryName);
        }
    }

    /// <summary>The mode in effect (after a fallback, <see cref="StorageMode.System"/>).</summary>
    public StorageMode Mode { get; }

    /// <summary>What selected <see cref="Mode"/>.</summary>
    public StorageModeSource Source { get; }

    /// <summary>The root folder of a portable or custom layout; null for the system folders.</summary>
    public string? Root { get; }

    /// <summary>The platform whose conventions the layout follows.</summary>
    public StoragePlatform Platform { get; }

    /// <summary>The folder that contains the package (where the portable marker and the portable <c>data</c> folder go).</summary>
    public string PackageDirectory { get; }

    /// <summary>The portable marker file of this package.</summary>
    public string PortableMarkerFile => Path.Combine(PackageDirectory, PortableMarkerFileName);

    /// <summary>The portable root of this package: the <c>data</c> folder next to it.</summary>
    public string PortableRoot => Path.Combine(PackageDirectory, PortableDirectoryName);

    /// <summary>The pointer file in the system configuration folder (<see cref="StoragePointer"/>).</summary>
    public string PointerFile { get; }

    /// <summary>Why the layout differs from what was asked for (a portable or custom root that is not writable), or null.</summary>
    public string? Notice { get; }

    /// <summary>Settings and other configuration files.</summary>
    public string ConfigDirectory { get; }

    /// <summary>Application data (databases, history, outputs); the model store is its <c>models</c> subfolder.</summary>
    public string DataDirectory { get; }

    /// <summary>Caches that can be rebuilt (compiled kernels, shader sources).</summary>
    public string CacheDirectory { get; }

    /// <summary>Log files.</summary>
    public string LogsDirectory { get; }

    /// <summary>The model store (<c>models</c> in <see cref="DataDirectory"/>).</summary>
    public string ModelsDirectory => Path.Combine(DataDirectory, ModelsDirectoryName);

    /// <summary>The per-user artifact store of compiled kernels (<c>artifacts</c> in <see cref="CacheDirectory"/>).</summary>
    public string ArtifactsDirectory => Path.Combine(CacheDirectory, "artifacts");

    /// <summary>Extracted kernel and shader sources (<c>shaders</c> in <see cref="CacheDirectory"/>).</summary>
    public string ShaderSourcesDirectory => Path.Combine(CacheDirectory, "shaders");

    /// <summary>The configuration folder of one application (<c>studio</c>, ...).</summary>
    /// <param name="application">A folder name.</param>
    public string GetConfigDirectory(string application) => Path.Combine(ConfigDirectory, application);

    /// <summary>The data folder of one application (<c>studio</c>, ...).</summary>
    /// <param name="application">A folder name.</param>
    public string GetDataDirectory(string application) => Path.Combine(DataDirectory, application);

    /// <summary>The folder of <paramref name="category"/>.</summary>
    /// <param name="category">The kind of files.</param>
    public string GetDirectory(StorageCategory category) => category switch
    {
        StorageCategory.Config => ConfigDirectory,
        StorageCategory.Data => DataDirectory,
        StorageCategory.Models => ModelsDirectory,
        StorageCategory.Cache => CacheDirectory,
        StorageCategory.Logs => LogsDirectory,
        _ => throw new ArgumentOutOfRangeException(nameof(category), category, null),
    };

    /// <summary>A short description of the layout and why it is in effect, for settings pages and command output.</summary>
    public string Description
    {
        get
        {
            var what = Mode switch
            {
                StorageMode.Portable => "Portable: next to the application",
                StorageMode.Custom => "Custom folder",
                _ => "System folders",
            };
            var why = Source switch
            {
                StorageModeSource.EnvironmentVariable => $" (set by {HomeVariable})",
                StorageModeSource.PortableMarker => $" ({PortableMarkerFileName})",
                _ => "",
            };
            return Notice is null ? what + why : $"{what}{why}; {Notice}";
        }
    }

    /// <summary>Resolves the layout of <paramref name="environment"/> (see the remarks of <see cref="StorageLayout"/>).</summary>
    /// <param name="environment">The process environment, or one built by a test.</param>
    public static StorageLayout Resolve(StorageEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        if (environment.Variable(HomeVariable) is { } home)
        {
            return new StorageLayout(StorageMode.Custom, StorageModeSource.EnvironmentVariable, home, environment, null);
        }

        string? notice = null;
        var package = PackageLocation.FindPackageDirectory(environment);
        if (File.Exists(Path.Combine(package, PortableMarkerFileName)))
        {
            var root = Path.Combine(package, PortableDirectoryName);
            if (IsUsable(root, environment))
            {
                return new StorageLayout(StorageMode.Portable, StorageModeSource.PortableMarker, root, environment, null);
            }

            notice = $"the portable folder {root} is not writable, so the system folders are used";
        }

        var pointer = StoragePointer.Load(Path.Combine(SystemStorageFolders.For(environment).Config, StoragePointer.FileName));
        if (notice is null && pointer is { Mode: StorageMode.Custom, Root: { } custom } && !string.IsNullOrWhiteSpace(custom))
        {
            if (IsUsable(custom.Trim(), environment))
            {
                return new StorageLayout(StorageMode.Custom, StorageModeSource.PointerFile, custom.Trim(), environment, null);
            }

            notice = $"the custom folder {custom.Trim()} is not available, so the system folders are used";
        }

        var source = pointer is null ? StorageModeSource.Default : StorageModeSource.PointerFile;
        return new StorageLayout(StorageMode.System, notice is null ? source : StorageModeSource.Default, null, environment, notice);
    }

    /// <summary>
    /// The layout <paramref name="mode"/> would give in <paramref name="environment"/>, without checking that its folders can be
    /// written (to preview a change, or as the target of a <see cref="StorageMigration"/>).
    /// </summary>
    /// <param name="mode">The mode.</param>
    /// <param name="customRoot">The root of <see cref="StorageMode.Custom"/>; ignored otherwise.</param>
    /// <param name="environment">The environment.</param>
    /// <exception cref="ArgumentException">A custom layout without a root.</exception>
    public static StorageLayout Create(StorageMode mode, string? customRoot, StorageEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        var root = mode switch
        {
            StorageMode.Portable => Path.Combine(PackageLocation.FindPackageDirectory(environment), PortableDirectoryName),
            StorageMode.Custom when !string.IsNullOrWhiteSpace(customRoot) => customRoot.Trim(),
            StorageMode.Custom => throw new ArgumentException("A custom storage layout needs a root folder.", nameof(customRoot)),
            _ => null,
        };
        return new StorageLayout(mode, StorageModeSource.Explicit, root, environment, null);
    }

    /// <summary>True when the portable root of <paramref name="environment"/>'s package can be used (not an install location, writable).</summary>
    /// <param name="environment">The environment.</param>
    public static bool CanUsePortable(StorageEnvironment environment)
    {
        ArgumentNullException.ThrowIfNull(environment);
        return IsUsable(Path.Combine(PackageLocation.FindPackageDirectory(environment), PortableDirectoryName), environment);
    }

    /// <summary>True when <paramref name="path"/> is this layout's folder of some category or inside one.</summary>
    /// <param name="path">A path.</param>
    public bool Contains(string path)
    {
        var comparison = PackageLocation.Comparison(Platform);
        var full = PackageLocation.Normalize(path);
        return Enum.GetValues<StorageCategory>().Any(category => PackageLocation.IsInside(full, PackageLocation.Normalize(GetDirectory(category)), comparison));
    }

    /// <summary>True when this layout puts every category in the same folders as <paramref name="other"/>.</summary>
    /// <param name="other">Another layout.</param>
    public bool HasSameFolders(StorageLayout other)
    {
        ArgumentNullException.ThrowIfNull(other);
        var comparison = PackageLocation.Comparison(Platform);
        return Enum.GetValues<StorageCategory>().All(category =>
            string.Equals(PackageLocation.Normalize(GetDirectory(category)), PackageLocation.Normalize(other.GetDirectory(category)), comparison));
    }

    private static bool IsUsable(string root, StorageEnvironment environment)
    {
        return !PackageLocation.IsInstallLocation(root, environment) && environment.CanWrite(root);
    }
}
