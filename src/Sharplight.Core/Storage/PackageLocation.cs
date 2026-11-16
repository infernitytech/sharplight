namespace Sharplight.Storage;

/// <summary>
/// Where the package of the running application is, and which folders belong to its installation. The package is what the user
/// sees and moves: the AppImage file (never the image mounted at <c>APPDIR</c>, which is replaced on every update), the Velopack
/// install root (never its <c>current</c> folder, which updates replace), the <c>.app</c> bundle (never its inside, which would
/// break the signature), or else the executable's folder (a tar.gz, a zip or a build output).
/// </summary>
internal static class PackageLocation
{
    internal const string AppImageVariable = "APPIMAGE";
    internal const string AppImageDirectoryVariable = "APPDIR";
    private const string VelopackCurrentDirectory = "current";
    private static readonly string[] s_velopackUpdaters = ["Update.exe", "UpdateNix", "UpdateMac"];

    /// <summary>The folder that contains the package; a portable <c>data</c> folder goes next to the package, inside this folder.</summary>
    public static string FindPackageDirectory(StorageEnvironment environment)
    {
        if (environment.Variable(AppImageVariable) is { } appImage)
        {
            return Path.GetDirectoryName(Path.GetFullPath(appImage))!;
        }

        var application = Normalize(environment.ApplicationDirectory);
        if (FindBundle(application) is { } bundle)
        {
            return Path.GetDirectoryName(bundle)!;
        }

        return FindVelopackRoot(application) ?? application;
    }

    /// <summary>
    /// True for folders that belong to an installation rather than to the user, even when the process could write to them: the
    /// AppImage's <c>APPDIR</c>, a Velopack <c>current</c> folder, the inside of a <c>.app</c> bundle, <c>/usr</c> and Program Files.
    /// </summary>
    public static bool IsInstallLocation(string path, StorageEnvironment environment)
    {
        var fullPath = Normalize(path);
        var comparison = Comparison(environment.Platform);
        if (environment.Variable(AppImageDirectoryVariable) is { } appDirectory && IsInside(fullPath, Normalize(appDirectory), comparison))
        {
            return true;
        }

        var application = Normalize(environment.ApplicationDirectory);
        if (FindVelopackRoot(application) is not null && IsInside(fullPath, application, comparison))
        {
            return true;
        }

        if (FindBundle(fullPath) is { } bundle && !string.Equals(fullPath, bundle, comparison))
        {
            return true;
        }

        return environment.Platform switch
        {
            StoragePlatform.Windows => IsInsideProgramFiles(fullPath, environment),
            _ => fullPath.StartsWith("/usr/", StringComparison.Ordinal),
        };
    }

    /// <summary>True when <paramref name="path"/> is <paramref name="root"/> or inside it.</summary>
    public static bool IsInside(string path, string root, StringComparison comparison)
    {
        var trimmedPath = path.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var trimmedRoot = root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        return string.Equals(trimmedPath, trimmedRoot, comparison)
            || trimmedPath.StartsWith(trimmedRoot + Path.DirectorySeparatorChar, comparison);
    }

    /// <summary>How paths compare on <paramref name="platform"/>.</summary>
    public static StringComparison Comparison(StoragePlatform platform) =>
        platform is StoragePlatform.Windows or StoragePlatform.MacOS ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>The full path without a trailing separator.</summary>
    public static string Normalize(string path)
    {
        var full = Path.GetFullPath(path);
        var root = Path.GetPathRoot(full) ?? "";
        return full.Length > root.Length ? full.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) : full;
    }

    /// <summary>The <c>.app</c> bundle <paramref name="path"/> is in (or is), or null.</summary>
    private static string? FindBundle(string path)
    {
        var directory = new DirectoryInfo(path);
        while (directory is not null)
        {
            if (directory.Name.EndsWith(".app", StringComparison.OrdinalIgnoreCase))
            {
                return directory.FullName.TrimEnd(Path.DirectorySeparatorChar);
            }

            directory = directory.Parent;
        }

        return null;
    }

    /// <summary>The install root when <paramref name="application"/> is a Velopack <c>current</c> folder (an updater next to it), else null.</summary>
    private static string? FindVelopackRoot(string application)
    {
        var directory = new DirectoryInfo(application);
        if (!string.Equals(directory.Name, VelopackCurrentDirectory, StringComparison.OrdinalIgnoreCase) || directory.Parent is null)
        {
            return null;
        }

        var root = directory.Parent.FullName.TrimEnd(Path.DirectorySeparatorChar);
        return s_velopackUpdaters.Any(updater => File.Exists(Path.Combine(root, updater))) ? root : null;
    }

    private static bool IsInsideProgramFiles(string fullPath, StorageEnvironment environment)
    {
        foreach (var variable in new[] { "ProgramFiles", "ProgramFiles(x86)", "ProgramW6432" })
        {
            if (environment.Variable(variable) is { } programFiles && IsInside(fullPath, Normalize(programFiles), StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }
}
