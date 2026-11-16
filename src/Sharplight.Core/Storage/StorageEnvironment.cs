namespace Sharplight.Storage;

/// <summary>
/// The inputs of <see cref="StorageLayout.Resolve"/>: the application folder, the platform, the home folder and the environment
/// variables. <see cref="Process"/> describes the running process; tests build their own. This is the only place that asks the
/// operating system for well-known folders: everything else gets its paths from a <see cref="StorageLayout"/>.
/// </summary>
public sealed class StorageEnvironment
{
    /// <summary>The folder of the running executable and its assemblies (<see cref="AppContext.BaseDirectory"/> for a process).</summary>
    public required string ApplicationDirectory { get; init; }

    /// <summary>The platform whose conventions apply.</summary>
    public required StoragePlatform Platform { get; init; }

    /// <summary>The user's home folder.</summary>
    public required string HomeDirectory { get; init; }

    /// <summary>Reads an environment variable; null when it is not set. Default: none is set.</summary>
    public Func<string, string?> GetVariable { get; init; } = _ => null;

    /// <summary>
    /// True when files can be written into a directory, creating it (see <see cref="StorageWriteProbe.CanWrite"/>). Replaced by tests
    /// to simulate read-only locations.
    /// </summary>
    public Func<string, bool> CanWrite { get; init; } = StorageWriteProbe.CanWrite;

    /// <summary>The environment of the running process.</summary>
    public static StorageEnvironment Process => new()
    {
        ApplicationDirectory = AppContext.BaseDirectory,
        Platform = CurrentPlatform,
        HomeDirectory = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile, Environment.SpecialFolderOption.DoNotVerify),
        GetVariable = Environment.GetEnvironmentVariable,
    };

    /// <summary>The platform of the running process.</summary>
    public static StoragePlatform CurrentPlatform =>
        OperatingSystem.IsWindows() ? StoragePlatform.Windows : OperatingSystem.IsMacOS() ? StoragePlatform.MacOS : StoragePlatform.Linux;

    /// <summary>The value of <paramref name="name"/>, or null when it is unset or blank.</summary>
    internal string? Variable(string name)
    {
        var value = GetVariable(name);
        return string.IsNullOrWhiteSpace(value) ? null : value.Trim();
    }
}
