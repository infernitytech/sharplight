namespace Sharplight.Storage;

/// <summary>The kinds of files Sharplight writes, each with its own directory in a <see cref="StorageLayout"/>.</summary>
public enum StorageCategory
{
    /// <summary>Settings and other small configuration files (<see cref="StorageLayout.ConfigDirectory"/>).</summary>
    Config,

    /// <summary>Application data: databases, history, outputs (<see cref="StorageLayout.DataDirectory"/>, without the models).</summary>
    Data,

    /// <summary>The model store (<see cref="StorageLayout.ModelsDirectory"/>).</summary>
    Models,

    /// <summary>Caches that can be rebuilt: compiled kernels, shader sources, Hub listings (<see cref="StorageLayout.CacheDirectory"/>).</summary>
    Cache,

    /// <summary>Log files (<see cref="StorageLayout.LogsDirectory"/>).</summary>
    Logs,
}
