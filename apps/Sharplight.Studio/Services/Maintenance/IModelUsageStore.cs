namespace Sharplight.Studio.Services.Maintenance;

/// <summary>When each library model was last loaded (by path), kept in a small JSON file next to the other app data.</summary>
internal interface IModelUsageStore
{
    /// <summary>When the model at <paramref name="path"/> was last loaded, or null when it never was since tracking began.</summary>
    public DateTimeOffset? LastUsed(string path);

    /// <summary>Records that the model at <paramref name="path"/> was loaded at <paramref name="when"/>.</summary>
    public void Record(string path, DateTimeOffset when);

    /// <summary>Forgets the model at <paramref name="path"/> (it was deleted).</summary>
    public void Forget(string path);
}
