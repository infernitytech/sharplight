namespace Sharplight.Hub;

/// <summary>
/// Moves or copies the models of one store into another, keeping their <c>owner/name/tag</c> layout. Only complete models (with a
/// manifest) are transferred; a model the destination already holds is skipped. The manifest is written last, so an interrupted
/// transfer never leaves a half-registered model in the destination.
/// </summary>
public static class ModelStoreTransfer
{
    /// <summary>Moves or copies every model of <paramref name="source"/> into <paramref name="destination"/>.</summary>
    /// <param name="source">The store that has the models.</param>
    /// <param name="destination">The store to move or copy them to.</param>
    /// <param name="move">Move (rename on the same volume, else copy and delete) instead of copying.</param>
    /// <param name="log">Receives one line per model, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Stops between files; a model copied halfway has no manifest in the destination.</param>
    /// <returns>The number of models transferred.</returns>
    public static int Transfer(ModelStore source, ModelStore destination, bool move, Action<string>? log = null, CancellationToken cancellationToken = default)
    {
        int count = 0;
        foreach (var manifest in source.List())
        {
            cancellationToken.ThrowIfCancellationRequested();
            var target = Path.Combine(destination.Root, Path.GetRelativePath(source.Root, manifest.Directory));
            if (File.Exists(Path.Combine(target, ModelStore.ManifestFileName)))
            {
                log?.Invoke($"{manifest.Name}: already in {destination.Root}, skipped");
                continue;
            }

            if (move && TryRename(manifest.Directory, target))
            {
                source.PruneEmptyParents(manifest.Directory);
            }
            else
            {
                CopyModel(manifest.Directory, target, cancellationToken);
                if (move)
                {
                    source.Remove(manifest);
                }
            }

            log?.Invoke($"{manifest.Name}: {(move ? "moved" : "copied")}");
            count++;
        }

        return count;
    }

    private static bool TryRename(string from, string to)
    {
        if (Directory.Exists(to))
        {
            return false;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(to)!);
            Directory.Move(from, to);
            return true;
        }
        catch (IOException)
        {
            // Another volume: copy instead.
            return false;
        }
    }

    private static void CopyModel(string from, string to, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(to);
        foreach (var file in Directory.EnumerateFiles(from, "*", SearchOption.AllDirectories))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var name = Path.GetFileName(file);
            if (name is ModelStore.ManifestFileName or ModelStore.LockFileName)
            {
                continue;
            }

            var target = Path.Combine(to, Path.GetRelativePath(from, file));
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.Copy(file, target, overwrite: true);
        }

        var manifestTarget = Path.Combine(to, ModelStore.ManifestFileName);
        File.Copy(Path.Combine(from, ModelStore.ManifestFileName), manifestTarget + ".tmp", overwrite: true);
        File.Move(manifestTarget + ".tmp", manifestTarget, overwrite: true);
    }
}
