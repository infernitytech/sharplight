using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// All shards of a safetensors checkpoint. Shards are discovered via <c>model.safetensors.index.json</c>
/// (<c>weight_map</c>) or, without an index, as all <c>*.safetensors</c> files in the directory.
/// </summary>
internal sealed class SafetensorsCheckpoint : IDisposable
{
    private readonly List<SafetensorsFile> _files = [];
    private readonly Dictionary<string, SafetensorsTensor> _tensors = new(StringComparer.Ordinal);

    /// <summary>The tensors of all shards by name.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensor> Tensors => _tensors;

    private SafetensorsCheckpoint()
    {
    }

    /// <summary>Opens every shard of the checkpoint in <paramref name="directory"/>.</summary>
    /// <exception cref="FileNotFoundException">No shard exists, or the index lists a missing shard.</exception>
    /// <exception cref="InvalidDataException">A tensor name appears in more than one shard.</exception>
    public static SafetensorsCheckpoint Open(string directory)
    {
        var checkpoint = new SafetensorsCheckpoint();
        try
        {
            foreach (var shard in DiscoverShards(directory))
            {
                var file = SafetensorsFile.Open(shard);
                checkpoint._files.Add(file);
                foreach (var (name, tensor) in file.Tensors)
                {
                    if (!checkpoint._tensors.TryAdd(name, tensor))
                    {
                        throw new InvalidDataException($"Tensor '{name}' appears in more than one shard ({checkpoint._tensors[name].File.Path}, {shard}).");
                    }
                }
            }

            if (checkpoint._files.Count == 0)
            {
                throw new FileNotFoundException($"No *.safetensors files in '{directory}'.");
            }

            return checkpoint;
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    private static IEnumerable<string> DiscoverShards(string dir)
    {
        var all = Directory.EnumerateFiles(dir, "*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToList();
        string? index = Directory.EnumerateFiles(dir, "*.safetensors.index.json").OrderBy(p => p, StringComparer.Ordinal).FirstOrDefault();
        if (index == null)
        {
            return all;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(index));
        var names = new SortedSet<string>(StringComparer.Ordinal);
        if (doc.RootElement.TryGetProperty("weight_map", out var weightMap))
        {
            foreach (var entry in weightMap.EnumerateObject())
            {
                if (entry.Value.GetString() is { } shardName)
                {
                    names.Add(shardName);
                }
            }
        }

        if (names.Count == 0)
        {
            return all;
        }

        return names.Select(shardName =>
        {
            var full = Path.Combine(dir, shardName);
            return File.Exists(full) ? full : throw new FileNotFoundException($"Shard '{shardName}' listed in {Path.GetFileName(index)} is missing.", full);
        });
    }

    /// <summary>Unmaps every shard.</summary>
    public void Dispose()
    {
        foreach (var file in _files)
        {
            file.Dispose();
        }

        _files.Clear();
    }
}
