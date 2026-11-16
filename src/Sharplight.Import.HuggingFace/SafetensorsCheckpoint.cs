using Sharplight.Import.PyTorch;
using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// All shards of a safetensors checkpoint. Shards are discovered via <c>model.safetensors.index.json</c>
/// (<c>weight_map</c>) or, without an index, as all <c>*.safetensors</c> files in the directory except the heads that decision models keep
/// beside the checkpoint (<c>joint_head.safetensors</c>, <c>decision_head.safetensors</c>) and weight variants beside <c>model.safetensors</c>
/// (<c>model.fp32.safetensors</c>, ...).
/// </summary>
internal sealed class SafetensorsCheckpoint : IDisposable
{
    private readonly List<SafetensorsFile> _files = [];
    private readonly Dictionary<string, SafetensorsTensor> _tensors = new(StringComparer.Ordinal);

    /// <summary>The tensors of all shards by name.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensor> Tensors => _tensors;

    // Heads stored beside a checkpoint and imported on their own (decision models), not shards of it.
    private static readonly HashSet<string> s_sideFiles = new(StringComparer.Ordinal) { "joint_head.safetensors", "decision_head.safetensors" };

    private SafetensorsCheckpoint()
    {
    }

    /// <summary>Opens every shard of the checkpoint in <paramref name="directory"/>.</summary>
    /// <exception cref="FileNotFoundException">No shard exists, or the index lists a missing shard.</exception>
    /// <exception cref="InvalidDataException">A tensor name appears in more than one shard.</exception>
    /// <param name="directory">The checkpoint directory.</param>
    /// <param name="torch">Unpickler settings for <c>pytorch_model*.bin</c> shards, or <see langword="null"/> for the defaults.</param>
    public static SafetensorsCheckpoint Open(string directory, TorchCheckpointOptions? torch = null)
    {
        var checkpoint = new SafetensorsCheckpoint();
        try
        {
            foreach (var shard in DiscoverShards(directory))
            {
                var file = shard.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) ? SafetensorsFile.Open(shard) : SafetensorsFile.OpenTorch(shard, torch);
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
                throw new FileNotFoundException($"No *.safetensors or pytorch_model*.bin files in '{directory}'.");
            }

            return checkpoint;
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }

    /// <summary>True when <paramref name="dir"/> holds weights: safetensors shards or, failing those, PyTorch <c>pytorch_model*.bin</c> shards.</summary>
    public static bool HasWeights(string dir) => HasSafetensors(dir) || TorchShards(dir).Any();

    private static bool HasSafetensors(string dir) =>
        Directory.EnumerateFiles(dir, "*.safetensors").Any(p => !s_sideFiles.Contains(Path.GetFileName(p)));

    // PyTorch checkpoints are read only when the directory has no safetensors, as transformers prefers safetensors.
    private static IEnumerable<string> TorchShards(string dir)
    {
        string? index = Directory.EnumerateFiles(dir, "pytorch_model*.bin.index.json").OrderBy(p => p, StringComparer.Ordinal).FirstOrDefault();
        if (index != null)
        {
            using var doc = JsonDocument.Parse(File.ReadAllBytes(index));
            var names = new SortedSet<string>(StringComparer.Ordinal);
            if (doc.RootElement.TryGetProperty("weight_map", out var weightMap))
            {
                foreach (var entry in weightMap.EnumerateObject())
                {
                    if (entry.Value.GetString() is { } shardName && Path.GetFileName(shardName) == shardName)
                    {
                        names.Add(shardName);
                    }
                }
            }

            return names.Select(shardName =>
            {
                var full = Path.Combine(dir, shardName);
                return File.Exists(full) ? full : throw new FileNotFoundException($"Shard '{shardName}' listed in {Path.GetFileName(index)} is missing.", full);
            }).ToList();
        }

        return Directory.EnumerateFiles(dir, "pytorch_model*.bin").OrderBy(p => p, StringComparer.Ordinal).ToList();
    }

    private static IEnumerable<string> DiscoverShards(string dir)
    {
        if (!HasSafetensors(dir))
        {
            return TorchShards(dir);
        }

        var all = Directory.EnumerateFiles(dir, "*.safetensors").Where(p => !s_sideFiles.Contains(Path.GetFileName(p)))
            .OrderBy(p => p, StringComparer.Ordinal).ToList();
        string? index = Directory.EnumerateFiles(dir, "*.safetensors.index.json").OrderBy(p => p, StringComparer.Ordinal).FirstOrDefault();
        if (index == null)
        {
            return WithoutVariants(all);
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

    /// <summary>
    /// Drops weight variants (<c>model.fp32.safetensors</c>, <c>model.fp16.safetensors</c>, ...) and open_clip's export (<c>open_clip_*.safetensors</c>) when the default <c>model.safetensors</c> is
    /// present: transformers loads a variant only when asked for it, and a variant repeats every tensor of the default file.
    /// </summary>
    /// <param name="shards">The safetensors files of the directory.</param>
    private static List<string> WithoutVariants(List<string> shards)
    {
        if (!shards.Any(p => Path.GetFileName(p) == "model.safetensors"))
        {
            return shards;
        }

        return shards.Where(p =>
        {
            string name = Path.GetFileName(p);
            // open_clip_model.safetensors (laion repositories) is the same checkpoint in open_clip's tensor names.
            return name == "model.safetensors" || !(name.StartsWith("model.", StringComparison.Ordinal) || name.StartsWith("open_clip_", StringComparison.Ordinal));
        }).ToList();
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
