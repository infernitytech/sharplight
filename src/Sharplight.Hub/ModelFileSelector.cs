using System.Text.Json;

namespace Sharplight.Hub;

/// <summary>
/// Chooses the files of one model variant in a repository: a GGUF quantization (all shards of a split model, plus a vision
/// projector when the repository has one), a safetensors checkpoint (configuration, tokenizer, index and the shards it lists)
/// or an ONNX model file with its external data.
/// </summary>
public static class ModelFileSelector
{
    /// <summary>Tag that selects the safetensors checkpoint of a repository that also has GGUF files.</summary>
    public const string SafetensorsTag = "safetensors";

    /// <summary>Tag that selects the default ONNX model of a repository that also has other formats.</summary>
    public const string OnnxTag = "onnx";

    /// <summary>Tag accepted for the single variant of a safetensors or ONNX repository.</summary>
    public const string DefaultTag = "default";

    private const string SafetensorsExtension = ".safetensors";
    private const string IndexSuffix = ".safetensors.index.json";
    private const string OnnxExtension = ".onnx";
    private const string OnnxDirectory = "onnx";
    private const string OnnxDefaultStem = "model";
    private const string ConfigFile = "config.json";
    private const string PoolingDirectorySuffix = "_Pooling";
    private const string WeightMapProperty = "weight_map";
    private static readonly string[] s_metadataExtensions = [".json", ".jinja"];
    private static readonly string[] s_metadataFiles = ["tokenizer.model"];
    private static readonly string[] s_projectorPreference = ["F16", "BF16", "F32", "Q8_0"];

    /// <summary>
    /// GGUF quantizations tried in order when no tag is given: a good quality/size balance first, then near-lossless 8-bit,
    /// then other common choices. A repository with none of them gets its smallest variant.
    /// </summary>
    public static IReadOnlyList<string> DefaultQuantizations { get; } =
        ["Q4_K_M", "Q8_0", "Q4_K_S", "Q5_K_M", "Q4_0", "Q6_K", "Q5_K_S", "IQ4_XS", "IQ4_NL", "UD-Q4_K_XL", "Q4_K_L", "Q5_0", "Q8_K_XL", "BF16", "F16", "F32"];

    /// <summary>The format a repository with these files is pulled as, or null when it has no supported model files.</summary>
    /// <param name="paths">Repository paths.</param>
    public static ModelFormat? DetectFormat(IEnumerable<string> paths)
    {
        var list = paths.ToList();
        if (list.Any(IsGgufModel))
        {
            return ModelFormat.Gguf;
        }

        if (IsCheckpoint(list))
        {
            return ModelFormat.Safetensors;
        }

        return OnnxCandidates(list).Any() ? ModelFormat.Onnx : null;
    }

    /// <summary>The tags a repository with these files offers, most recommended first; empty for a single variant.</summary>
    /// <param name="paths">Repository paths.</param>
    public static IReadOnlyList<string> ListVariants(IEnumerable<string> paths)
    {
        var list = paths.ToList();
        return DetectFormat(list) switch
        {
            ModelFormat.Gguf => [.. OrderByPreference(GgufVariants(list)).Select(variant => variant.Label)],
            ModelFormat.Onnx => [.. OnnxVariants(list).Select(variant => variant.Label).Where(label => label.Length > 0)],
            _ => [],
        };
    }

    /// <summary>The safetensors index the selection needs (download it and pass its text to <see cref="Select"/>), or null.</summary>
    /// <param name="repository">The repository.</param>
    /// <param name="tag">The requested tag, or null.</param>
    public static string? IndexPath(HubRepository repository, string? tag)
    {
        var paths = repository.Files.Select(file => file.Path).ToList();
        bool safetensors = string.Equals(tag, SafetensorsTag, StringComparison.OrdinalIgnoreCase) || DetectFormat(paths) == ModelFormat.Safetensors;
        return safetensors ? paths.Where(IsRootFile).Where(path => path.EndsWith(IndexSuffix, StringComparison.OrdinalIgnoreCase)).Order(StringComparer.Ordinal).FirstOrDefault() : null;
    }

    /// <summary>Chooses the files of the variant <paramref name="tag"/> (or the default variant).</summary>
    /// <param name="repository">The repository and its files.</param>
    /// <param name="tag">A quantization label, file name or ONNX variant; null for the default.</param>
    /// <param name="indexJson">Text of the file named by <see cref="IndexPath"/>, when there is one.</param>
    /// <exception cref="HubException">The repository has no supported model files, the tag matches no variant, or a split model is incomplete.</exception>
    public static ModelSelection Select(HubRepository repository, string? tag, string? indexJson = null)
    {
        var paths = repository.Files.Select(file => file.Path).ToList();
        var format = ForcedFormat(tag) ?? DetectFormat(paths)
            ?? throw new HubException($"{repository.Id} has no model files Sharplight can pull (GGUF, safetensors checkpoint or ONNX).");
        if (ForcedFormat(tag) is not null)
        {
            tag = null;
        }

        return format switch
        {
            ModelFormat.Gguf => SelectGguf(repository, tag),
            ModelFormat.Safetensors => SelectCheckpoint(repository, tag, indexJson),
            _ => SelectOnnx(repository, tag),
        };
    }

    private static ModelFormat? ForcedFormat(string? tag)
    {
        if (string.Equals(tag, SafetensorsTag, StringComparison.OrdinalIgnoreCase))
        {
            return ModelFormat.Safetensors;
        }

        return string.Equals(tag, OnnxTag, StringComparison.OrdinalIgnoreCase) ? ModelFormat.Onnx : null;
    }

    private static bool IsRootFile(string path) => RepoPath.IsRoot(path);

    private static bool IsGgufModel(string path) => GgufFileName.IsGguf(path) && !GgufFileName.IsProjector(path);

    private static bool IsCheckpoint(IReadOnlyList<string> paths)
    {
        var root = paths.Where(IsRootFile).ToList();
        return root.Contains(ConfigFile, StringComparer.Ordinal) && root.Any(path => path.EndsWith(SafetensorsExtension, StringComparison.OrdinalIgnoreCase));
    }

    private static ModelSelection SelectGguf(HubRepository repository, string? tag)
    {
        var variants = GgufVariants(repository.Files.Select(file => file.Path).ToList());
        if (variants.Count == 0)
        {
            throw new HubException($"{repository.Id} has no GGUF model files.");
        }

        var chosen = tag is null ? DefaultGguf(variants, repository) : FindGguf(variants, tag)
            ?? throw new HubException($"{repository.Id} has no variant '{tag}'. Available: {string.Join(", ", OrderByPreference(variants).Select(variant => variant.Label))}.");
        var files = chosen.Paths.Select(path => repository.Files.First(file => file.Path == path)).ToList();
        var primary = CheckShards(repository, chosen);
        var projector = SelectProjector(repository, primary);
        if (projector is not null)
        {
            files.Add(projector);
        }

        return new ModelSelection(ModelFormat.Gguf, chosen.Label, files, primary, projector?.Path);
    }

    /// <summary>Returns the first shard after checking that every shard of a split model is present.</summary>
    private static string CheckShards(HubRepository repository, GgufVariant variant)
    {
        var shards = variant.Paths.Select(path => (Path: path, Shard: GgufFileName.Shard(path))).ToList();
        if (shards.All(shard => shard.Shard is null))
        {
            return shards[0].Path;
        }

        int count = shards.Max(shard => shard.Shard?.Count ?? 0);
        var indices = shards.Where(shard => shard.Shard is not null).Select(shard => shard.Shard!.Value.Index).ToHashSet();
        if (indices.Count != count || Enumerable.Range(1, count).Any(index => !indices.Contains(index)))
        {
            throw new HubException($"{repository.Id}: variant '{variant.Label}' is split into {count} files but only {indices.Count} are in the repository.");
        }

        return shards.First(shard => shard.Shard?.Index == 1).Path;
    }

    private static HubFile? SelectProjector(HubRepository repository, string modelPath)
    {
        var projectors = repository.Files.Where(file => GgufFileName.IsGguf(file.Path) && GgufFileName.IsProjector(file.Path)).ToList();
        if (projectors.Count == 0)
        {
            return null;
        }

        // A projector next to the chosen model wins (repositories with one directory per variant).
        var directory = RepoPath.Directory(modelPath);
        var nearby = projectors.Where(file => file.Directory == directory).ToList();
        var candidates = nearby.Count > 0 ? nearby : projectors;
        foreach (var label in s_projectorPreference)
        {
            if (candidates.FirstOrDefault(file => string.Equals(GgufFileName.QuantLabel(file.Path), label, StringComparison.OrdinalIgnoreCase)) is { } preferred)
            {
                return preferred;
            }
        }

        return candidates.OrderBy(file => file.Path, StringComparer.Ordinal).First();
    }

    private static GgufVariant DefaultGguf(IReadOnlyList<GgufVariant> variants, HubRepository repository)
    {
        var preferred = OrderByPreference(variants).First();
        if (Preference(preferred.Label) < DefaultQuantizations.Count)
        {
            return preferred;
        }

        return variants.OrderBy(variant => variant.Paths.Sum(path => repository.Files.First(file => file.Path == path).Size)).First();
    }

    private static GgufVariant? FindGguf(IReadOnlyList<GgufVariant> variants, string tag)
    {
        var byLabel = variants.FirstOrDefault(variant => string.Equals(variant.Label, tag, StringComparison.OrdinalIgnoreCase));
        if (byLabel is not null)
        {
            return byLabel;
        }

        // A file name or path, with or without the extension, of the model or of any of its shards.
        var name = tag.EndsWith(GgufFileName.Extension, StringComparison.OrdinalIgnoreCase) ? tag : tag + GgufFileName.Extension;
        return variants.FirstOrDefault(variant => variant.Paths.Any(path =>
            string.Equals(path, name, StringComparison.OrdinalIgnoreCase)
            || string.Equals(RepoPath.Name(path), name, StringComparison.OrdinalIgnoreCase)));
    }

    private static int Preference(string label)
    {
        int index = DefaultQuantizations.ToList().FindIndex(preferred => string.Equals(preferred, label, StringComparison.OrdinalIgnoreCase));
        return index >= 0 ? index : DefaultQuantizations.Count;
    }

    private static IEnumerable<GgufVariant> OrderByPreference(IEnumerable<GgufVariant> variants)
    {
        return variants.OrderBy(variant => Preference(variant.Label)).ThenBy(variant => variant.Label, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>Groups GGUF model files (shards together) and labels each group with its quantization, falling back to file names when labels collide.</summary>
    private static List<GgufVariant> GgufVariants(IReadOnlyList<string> paths)
    {
        var groups = paths.Where(IsGgufModel)
            .GroupBy(GgufFileName.ModelKey, StringComparer.Ordinal)
            .Select(group => (Key: group.Key, Paths: group.Order(StringComparer.Ordinal).ToList(), Label: GgufFileName.QuantLabel(group.First())))
            .ToList();
        var labelCounts = groups.Where(group => group.Label is not null)
            .GroupBy(group => group.Label!, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(group => group.Key, group => group.Count(), StringComparer.OrdinalIgnoreCase);
        return [.. groups.Select(group =>
        {
            bool unique = group.Label is not null && labelCounts[group.Label] == 1;
            var label = unique ? group.Label! : RepoPath.Name(group.Key);
            return new GgufVariant(label, group.Paths);
        })];
    }

    private static ModelSelection SelectCheckpoint(HubRepository repository, string? tag, string? indexJson)
    {
        if (tag is not null && !string.Equals(tag, DefaultTag, StringComparison.OrdinalIgnoreCase))
        {
            throw new HubException($"{repository.Id} is a single safetensors checkpoint without variants; drop ':{tag}'.");
        }

        var root = repository.Files.Where(file => IsRootFile(file.Path)).ToList();
        if (!root.Any(file => file.Path == ConfigFile))
        {
            throw new HubException($"{repository.Id} has no {ConfigFile}; it is not a Hugging Face checkpoint.");
        }

        var files = root.Where(IsCheckpointMetadata).ToList();
        files.AddRange(repository.Files.Where(IsPoolingConfig));
        files.AddRange(indexJson is null
            ? root.Where(file => file.Path.EndsWith(SafetensorsExtension, StringComparison.OrdinalIgnoreCase))
            : IndexedShards(repository, indexJson));
        return new ModelSelection(ModelFormat.Safetensors, "", files, "", null);
    }

    private static bool IsCheckpointMetadata(HubFile file)
    {
        return s_metadataExtensions.Any(extension => file.Path.EndsWith(extension, StringComparison.OrdinalIgnoreCase))
            || s_metadataFiles.Contains(file.Path, StringComparer.Ordinal);
    }

    /// <summary>The <c>config.json</c> of a sentence-transformers pooling module (<c>1_Pooling/config.json</c>).</summary>
    private static bool IsPoolingConfig(HubFile file)
    {
        return file.Name == ConfigFile && file.Directory.EndsWith(PoolingDirectorySuffix, StringComparison.Ordinal) && !file.Directory.Contains('/');
    }

    private static IEnumerable<HubFile> IndexedShards(HubRepository repository, string indexJson)
    {
        HashSet<string> shardNames;
        try
        {
            using var document = JsonDocument.Parse(indexJson);
            shardNames = document.RootElement.GetProperty(WeightMapProperty).EnumerateObject()
                .Select(entry => entry.Value.GetString() ?? "")
                .ToHashSet(StringComparer.Ordinal);
        }
        catch (Exception error) when (error is JsonException or KeyNotFoundException or InvalidOperationException)
        {
            throw new HubException($"{repository.Id}: the safetensors index is not valid ({error.Message}).", inner: error);
        }

        foreach (var name in shardNames.Order(StringComparer.Ordinal))
        {
            yield return repository.Files.FirstOrDefault(file => file.Path == name)
                ?? throw new HubException($"{repository.Id}: the safetensors index lists '{name}', which is not in the repository.");
        }
    }

    private static ModelSelection SelectOnnx(HubRepository repository, string? tag)
    {
        var variants = OnnxVariants(repository.Files.Select(file => file.Path).ToList());
        var label = tag is null || string.Equals(tag, DefaultTag, StringComparison.OrdinalIgnoreCase) ? "" : tag;
        var chosen = variants.FirstOrDefault(variant => string.Equals(variant.Label, label, StringComparison.OrdinalIgnoreCase))
            ?? variants.FirstOrDefault(variant => string.Equals(variant.Path, label, StringComparison.OrdinalIgnoreCase) || string.Equals(RepoPath.Name(variant.Path), label, StringComparison.OrdinalIgnoreCase))
            ?? (tag is null ? variants.FirstOrDefault() : null)
            ?? throw new HubException($"{repository.Id} has no ONNX variant '{tag}'. Available: {string.Join(", ", variants.Select(variant => variant.Label.Length == 0 ? DefaultTag : variant.Label))}.");

        var model = repository.Files.First(file => file.Path == chosen.Path);
        var externalData = repository.Files.Where(file => file.Directory == model.Directory && IsExternalData(model.Name, file.Name));
        return new ModelSelection(ModelFormat.Onnx, chosen.Label, [model, .. externalData], model.Path, null);
    }

    private static bool IsExternalData(string modelName, string candidate)
    {
        var stem = modelName[..^OnnxExtension.Length];
        return candidate != modelName
            && (candidate.StartsWith(modelName + "_data", StringComparison.Ordinal)
                || candidate.StartsWith(modelName + ".data", StringComparison.Ordinal)
                || candidate.StartsWith(stem + ".data", StringComparison.Ordinal));
    }

    private static IEnumerable<string> OnnxCandidates(IReadOnlyList<string> paths)
    {
        return paths.Where(path => path.EndsWith(OnnxExtension, StringComparison.OrdinalIgnoreCase)
            && (IsRootFile(path) || RepoPath.Directory(path) == OnnxDirectory));
    }

    /// <summary>ONNX model files at the root or in <c>onnx/</c>; <c>model.onnx</c> is the default (empty label), <c>model_fp16.onnx</c> is <c>fp16</c>.</summary>
    private static List<OnnxVariant> OnnxVariants(IReadOnlyList<string> paths)
    {
        var variants = new List<OnnxVariant>();
        foreach (var path in OnnxCandidates(paths).OrderBy(path => IsRootFile(path) ? 1 : 0).ThenBy(path => path, StringComparer.Ordinal))
        {
            var stem = RepoPath.Name(path)[..^OnnxExtension.Length];
            var label = stem == OnnxDefaultStem ? "" : stem.StartsWith(OnnxDefaultStem + "_", StringComparison.Ordinal) ? stem[(OnnxDefaultStem.Length + 1)..] : stem;
            if (!variants.Any(variant => string.Equals(variant.Label, label, StringComparison.OrdinalIgnoreCase)))
            {
                variants.Add(new OnnxVariant(label, path));
            }
        }

        return variants;
    }
}
