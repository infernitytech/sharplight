using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face checkpoint importer: a directory with <c>config.json</c>, <c>*.safetensors</c> shards and tokenizer
/// files is translated to the same canonical <see cref="ModelSource"/> a GGUF conversion would yield, so
/// architecture adapters run unchanged. Per-architecture knowledge (hyper-parameter and tensor name maps) is selected by the
/// config's <c>model_type</c>; Hugging Face naming stays inside this assembly.
/// </summary>
/// <remarks>
/// Register the importer once with <see cref="Register"/> to let <see cref="ModelImporterRegistry"/> open checkpoint
/// directories, or call <see cref="Open"/> directly. Weights are zero-copy views over the memory-mapped safetensors
/// shards where names are the only change, and lazily derived otherwise.
/// </remarks>
/// <param name="options">Import options used when opened through <see cref="ModelImporterRegistry"/>; null for the defaults.</param>
public sealed class HuggingFaceImporter(HfImportOptions? options = null) : IModelImporter
{
    /// <inheritdoc/>
    public string Format => "hf";

    /// <summary>Registers the Hugging Face importer with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new HuggingFaceImporter());

    /// <inheritdoc/>
    /// <remarks>True for a directory (or a <c>.safetensors</c> file inside one) that holds <c>config.json</c> and at least one <c>*.safetensors</c> file.</remarks>
    public bool CanOpen(string path)
    {
        try
        {
            string? dir = ResolveDirectory(path);
            return dir != null && File.Exists(Path.Combine(dir, "config.json"))
                   && Directory.EnumerateFiles(dir, "*.safetensors").Any();
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }

    ModelSource IModelImporter.Open(string path) => Open(path, options);

    private static string? ResolveDirectory(string path)
    {
        if (Directory.Exists(path))
        {
            return Path.GetFullPath(path);
        }

        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return Path.GetDirectoryName(Path.GetFullPath(path));
        }

        return null;
    }

    /// <summary>Opens a Hugging Face checkpoint directory (or any <c>.safetensors</c> file inside one).</summary>
    /// <param name="path">The checkpoint directory, or a <c>.safetensors</c> file whose directory is the checkpoint.</param>
    /// <param name="options">Import options; null for <see cref="HfImportOptions.Default"/>.</param>
    /// <returns>
    /// A model source named after the directory. It owns the memory-mapped shards; dispose it to unmap them.
    /// </returns>
    /// <exception cref="FileNotFoundException"><paramref name="path"/> is neither a directory nor a <c>.safetensors</c> file, or the checkpoint has no shards.</exception>
    /// <exception cref="NotSupportedException">The <c>model_type</c> is unsupported or tensors have no canonical mapping.</exception>
    public static ModelSource Open(string path, HfImportOptions? options = null)
    {
        options ??= HfImportOptions.Default;
        string dir = ResolveDirectory(path) ?? throw new FileNotFoundException($"'{path}' is neither a directory nor a .safetensors file.");
        using var config = HfConfig.Load(dir);
        var mapping = HfArchitectureMappings.Find(config.ModelType)
            ?? throw new NotSupportedException(
                $"Hugging Face model_type '{config.ModelType}' ({string.Join(", ", config.Architectures)}) is not supported. " +
                $"Supported: {string.Join(", ", HfArchitectureMappings.All.SelectMany(m => m.ModelTypes))}.");

        var checkpoint = SafetensorsCheckpoint.Open(dir);
        List<IDisposable> owned = [checkpoint];
        try
        {
            var md = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = mapping.Architecture,
                [CanonicalKeys.Name] = new DirectoryInfo(dir).Name,
            };
            foreach (var (k, v) in mapping.MapHyperParameters(config, options))
            {
                md[k] = v;
            }

            HfTokenizerTranslator.AddTo(md, dir, config);
            HfSentenceTransformers.AddTo(md, dir, config);

            var weights = mapping.MapWeights(config, HfFp8Weights.Resolve(config, checkpoint.Tensors), options);
            var dense = HfSentenceTransformers.DenseWeights(dir, owned);
            if (dense.Count > 0)
            {
                weights = weights.Concat(dense).ToDictionary(StringComparer.Ordinal);
            }

            return new ModelSource
            {
                Format = "hf",
                Architecture = mapping.Architecture,
                Metadata = md,
                Weights = weights,
                OwnedResources = owned,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in owned)
            {
                resource.Dispose();
            }

            throw;
        }
    }
}
