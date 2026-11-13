using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face checkpoint importer: a directory with <c>config.json</c>, <c>*.safetensors</c> shards and tokenizer
/// files is translated to the same canonical <see cref="ModelSource"/> a GGUF conversion would yield, so
/// architecture adapters run unchanged. Per-architecture knowledge lives in <see cref="IHfArchitectureMapping"/>s.
/// </summary>
public sealed class HuggingFaceImporter(HfImportOptions? options = null) : IModelImporter
{
    public string Format => "hf";

    /// <summary>Registers the Hugging Face importer with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new HuggingFaceImporter());

    public bool CanOpen(string path)
    {
        try
        {
            string? dir = ResolveDirectory(path);
            return dir != null && File.Exists(Path.Combine(dir, "config.json"))
                   && Directory.EnumerateFiles(dir, "*.safetensors").Any();
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    ModelSource IModelImporter.Open(string path) => Open(path, options);

    static string? ResolveDirectory(string path)
    {
        if (Directory.Exists(path)) return Path.GetFullPath(path);
        if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            return Path.GetDirectoryName(Path.GetFullPath(path));
        return null;
    }

    /// <summary>Opens a Hugging Face checkpoint directory (or any <c>.safetensors</c> file inside one).</summary>
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
        try
        {
            var md = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Architecture] = mapping.Architecture,
                [CanonicalKeys.Name] = new DirectoryInfo(dir).Name,
            };
            foreach (var (k, v) in mapping.MapHyperParameters(config, options)) md[k] = v;
            HfTokenizerTranslator.AddTo(md, dir, config);

            return new ModelSource
            {
                Format = "hf",
                Architecture = mapping.Architecture,
                Metadata = md,
                Weights = mapping.MapWeights(config, checkpoint.Tensors, options),
                OwnedResources = [checkpoint],
                Path = path,
            };
        }
        catch
        {
            checkpoint.Dispose();
            throw;
        }
    }
}
