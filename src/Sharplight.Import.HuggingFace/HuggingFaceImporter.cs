using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face checkpoint importer: a directory with <c>config.json</c>, <c>*.safetensors</c> shards and tokenizer
/// files is translated to the same canonical <see cref="ModelSource"/> a GGUF conversion would yield, so
/// architecture adapters run unchanged. Per-architecture knowledge (hyper-parameter and tensor name maps) is selected by the
/// config's <c>model_type</c>; Hugging Face naming stays inside this assembly.
/// </summary>
/// <remarks>
/// Register the importer with <c>services.AddHuggingFaceImporter()</c> to let the <see cref="ModelLoader"/> open checkpoint
/// directories, or call <see cref="Open"/> directly. Weights are zero-copy views over the memory-mapped safetensors
/// shards where names are the only change, and lazily derived otherwise.
/// </remarks>
/// <param name="options">Import options used when opened through the <see cref="ModelLoader"/>; null for the defaults.</param>
public sealed class HuggingFaceImporter(HuggingFaceImportOptions? options = null) : IModelImporter
{
    /// <inheritdoc/>
    public string Format => "hf";

    /// <inheritdoc/>
    /// <remarks>True for a directory (or a <c>.safetensors</c> file inside one) that holds <c>config.json</c> and at least one <c>*.safetensors</c> file (a Decision 2.0 package keeps its weights in <c>backbone/</c>) or PyTorch <c>pytorch_model*.bin</c> shard (read through the restricted unpickler).</remarks>
    public bool CanOpen(string path)
    {
        try
        {
            string? dir = ResolveDirectory(path);
            return dir != null && File.Exists(Path.Combine(dir, "config.json"))
                   && (SafetensorsCheckpoint.HasWeights(dir) || HfDecisionPackage.Find(dir) != null);
        }
        catch (InvalidDataException)
        {
            return false;
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

        if (File.Exists(path) && (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                                || Path.GetFileName(path).StartsWith("pytorch_model", StringComparison.OrdinalIgnoreCase) && path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase)))
        {
            return Path.GetDirectoryName(Path.GetFullPath(path));
        }

        return null;
    }

    /// <summary>Opens a Hugging Face checkpoint directory (or any <c>.safetensors</c> file inside one).</summary>
    /// <param name="path">The checkpoint directory, or a <c>.safetensors</c> file whose directory is the checkpoint.</param>
    /// <param name="options">Import options; null for <see cref="HuggingFaceImportOptions.Default"/>.</param>
    /// <returns>
    /// A model source named after the directory. It owns the memory-mapped shards; dispose it to unmap them.
    /// </returns>
    /// <exception cref="FileNotFoundException"><paramref name="path"/> is neither a directory nor a <c>.safetensors</c> file, or the checkpoint has no shards.</exception>
    /// <exception cref="NotSupportedException">
    /// Tensors have no canonical mapping, or the mapping rejects the checkpoint's hyper-parameters or features (as
    /// <see cref="HfSupport.Validate"/> reports them).
    /// </exception>
    /// <exception cref="UnsupportedArchitectureException">The <c>model_type</c> has no mapping.</exception>
    /// <exception cref="UnsupportedWeightFormatException">The checkpoint is quantized with a method the importer does not read (see <see cref="HfSupport.GetQuantizationSupport"/>).</exception>
    public static ModelSource Open(string path, HuggingFaceImportOptions? options = null)
    {
        options ??= HuggingFaceImportOptions.Default;
        string dir = ResolveDirectory(path) ?? throw new FileNotFoundException($"'{path}' is neither a directory nor a .safetensors file.");
        // A Decision 2.0 package keeps the backbone checkpoint in a subdirectory and the span scoring head beside it.
        var decision = HfDecisionPackage.Find(dir);
        string weightsDir = decision?.BackboneDirectory ?? dir;
        using var config = HfConfig.Load(weightsDir);
        var mapping = HfArchitectureMappings.Find(config.ModelType)
            ?? throw new UnsupportedArchitectureException(config.ModelType ?? string.Join(", ", config.Architectures),
                $"Hugging Face model_type '{config.ModelType}' ({string.Join(", ", config.Architectures)}) is not supported. " +
                $"Supported: {string.Join(", ", HfArchitectureMappings.All.SelectMany(m => m.ModelTypes))}.");
        RejectUnsupportedQuantization(config, dir);
        if (mapping.Validate(config).ToList() is { Count: > 0 } problems)
        {
            throw new NotSupportedException($"{dir}: {string.Join(" ", problems)}");
        }

        var checkpoint = SafetensorsCheckpoint.Open(weightsDir, options.PyTorch);
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

            if (decision != null)
            {
                HfDecisionPackage.AddHyperParameters(md);
            }

            // Cloudflare Clef: a joint schema head beside the checkpoint.
            bool jointHead = HfJointSchemaHead.IsPresent(dir);
            if (jointHead)
            {
                HfJointSchemaHead.AddHyperParameters(md, dir);
            }

            // EAGLE-3 drafters ship without a tokenizer: they use their target's.
            if (mapping.NeedsTokenizer && (!Eagle3HfMapping.IsDrafter(config) || File.Exists(Path.Combine(dir, "tokenizer.json"))))
            {
                HfTokenizerTranslator.AddTo(md, File.Exists(Path.Combine(dir, "tokenizer.json")) ? dir : weightsDir, config);
            }
            HfSentenceTransformers.AddTo(md, dir, config);
            mapping.AddMetadata(md, config, dir);

            var weights = mapping.MapWeights(config, HfQuantizedWeights.Resolve(config, checkpoint.Tensors), options);
            if (decision != null)
            {
                weights = weights.Concat(decision.HeadWeights(owned)).ToDictionary(StringComparer.Ordinal);
            }

            if (jointHead)
            {
                weights = weights.Concat(HfJointSchemaHead.Weights(dir, owned)).ToDictionary(StringComparer.Ordinal);
            }

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
                Attachments = mapping.MapAttachments(config, dir),
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

    /// <summary>Fails early, with the method's name, for checkpoints whose quantized weights cannot be read.</summary>
    /// <exception cref="UnsupportedWeightFormatException">The quantization method is not read.</exception>
    private static void RejectUnsupportedQuantization(HfConfig config, string dir)
    {
        string modelOptPath = Path.Combine(dir, "hf_quant_config.json");
        using var modelOpt = File.Exists(modelOptPath) ? System.Text.Json.JsonDocument.Parse(File.ReadAllText(modelOptPath)) : null;
        var quantization = HfSupport.AssessQuantization(config, modelOpt?.RootElement);
        if (quantization.Level == SupportLevel.Unsupported)
        {
            throw new UnsupportedWeightFormatException(quantization.Method!,
                $"{new DirectoryInfo(dir).Name}: weights quantized with {quantization.Method} are not supported (supported: unquantized, FP8, NVFP4, MXFP4, GPTQ, AWQ, compressed-tensors, bitsandbytes).");
        }
    }
}
