using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// GGUF importer: maps GGUF files (llama.cpp's format) onto the canonical <see cref="ModelSource"/>.
/// llama.cpp's architecture ids, hyper-parameter keys and tensor names are translated by an architecture mapping
/// per <c>general.architecture</c>; tokenizer keys (<c>tokenizer.ggml.*</c>) map to
/// <see cref="CanonicalKeys"/>. Weight data is shared zero-copy (only names change); where the file layout differs from
/// the canonical one (fused or permuted projections) weights are derived lazily on first read.
/// </summary>
/// <remarks>
/// Register the importer once with <see cref="Register"/> to let <see cref="ModelImporterRegistry"/> open <c>.gguf</c>
/// paths, or call <see cref="Open"/> directly. Split models are opened through their first shard.
/// </remarks>
public sealed class GgufImporter : IModelImporter
{
    private const string GgmlTokenizerPrefix = "tokenizer.ggml.";
    private const int MaxUnmappedNamesReported = 10;

    /// <inheritdoc/>
    public string Format => "gguf";

    /// <summary>Registers the GGUF importer with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new GgufImporter());

    /// <inheritdoc/>
    /// <remarks>True when <paramref name="path"/> is an existing file starting with the GGUF magic bytes.</remarks>
    public bool CanOpen(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var file = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[4];
            return file.Read(magic) == 4 && magic.SequenceEqual("GGUF"u8);
        }
        catch (IOException)
        {
            return false;
        }
    }

    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a GGUF model (single file or first shard of a split model).</summary>
    /// <param name="path">The <c>.gguf</c> file, or the <c>-00001-of-N.gguf</c> shard of a split model.</param>
    /// <returns>
    /// A model source whose weights are views over the memory-mapped file(s). The source owns the mapping; dispose it
    /// to unmap the file.
    /// </returns>
    /// <exception cref="InvalidDataException">The file is malformed or two tensors map to the same canonical name.</exception>
    /// <exception cref="NotSupportedException">
    /// Some tensors have no canonical mapping, or the mapping rejects the file's hyper-parameters or features (the same checks as
    /// <see cref="GgufSupport.Validate"/>).
    /// </exception>
    /// <exception cref="UnsupportedArchitectureException">The architecture has no mapping.</exception>
    /// <exception cref="UnsupportedWeightFormatException">A tensor's ggml type id is unknown.</exception>
    public static ModelSource Open(string path)
    {
        var files = GgufModelFiles.Open(path);
        try
        {
            string ggufArchitecture = files.Architecture ?? throw new InvalidDataException($"{path}: missing general.architecture");
            var mapping = GgufArchitectureMappings.Find(ggufArchitecture) ?? throw new UnsupportedArchitectureException(ggufArchitecture,
                $"{path}: GGUF architecture '{ggufArchitecture}' has no canonical mapping. Supported: {string.Join(", ", GgufArchitectureMappings.All.SelectMany(m => m.GgufArchitectures))}");
            var view = new GgufMetadataView(files.Metadata, ggufArchitecture, files.FindTensor);
            if (mapping.Validate(view).ToList() is { Count: > 0 } problems)
            {
                throw new NotSupportedException($"{path}: {string.Join(" ", problems)}");
            }

            var (architecture, metadata, weights) = Map(path, files.Metadata, files.Tensors, mapping, view);
            return new ModelSource
            {
                Format = "gguf",
                Architecture = architecture,
                Metadata = metadata,
                Weights = weights,
                OwnedResources = [files],
                Path = path,
            };
        }
        catch
        {
            files.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The mapping steps of an import after <see cref="IGgufArchitectureMapping.Validate"/>: canonical weights (renamed, split, transformed)
    /// and canonical metadata. They read tensor data only lazily, so <see cref="GgufSupport.Validate"/> runs them on a header alone.
    /// </summary>
    /// <param name="path">The file, for errors.</param>
    /// <param name="gguf">The GGUF metadata.</param>
    /// <param name="tensors">Every tensor of every shard.</param>
    /// <param name="mapping">The architecture's mapping.</param>
    /// <param name="view">The metadata view over <paramref name="gguf"/> and <paramref name="tensors"/>.</param>
    internal static (string Architecture, Dictionary<string, object> Metadata, Dictionary<string, WeightInfo> Weights) Map(string path,
        IReadOnlyDictionary<string, object> gguf, IReadOnlyList<GgufTensorInfo> tensors, IGgufArchitectureMapping mapping, GgufMetadataView view)
    {
        var weights = MapWeights(path, tensors, mapping, view);
        if (GgufTensorNames.HoldsOnlyNextnBlocks(tensors.Select(t => t.Name), view)
            && !weights.Keys.Any(k => k.StartsWith("mtp.layers.", StringComparison.Ordinal) || k.StartsWith("layers.", StringComparison.Ordinal)))
        {
            // An MTP draft file of an architecture whose NextN head is not imported (it maps to no layer at all): neither a model nor a
            // usable draft. Drafter architectures of their own (gemma4-assistant) map their blocks to layers and are not affected.
            throw new NotSupportedException($"{path}: the file holds only multi-token-prediction layers (an MTP draft file), and the "
                + $"multi-token-prediction head of '{view.GgufArchitecture}' is not imported, so it cannot be loaded as a draft either.");
        }

        if (PrismHadamard.Applies(gguf))
        {
            PrismHadamard.Apply(weights, gguf, name => mapping.MapWeightName(name, view));
        }

        if (mapping is IGgufWeightTransformer transformer)
        {
            transformer.Transform(weights, view);
        }

        string architecture = mapping.ResolveArchitecture(view);
        var metadata = MapMetadata(gguf, mapping, view);
        metadata[CanonicalKeys.Architecture] = architecture;
        mapping.RefineHyperParameters(metadata, view, weights);
        return (architecture, metadata, weights);
    }

    /// <summary>Renames every tensor (splitting fused ones, adding synthetic ones) to its canonical weight.</summary>
    private static Dictionary<string, WeightInfo> MapWeights(string path, IReadOnlyList<GgufTensorInfo> tensors, IGgufArchitectureMapping mapping, GgufMetadataView view)
    {
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var unmapped = new List<string>();
        foreach (var tensor in tensors)
        {
            if (mapping is IGgufSplitMapping splitter && splitter.Split(tensor.Name, tensor.ToWeightInfo(), view) is { } parts)
            {
                foreach (var part in parts)
                {
                    AddWeight(weights, path, part.Name, mapping.MapWeight(part, view));
                }

                continue;
            }

            if (mapping.MapWeightName(tensor.Name, view) is not { } name)
            {
                unmapped.Add(tensor.Name);
                continue;
            }

            AddWeight(weights, path, name, mapping.MapWeight(tensor.ToWeightInfo().As(name), view));
        }

        foreach (var synthetic in mapping.SyntheticWeights(view, weights).ToList())
        {
            if (!weights.TryAdd(synthetic.Name, synthetic))
            {
                throw new InvalidDataException($"{path}: synthetic weight '{synthetic.Name}' collides with a tensor.");
            }
        }

        if (unmapped.Count > 0)
        {
            string shown = string.Join(", ", unmapped.Take(MaxUnmappedNamesReported));
            string more = unmapped.Count > MaxUnmappedNamesReported ? ", ..." : "";
            throw new NotSupportedException($"{path}: {unmapped.Count} tensor(s) of '{view.GgufArchitecture}' have no canonical mapping: {shown}{more}");
        }

        return weights;
    }

    private static void AddWeight(Dictionary<string, WeightInfo> weights, string path, string name, WeightInfo weight)
    {
        if (!weights.TryAdd(name, weight))
        {
            throw new InvalidDataException($"{path}: two tensors map to '{name}'.");
        }
    }

    /// <summary>Canonical metadata: translated tokenizer keys plus the mapping's hyper-parameters instead of llama.cpp's per-architecture keys.</summary>
    private static Dictionary<string, object> MapMetadata(IReadOnlyDictionary<string, object> gguf, IGgufArchitectureMapping mapping, GgufMetadataView view)
    {
        var metadata = new Dictionary<string, object>(ToCanonical(gguf), StringComparer.Ordinal);
        string architecturePrefix = view.GgufArchitecture + ".";
        foreach (var key in metadata.Keys.Where(k => k.StartsWith(architecturePrefix, StringComparison.Ordinal)).ToList())
        {
            metadata.Remove(key);
        }

        foreach (var (key, value) in mapping.MapHyperParameters(view))
        {
            metadata[key] = value;
        }

        return metadata;
    }

    /// <summary>Translates GGUF metadata keys to <see cref="CanonicalKeys"/>; other keys pass through.</summary>
    private static Dictionary<string, object> ToCanonical(IReadOnlyDictionary<string, object> gguf)
    {
        var md = new Dictionary<string, object>(gguf.Count, StringComparer.Ordinal);
        foreach (var (key, value) in gguf)
        {
            string canonicalKey = key.StartsWith(GgmlTokenizerPrefix, StringComparison.Ordinal) ? "tokenizer." + key[GgmlTokenizerPrefix.Length..] : key;
            md[canonicalKey] = value;
        }

        // llama.cpp's spelling and names of the pair / classifier tokens.
        if (md.Remove("tokenizer.seperator_token_id", out var separator))
        {
            md[CanonicalKeys.TokenizerSeparatorId] = separator;
        }

        if (md.Remove("tokenizer.cls_token_id", out var classifier))
        {
            md[CanonicalKeys.TokenizerClassifierId] = classifier;
        }

        md.TryGetValue(CanonicalKeys.TokenizerModel, out var tokenizerModel);
        switch (tokenizerModel)
        {
            // llama.cpp calls SentencePiece-style tokenizers "llama"; the canonical name is "spm".
            case "llama":
                md[CanonicalKeys.TokenizerModel] = "spm";
                break;
            // llama.cpp's "gemma4" tokenizer is a character-level, merge-ranked BPE with ▁ spaces and byte fallback: canonical "bpe".
            case "gemma4":
                md[CanonicalKeys.TokenizerModel] = "bpe";
                break;
            // SentencePiece unigram models (T5, XLM-RoBERTa) are "t5" in llama.cpp.
            case "t5":
                md[CanonicalKeys.TokenizerModel] = "unigram";
                break;
            case "bert":
                md[CanonicalKeys.TokenizerModel] = "wordpiece";
                if (md.TryGetValue(CanonicalKeys.TokenizerTokens, out var tokens) && tokens is string[] pieces)
                {
                    md[CanonicalKeys.TokenizerTokens] = FromPhantomSpace(pieces);
                }

                md.TryAdd(CanonicalKeys.TokenizerLowercase, true);
                break;
        }

        return md;
    }

    /// <summary>
    /// llama.cpp stores WordPiece vocabularies in "phantom space" form: word-initial pieces get a ▁ prefix, continuation pieces lose
    /// their ## and bracketed specials stay as they are. This restores the canonical (Hugging Face) form.
    /// </summary>
    private static string[] FromPhantomSpace(string[] tokens)
    {
        var restored = new string[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            string token = tokens[i];
            bool isBracketedSpecial = token.Length > 1 && token[0] == '[' && token[^1] == ']';
            restored[i] = token.StartsWith('▁') ? token[1..] : isBracketedSpecial ? token : "##" + token;
        }

        return restored;
    }
}
