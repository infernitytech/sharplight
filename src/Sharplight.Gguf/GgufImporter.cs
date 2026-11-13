using Sharplight.Models;

namespace Sharplight.Gguf;

/// <summary>
/// GGUF importer: maps GGUF files (llama.cpp's format) onto the canonical <see cref="ModelSource"/>.
/// llama.cpp's architecture ids, hyper-parameter keys and tensor names are translated by an
/// <see cref="IGgufArchitectureMapping"/> per architecture; tokenizer keys (<c>tokenizer.ggml.*</c>) map to
/// <see cref="CanonicalKeys"/>. Weight data is shared zero-copy (only names change).
/// </summary>
public sealed class GgufImporter : IModelImporter
{
    public string Format => "gguf";

    /// <summary>Registers the GGUF importer with <see cref="ModelImporterRegistry"/>.</summary>
    public static void Register() => ModelImporterRegistry.Register(new GgufImporter());

    public bool CanOpen(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var f = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[4];
            return f.Read(magic) == 4 && magic.SequenceEqual("GGUF"u8);
        }
        catch (IOException) { return false; }
    }

    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a GGUF model (single file or first shard of a split model).</summary>
    public static ModelSource Open(string path)
    {
        var files = GgufModelFiles.Open(path);
        try
        {
            var ggufArch = files.Architecture ?? throw new InvalidDataException($"{path}: missing general.architecture");
            var mapping = GgufArchitectureMappings.Find(ggufArch) ?? throw new NotSupportedException(
                $"{path}: GGUF architecture '{ggufArch}' has no canonical mapping. Supported: {string.Join(", ", GgufArchitectureMappings.All.SelectMany(m => m.GgufArchitectures))}");
            var view = new GgufMetadataView(files.Metadata, ggufArch);

            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var unmapped = new List<string>();
            foreach (var t in files.Tensors)
            {
                if (mapping is IGgufSplitMapping splitter && splitter.Split(t.Name, t.ToWeightInfo(), view) is { } parts)
                {
                    foreach (var part in parts)
                    {
                        if (!weights.TryAdd(part.Name, mapping.MapWeight(part, view)))
                        {
                            throw new InvalidDataException($"{path}: two tensors map to '{part.Name}'.");
                        }
                    }

                    continue;
                }
                if (mapping.MapWeightName(t.Name, view) is not { } name) { unmapped.Add(t.Name); continue; }
                if (!weights.TryAdd(name, mapping.MapWeight(t.ToWeightInfo().As(name), view)))
                {
                    throw new InvalidDataException($"{path}: two tensors map to '{name}'.");
                }
            }
            foreach (var w in mapping.SyntheticWeights(view, weights).ToList())
            {
                if (!weights.TryAdd(w.Name, w))
                {
                    throw new InvalidDataException($"{path}: synthetic weight '{w.Name}' collides with a tensor.");
                }
            }

            if (unmapped.Count > 0)
            {
                throw new NotSupportedException($"{path}: {unmapped.Count} tensor(s) of '{ggufArch}' have no canonical mapping: {string.Join(", ", unmapped.Take(10))}{(unmapped.Count > 10 ? ", ..." : "")}");
            }

            if (PrismHadamard.Applies(files.Metadata))
            {
                PrismHadamard.Apply(weights, files.Metadata, n => mapping.MapWeightName(n, view));
            }

            if (mapping is IGgufWeightTransformer transformer)
            {
                transformer.Transform(weights, view);
            }

            string architecture = mapping.ResolveArchitecture(view);

            var metadata = new Dictionary<string, object>(ToCanonical(files.Metadata), StringComparer.Ordinal);
            // Drop llama.cpp's per-architecture keys; the mapping provides the canonical ones.
            foreach (var k in metadata.Keys.Where(k => k.StartsWith(ggufArch + ".", StringComparison.Ordinal)).ToList())
            {
                metadata.Remove(k);
            }

            foreach (var (k, v) in mapping.MapHyperParameters(view))
            {
                metadata[k] = v;
            }

            metadata[CanonicalKeys.Architecture] = architecture;
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

    private const string GgmlTokenizerPrefix = "tokenizer.ggml.";

    /// <summary>Translates GGUF metadata keys to <see cref="CanonicalKeys"/>; other keys pass through.</summary>
    public static IReadOnlyDictionary<string, object> ToCanonical(IReadOnlyDictionary<string, object> gguf)
    {
        var md = new Dictionary<string, object>(gguf.Count, StringComparer.Ordinal);
        foreach (var (k, v) in gguf)
        {
            md[k.StartsWith(GgmlTokenizerPrefix, StringComparison.Ordinal) ? "tokenizer." + k[GgmlTokenizerPrefix.Length..] : k] = v;
        }
        // llama.cpp's spelling and names of the pair / classifier tokens.
        if (md.Remove("tokenizer.seperator_token_id", out var sep))
        {
            md[CanonicalKeys.TokenizerSeparatorId] = sep;
        }

        if (md.Remove("tokenizer.cls_token_id", out var cls))
        {
            md[CanonicalKeys.TokenizerClassifierId] = cls;
        }

        md.TryGetValue(CanonicalKeys.TokenizerModel, out var tm);
        switch (tm)
        {
            // llama.cpp calls SentencePiece-style tokenizers "llama"; the canonical name is "spm".
            case "llama": md[CanonicalKeys.TokenizerModel] = "spm"; break;
            // llama.cpp's "gemma4" tokenizer is a character-level, merge-ranked BPE with ▁ spaces and byte fallback: canonical "bpe".
            case "gemma4": md[CanonicalKeys.TokenizerModel] = "bpe"; break;
            // SentencePiece unigram models (T5, XLM-RoBERTa) are "t5" in llama.cpp.
            case "t5": md[CanonicalKeys.TokenizerModel] = "unigram"; break;
            case "bert":
                md[CanonicalKeys.TokenizerModel] = "wordpiece";
                if (md.TryGetValue(CanonicalKeys.TokenizerTokens, out var t) && t is string[] tokens)
                {
                    md[CanonicalKeys.TokenizerTokens] = FromPhantomSpace(tokens);
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
        var r = new string[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            string s = tokens[i];
            r[i] = s.StartsWith('▁') ? s[1..] : s.Length > 1 && s[0] == '[' && s[^1] == ']' ? s : "##" + s;
        }
        return r;
    }
}
