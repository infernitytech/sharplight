using System.Collections.Concurrent;

namespace Sharplight.Models;

/// <summary>
/// Reads a model container format (GGUF, Hugging Face safetensors checkpoints, ONNX, ...) and
/// translates it into the canonical <see cref="ModelSource"/> description defined by
/// <see cref="CanonicalKeys"/>. Importers live in their own assemblies and only reference Core;
/// nothing else in Sharplight depends on a particular importer.
/// </summary>
/// <remarks>
/// The canonical description fixes: architecture ids, hyper-parameter keys, tokenizer keys,
/// weight names and weight <em>semantics</em> (layout, sign conventions, folded constants).
/// An importer is responsible for every translation needed to reach it, typically with
/// lazily-derived weights (<see cref="WeightInfo.Derived"/>).
/// </remarks>
public interface IModelImporter
{
    /// <summary>Short format id, e.g. "gguf", "hf".</summary>
    string Format { get; }

    /// <summary>Cheap check (extension, magic bytes, directory layout) whether this importer handles <paramref name="path"/>.</summary>
    bool CanOpen(string path);

    /// <summary>Opens <paramref name="path"/> (a file or a directory) as a canonical model source.</summary>
    ModelSource Open(string path);
}

/// <summary>Global importer registry. Hosts register the formats they want to support.</summary>
public static class ModelImporterRegistry
{
    static readonly ConcurrentDictionary<string, IModelImporter> s_importers = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IModelImporter importer) => s_importers[importer.Format] = importer;

    public static IEnumerable<IModelImporter> All => s_importers.Values;

    public static IModelImporter? Find(string path) => s_importers.Values.FirstOrDefault(i => i.CanOpen(path));

    public static ModelSource Open(string path)
        => (Find(path) ?? throw new NotSupportedException(
               $"No registered importer can open '{path}'. Registered formats: {(s_importers.IsEmpty ? "(none)" : string.Join(", ", s_importers.Keys))}"))
           .Open(path);

    /// <summary>Opens a primary model plus auxiliary sources (e.g. ("mmproj", path) for a vision projector).</summary>
    public static ModelSourceSet OpenSet(string path, params (string Role, string Path)[] auxiliary)
    {
        var primary = Open(path);
        var aux = new Dictionary<string, ModelSource>();
        try
        {
            foreach (var (role, p) in auxiliary) aux[role] = Open(p);
        }
        catch
        {
            foreach (var r in primary.OwnedResources) r.Dispose();
            foreach (var s in aux.Values) foreach (var r in s.OwnedResources) r.Dispose();
            throw;
        }
        return new ModelSourceSet { Primary = primary, Auxiliary = aux };
    }
}

/// <summary>
/// Canonical metadata vocabulary of a <see cref="ModelSource"/>. Importers translate their format
/// into these keys; adapters, tokenizers and chat formatting read only these.
/// </summary>
/// <remarks>
/// This vocabulary is Sharplight's own and independent of any file format. Hyper-parameters are stored as
/// <c>{architecture}.{key}</c> (see <see cref="ModelSource.Arch{T}(string)"/>) with descriptive keys shared across
/// architectures where the concept is the same: <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>,
/// <c>context_length</c>, <c>norm_eps</c>, <c>attention.head_count</c>, <c>attention.kv_head_count</c>,
/// <c>attention.head_dim</c>, <c>rope.dim</c>, <c>rope.base</c>, <c>rope.sections</c>, <c>mtp.layer_count</c>, ...
/// Weight names are hierarchical: <c>token_embedding.weight</c>, <c>layers.{i}.input_norm.weight</c>,
/// <c>layers.{i}.attention.q.weight</c>, <c>layers.{i}.mlp.down.weight</c>, <c>output_norm.weight</c>,
/// <c>lm_head.weight</c>, ... Each architecture adapter documents the exact keys, names and weight semantics
/// it expects; importers (GGUF, Hugging Face, ...) translate into them.
/// </remarks>
public static class CanonicalKeys
{
    public const string Name = "general.name";
    public const string Architecture = "general.architecture";

    public const string TokenizerModel = "tokenizer.model";
    public const string TokenizerPre = "tokenizer.pre";
    public const string TokenizerTokens = "tokenizer.tokens";
    public const string TokenizerTokenTypes = "tokenizer.token_type";
    public const string TokenizerMerges = "tokenizer.merges";
    public const string TokenizerBosId = "tokenizer.bos_token_id";
    public const string TokenizerEosId = "tokenizer.eos_token_id";
    public const string TokenizerEotId = "tokenizer.eot_token_id";
    public const string TokenizerEomId = "tokenizer.eom_token_id";
    public const string TokenizerPadId = "tokenizer.padding_token_id";
    public const string TokenizerUnkId = "tokenizer.unknown_token_id";
    public const string TokenizerAddBos = "tokenizer.add_bos_token";
    public const string TokenizerAddEos = "tokenizer.add_eos_token";
    public const string ChatTemplate = "tokenizer.chat_template";
}
