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
    public string Format { get; }

    /// <summary>Cheap check (extension, magic bytes, directory layout) whether this importer handles <paramref name="path"/>.</summary>
    public bool CanOpen(string path);

    /// <summary>
    /// Rank among importers that accept the same path (<see cref="ModelImporterCatalog.TryFind"/> asks the highest first). Generic
    /// container importers keep the default 0; an importer for one kind of content inside a shared container (e.g. diffusion
    /// checkpoints stored as GGUF or safetensors) returns a higher value and checks the content in <see cref="CanOpen"/>.
    /// </summary>
    public int Priority => 0;

    /// <summary>Opens <paramref name="path"/> (a file or a directory) as a canonical model source.</summary>
    public ModelSource Open(string path);

    /// <summary>
    /// The companion files the model at <paramref name="path"/> needs and this importer finds next to it, by auxiliary role (e.g. the
    /// text encoders and the VAE of a transformer-only diffusion checkpoint). <see cref="ModelImporterCatalog.OpenSet"/> opens them for
    /// the roles the caller does not give; empty by default.
    /// </summary>
    /// <param name="path">The primary model file or directory.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string path) => new Dictionary<string, string>();

    /// <summary>
    /// The companion files the model at <paramref name="path"/> needs, in presentation order (e.g. the CLIP and T5 text encoders and the
    /// VAE of a transformer-only diffusion checkpoint); empty when it needs none (the default).
    /// </summary>
    /// <param name="path">The primary model file or directory.</param>
    public IReadOnlyList<CompanionFile> DescribeCompanions(string path) => [];
}
