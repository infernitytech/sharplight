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

    /// <summary>Opens <paramref name="path"/> (a file or a directory) as a canonical model source.</summary>
    public ModelSource Open(string path);
}
