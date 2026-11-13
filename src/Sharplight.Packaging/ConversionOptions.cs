using Sharplight.Tensors;

namespace Sharplight.Packaging;

/// <summary>How <see cref="ModelConverter"/> builds the model before it is written as a package.</summary>
public sealed class ConversionOptions
{
    /// <summary>Run the device-independent graph optimization before saving (default true), so loading the package skips it.</summary>
    public bool OptimizeGraphs { get; init; } = true;

    /// <summary>KV cache element type built into language-model graphs (default f16; a load can still override it).</summary>
    public DataType KvCacheType { get; init; } = DataType.F16;

    /// <summary>Context length language-model graphs are built for; 0 uses the model's own.</summary>
    public int ContextLength { get; init; }

    /// <summary>
    /// Store the precompiled artifact sets of a package source with the result (default true). Code entries are content addressed
    /// (a kernel the converted graphs no longer use is never asked for) and the driver validates pipeline caches, so they stay valid.
    /// </summary>
    public bool KeepArtifacts { get; init; } = true;

    /// <summary>Receives progress messages, or <see langword="null"/>.</summary>
    public Action<string>? Log { get; init; }
}
