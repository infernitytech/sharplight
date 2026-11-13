namespace Sharplight.Packaging;

/// <summary>What <see cref="ModelConverter"/> wrote.</summary>
/// <param name="PackagePath">The written package.</param>
/// <param name="ModelName">Name of the built model.</param>
/// <param name="Architecture">Graph layout of the built model (e.g. <c>decoder</c>).</param>
/// <param name="SourceFormat">Format of the primary source (e.g. <c>gguf</c>).</param>
/// <param name="GraphCount">Number of graphs stored.</param>
/// <param name="WeightCount">Number of weights stored.</param>
/// <param name="PackageBytes">Size of the written file in bytes.</param>
/// <param name="BuildTime">Time spent opening the source and building (and optimizing) the graphs.</param>
/// <param name="SaveTime">Time spent writing the package.</param>
/// <param name="KeptArtifactSets">Artifact sets of a source package stored with the result.</param>
/// <param name="DroppedArtifactSets">Artifact sets of a source package that were not copied (<see cref="ConversionOptions.KeepArtifacts"/> off).</param>
/// <param name="SourceArchitecture">Canonical architecture of the source (e.g. <c>qwen3</c>), or <see langword="null"/> when unknown.</param>
public sealed record ConversionResult(
    string PackagePath,
    string ModelName,
    string Architecture,
    string SourceFormat,
    int GraphCount,
    int WeightCount,
    long PackageBytes,
    TimeSpan BuildTime,
    TimeSpan SaveTime,
    int KeptArtifactSets,
    int DroppedArtifactSets,
    string? SourceArchitecture = null);
