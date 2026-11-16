namespace Sharplight.Models;

/// <summary>Whether a multi-component pipeline (a Diffusers repository's <c>model_index.json</c> class) can be loaded.</summary>
/// <param name="Supported">The importer can load the pipeline.</param>
/// <param name="Reason">Why not, for a pipeline that is not supported (names the pipeline family); empty when supported.</param>
public sealed record PipelineSupport(bool Supported, string Reason = "");
