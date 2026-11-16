using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// One adapted module of a LoRA file: the component it belongs to (<c>unet</c> / <c>transformer</c>, <c>text_encoder</c>,
/// <c>text_encoder_2</c>), the module name as stored after the format's prefixes (dotted, or kohya's underscored), the down and up
/// matrices and the alpha (the delta is up · down · alpha / rank).
/// </summary>
/// <param name="Component">The component (<see cref="LoraCheckpoint.Denoiser"/>, <see cref="LoraCheckpoint.TextEncoder"/>, <see cref="LoraCheckpoint.SecondTextEncoder"/>).</param>
/// <param name="Name">The module name within the component.</param>
/// <param name="Kohya">Whether the name is kohya's (dots replaced by underscores).</param>
/// <param name="Down">The down matrix [rank, in (, kh, kw)].</param>
/// <param name="Up">The up matrix [out, rank (, 1, 1)].</param>
/// <param name="Alpha">The alpha, or null (= rank).</param>
internal sealed record LoraModule(string Component, string Name, bool Kohya, WeightInfo Down, WeightInfo Up, float? Alpha)
{
    /// <summary>The rank.</summary>
    public long Rank => Down.Shape[0].Fixed;

    /// <summary>The factor of up · down: alpha / rank (1 without alpha).</summary>
    public float Factor => Alpha is { } alpha ? alpha / Rank : 1f;
}
