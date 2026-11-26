using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// One adapted module of a LoRA-family file: the component it belongs to (<c>unet</c> / <c>transformer</c>, <c>text_encoder</c>,
/// <c>text_encoder_2</c>), the module name as stored after the format's prefixes (dotted, or kohya's underscored), the weight difference
/// (plain LoRA, LoHa, LoKr, full), the alpha and, for DoRA, the magnitude the merged weight is renormalized to.
/// </summary>
/// <param name="Component">The component (<see cref="LoraCheckpoint.Denoiser"/>, <see cref="LoraCheckpoint.TextEncoder"/>, <see cref="LoraCheckpoint.SecondTextEncoder"/>).</param>
/// <param name="Name">The module name within the component.</param>
/// <param name="Kohya">Whether the name is kohya's (dots replaced by underscores).</param>
/// <param name="Delta">The weight difference.</param>
/// <param name="Alpha">The alpha, or null (= rank).</param>
/// <param name="Magnitude">DoRA's magnitude (PEFT's <c>lora_magnitude_vector</c> [out], LyCORIS' <c>dora_scale</c> [out, 1] or [1, in]), or null.</param>
internal sealed record LoraModule(string Component, string Name, bool Kohya, LoraDelta Delta, float? Alpha, WeightInfo? Magnitude = null)
{
    /// <summary>The factor of the difference: alpha / rank (1 without alpha, or for differences the alpha does not apply to).</summary>
    public float Factor => Alpha is { } alpha && Delta.Rank is { } rank ? alpha / rank : 1f;
}
