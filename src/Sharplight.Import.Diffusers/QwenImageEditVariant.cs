namespace Sharplight.Import.Diffusers;

/// <summary>
/// An editing variant of the Qwen-Image transformer: the reference images go through the Qwen2.5-VL text encoder in the prompt and their VAE
/// latents follow the image's tokens. Qwen-Image-Edit-2509 and -2511 (and FireRed-Image-Edit) are diffusers' <c>QwenImageEditPlusPipeline</c>:
/// several images, each labelled "Picture n:", read by the language model at 384 · 384 pixels; -2511 also modulates the reference tokens
/// at timestep 0 (<c>zero_cond_t</c>). The first Qwen-Image-Edit is <c>QwenImageEditPipeline</c>: one image, read at 1024 · 1024 pixels.
/// </summary>
/// <param name="Plus">Whether the variant is the multi-image <c>QwenImageEditPlusPipeline</c>.</param>
/// <param name="ZeroConditionTimestep">Whether the reference tokens are modulated at timestep 0.</param>
internal sealed record QwenImageEditVariant(bool Plus, bool ZeroConditionTimestep)
{
    private const string Vision = "<|vision_start|><|image_pad|><|vision_end|>";

    /// <summary>Reference images the transformer is built for (Plus: up to three; the first Edit: one).</summary>
    public int MaxReferences => Plus ? 3 : 1;

    /// <summary>Rows of the text conditioning after the template's: the prompt's tokens and the images' (Plus: 3 images of 196 tokens; the first Edit: 1296).</summary>
    public int PromptLength => Plus ? 1024 : 1536;

    /// <summary>The area the language model reads each image at, in pixels.</summary>
    public int ConditionPixels => Plus ? 384 * 384 : 1024 * 1024;

    /// <summary>The prompt text of an image (<c>{n}</c>: its number).</summary>
    public string ImagePrompt => Plus ? "Picture {n}: " + Vision : Vision;

    /// <summary>The variant of a pipeline's class name (<c>QwenImageEditPipeline</c>, <c>QwenImageEditPlusPipeline</c>), or null for other pipelines.</summary>
    /// <param name="pipelineClass">The <c>_class_name</c> of the <c>model_index.json</c>.</param>
    /// <param name="zeroConditionTimestep">The transformer configuration's <c>zero_cond_t</c>.</param>
    public static QwenImageEditVariant? FromPipeline(string pipelineClass, bool zeroConditionTimestep)
        => pipelineClass switch
        {
            "QwenImageEditPlusPipeline" => new QwenImageEditVariant(true, zeroConditionTimestep),
            "QwenImageEditPipeline" => new QwenImageEditVariant(false, false),
            _ => null,
        };

    /// <summary>
    /// The variant of a single transformer file (checkpoint or GGUF), which carries no pipeline class: the zero-timestep marker tensor of
    /// -2511, else the file name ("edit"; "2509" or any later 25xx release is the multi-image pipeline, no year the first Edit); null for a
    /// text-to-image file.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="hasZeroConditionMarker">Whether the file holds <see cref="QwenImageNames.ZeroConditionMarker"/>.</param>
    public static QwenImageEditVariant? FromFile(string path, bool hasZeroConditionMarker)
    {
        string name = Path.GetFileNameWithoutExtension(path);
        if (hasZeroConditionMarker)
        {
            return new QwenImageEditVariant(true, true);
        }

        if (!name.Contains("edit", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return new QwenImageEditVariant(System.Text.RegularExpressions.Regex.IsMatch(name, @"(^|[^0-9])25[0-9]{2}([^0-9]|$)"), false);
    }
}
