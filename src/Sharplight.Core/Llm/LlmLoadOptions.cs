using Sharplight.Models;
using Sharplight.Tasks;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Llm;

/// <summary>
/// Options for loading and compiling a language model with <see cref="LlmModel.Load(ModelLoader, string, LlmLoadOptions)"/>: the
/// settings every model kind shares (<see cref="TaskLoadOptions"/>; <see cref="TaskLoadOptions.ContextLength"/> defaults to 8192 here
/// and <see cref="TaskLoadOptions.MaxBatchTokens"/> to 0, the automatic choice of <see cref="LlmModel.BatchTokens"/>) plus the
/// text-generation ones. Settable properties, so it binds with Microsoft.Extensions.Options.
/// </summary>
public sealed class LlmLoadOptions : TaskLoadOptions
{
    /// <summary>Creates options with the text-generation defaults (8192-token context, automatic rows per forward pass).</summary>
    public LlmLoadOptions()
    {
        ContextLength = 8192;
        MaxBatchTokens = 0;
    }

    /// <summary>
    /// Element type of the attention KV caches, overriding <see cref="AdapterOptions.KvCacheType"/> of <see cref="TaskLoadOptions.Adapter"/>: f16,
    /// f32, or the block-quantized q8_0 (about half the memory of f16) / q4_0 (about a quarter). Caches whose rows per head are not
    /// whole 32-value blocks stay f16. Null keeps the adapter setting (default f16).
    /// </summary>
    public DataType? KvCacheType { get; set; }

    /// <summary>
    /// Element type of 16-bit-capable recurrent states (the gated delta rule's state matrices), overriding
    /// <see cref="AdapterOptions.RecurrentStateType"/> of <see cref="TaskLoadOptions.Adapter"/>: f32, bf16 or f16. Null: bf16 when
    /// <see cref="TaskLoadOptions.AllowReducedPrecision"/> is set, otherwise the adapter setting (default f32). The environment variable
    /// <c>SHARPLIGHT_RECURRENT_STATE</c> (f32, bf16, f16) overrides both. Has no effect on prebuilt packages.
    /// </summary>
    public DataType? RecurrentStateType { get; set; }

    /// <summary>
    /// Image token budget (output tokens per image): the vision encoder is compiled for at most
    /// <c>MaxTokens</c> (default 1024) tokens per image and images are resized to fit; <c>MinTokens</c> upsizes small images.
    /// </summary>
    public ImageTokenLimits? ImageTokens { get; set; }

    /// <summary>
    /// Choose tokens on the accelerator that computes the logits (default true): greedy and temperature / top-k / top-p / min-p sampling
    /// then download only token ids instead of the full logits. Sampling features the device path cannot express (repetition
    /// penalties, logit bias, alternatives for logprobs) still run on the CPU. Also disabled by <c>SHARPLIGHT_DEVICE_SAMPLING=0</c>.
    /// </summary>
    public bool DeviceSampling { get; set; } = true;

    /// <summary>
    /// Cancels loading: checked between load stages and during compilation (between segments, kernels and weight uploads), which
    /// then throws <see cref="OperationCanceledException"/> after releasing what it allocated. It applies to every compilation of
    /// the model, so cancel it only to abandon the model.
    /// </summary>
    public CancellationToken CancellationToken { get; set; }
}
