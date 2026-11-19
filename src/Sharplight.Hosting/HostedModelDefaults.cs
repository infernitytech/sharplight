namespace Sharplight.Hosting;

/// <summary>
/// Settings that apply to every model of a host that does not set its own (<see cref="HostedModelOptions"/>). Bound from the
/// <c>defaults</c> object of the hosting JSON file or set in code.
/// </summary>
public sealed class HostedModelDefaults
{
    /// <summary>
    /// Where models run when they name no device: <c>auto</c> (default: the GPU with the most free memory, else the CPU), <c>cpu</c>,
    /// a GPU index (<c>0</c>), a device id (<c>vulkan:1</c>) or several of them separated by commas to split a model over them.
    /// </summary>
    public string Device { get; set; } = DeviceSpecifications.Auto;

    /// <summary>Context length per sequence, in tokens.</summary>
    public int ContextLength { get; set; } = 4096;

    /// <summary>Maximum rows of one forward pass; 0 = chosen from the model's footprint and the free memory (chat models; 512 for the others).</summary>
    public int BatchTokens { get; set; }

    /// <summary>Concurrent sequences of a chat model (batched per forward pass).</summary>
    public int Slots { get; set; } = 4;

    /// <summary>Paged KV cache: one pool of this many tokens shared by the slots; null reserves the full context per slot.</summary>
    public long? KvCacheTokens { get; set; }

    /// <summary>Element type of the KV cache (<c>f16</c>, <c>f32</c>, <c>q8_0</c>, <c>q4_0</c>); null = the model's default.</summary>
    public string? KvCacheType { get; set; }

    /// <summary>Run layers that do not fit on the GPU, and unsupported ops, on the CPU.</summary>
    public bool CpuOffload { get; set; }

    /// <summary>Allow faster kernels with reduced (f16) precision where the default keeps near-f32 accuracy.</summary>
    public bool ReducedPrecision { get; set; }

    /// <summary>Waiting requests per chat model before new ones are refused (HTTP 429).</summary>
    public int MaxQueue { get; set; } = 64;

    /// <summary>Prompt rows per step and sequence, interleaved with decoding; 0 = the step's rows.</summary>
    public int PrefillChunk { get; set; }

    /// <summary>Reuse cached KV of earlier prompts with the same prefix (prefix caching; <see cref="Llm.Serving.EngineOptions.PrefixCache"/>).</summary>
    public bool PrefixCache { get; set; } = true;

    /// <summary>Recurrent-state snapshots for prefix caching of models with recurrent state (null: the default; 0: none).</summary>
    public int? KvStateSnapshots { get; set; }

    /// <summary>Token budget of one engine step; null = the model's batch size.</summary>
    public int? StepTokens { get; set; }

    /// <summary>
    /// Smallest token budget of the load-adaptive engine step (<see cref="Llm.Serving.EngineOptions.MinBatchTokens"/>); null = the
    /// engine's default, 0 = every step uses the whole budget.
    /// </summary>
    public int? MinStepTokens { get; set; }

    /// <summary>Draft length of prompt-lookup speculation; 0 disables it.</summary>
    public int NgramDraftTokens { get; set; }

    /// <summary>Draft length of multi-token-prediction speculation (models with an MTP head); 0 disables it.</summary>
    public int MtpDraftTokens { get; set; }

    /// <summary>Speculate only while at most this many sequences decode.</summary>
    public int SpeculationMaxBatch { get; set; } = 16;

    /// <summary>Vision projector file for chat models that have none of their own.</summary>
    public string? Projector { get; set; }

    /// <summary>Jinja chat template file that replaces the models' own.</summary>
    public string? ChatTemplate { get; set; }
}
