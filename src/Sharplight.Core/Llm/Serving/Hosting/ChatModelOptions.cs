using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>How a chat model is served: batching slots, engine and chat-service settings.</summary>
public sealed record ChatModelOptions
{
    /// <summary>Concurrent sequences batched per forward pass, each with the load option's context length.</summary>
    public int Slots { get; set; } = 4;
    /// <summary>
    /// Paged KV cache shared by the slots, in tokens. With paging, admission is bounded by free cache blocks rather than slots, so
    /// <see cref="Slots"/> can exceed what fixed per-slot contexts would fit, and prompts share cached blocks by content (prefix
    /// caching). Null (the default): paged with every slot's full context (<see cref="LlmModel.DefaultKvCacheTokens"/>) where the
    /// devices run the paged kernels, shrunk to fit when the model host has to load the model into less memory than it needs;
    /// 0: the contiguous per-slot cache.
    /// </summary>
    public long? KvCacheTokens { get; set; }
    /// <summary>
    /// Recurrent-state snapshots of a paged cache for prefix caching of models with recurrent state (null: <see cref="LlmModel.DefaultStateSnapshots"/>,
    /// 0: none). Each costs one slot's recurrent state.
    /// </summary>
    public int? KvStateSnapshots { get; set; }
    /// <summary>
    /// Serve image input through the engine for a model with a vision tower (default on): its decoder is compiled from the multimodal graph
    /// (<see cref="LlmModel.CreateBatchDecoder"/> with media input) and its vision encoder is loaded with it (both counted in the model's
    /// footprint), so image requests are batched with text requests and share the prefix cache. Off: text only (image parts are rejected).
    /// </summary>
    public bool ImageInput { get; set; } = true;
    /// <summary>Scheduler settings of the model's engine.</summary>
    public EngineOptions Engine { get; set; } = new();
    /// <summary>Name reported by the API is the registration name; the other settings (chat template override, tool formats, default max tokens) apply.</summary>
    public ModelServiceOptions Service { get; set; } = new();
    /// <summary>
    /// Creates the speculator of each loaded instance (speculative decoding), or null for none. A factory rather than an
    /// instance because the engine owns and disposes its speculator, and a model may be loaded again after eviction.
    /// The decoder is then sized for the verification rows (<see cref="ISpeculator.MaxDraftTokens"/> + 1 per speculating
    /// sequence), and models with recurrent state get their recurrent ops compiled for rollback of rejected drafts.
    /// </summary>
    public Func<LlmModel, ISpeculator?>? Speculator { get; set; }
    /// <summary>
    /// Multi-token-prediction speculation: drafts per step from the model's own MTP head (<see cref="LlmModel.CreateMtpSpeculator"/>),
    /// 0 for none. Takes precedence over <see cref="Speculator"/> for models that have a draft head; others fall back to it.
    /// </summary>
    public int MtpDraftTokens { get; set; }
}
