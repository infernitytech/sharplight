using System.Text.Json.Serialization;

namespace Sharplight.Llm.Serving;

/// <summary>
/// How a paged KV cache admits waiting requests when its blocks run short (<see cref="EngineOptions.KvAdmission"/>). In JSON (hosting
/// files) the names, case-insensitive: <c>queue</c>, <c>preempt</c>. Without a paged cache, or while the pool has room for every
/// waiting prompt, both behave the same.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<KvAdmissionPolicy>))]
public enum KvAdmissionPolicy
{
    /// <summary>
    /// The default: a request is admitted only when its prompt and some room to grow (up to 64 tokens) fit next to what the running
    /// sequences still need (the rest of their prompts plus the same growth reserve). When the pool is full, new requests wait;
    /// running sequences are preempted only when they outgrow even that reserve. Fewer recomputed tokens, a higher time to first token
    /// at KV saturation (benchmarks.md §105.2).
    /// </summary>
    Queue,

    /// <summary>
    /// Admits a request as soon as its prompt fits next to the rest of the running prompts, without a growth reserve; when a running
    /// sequence then needs a block and none is free, the most recently admitted sequences give their blocks back and go to the head of
    /// the waiting work (no new request is admitted before they are back), recomputing prompt and generated tokens when blocks are
    /// free again (recompute preemption, as vLLM does). A lower time to first token at KV saturation, at the cost of the recomputation.
    /// </summary>
    Preempt,
}
