using Sharplight.Models;

namespace Sharplight.Graph;

public enum ValueKind
{
    /// <summary>Produced by a node inside the graph.</summary>
    Intermediate,
    /// <summary>Supplied by the caller for each execution (token ids, positions, embeddings, ...).</summary>
    Input,
    /// <summary>Model parameter, backed by a <see cref="WeightInfo"/>.</summary>
    Weight,
    /// <summary>Persistent per-sequence state (KV cache, recurrent state) that survives across executions.</summary>
    State,
    /// <summary>A small constant embedded in the graph.</summary>
    Constant,
}
