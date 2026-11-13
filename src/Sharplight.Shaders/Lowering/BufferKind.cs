namespace Sharplight.Shaders.Lowering;

/// <summary>Where a buffer lives / how a backend should allocate it.</summary>
public enum BufferKind
{
    /// <summary>Model parameter; uploaded once, device local, read-only.</summary>
    Weight,
    /// <summary>Persistent per-sequence state; device local, survives executions.</summary>
    State,
    /// <summary>Transient activation arena; device local.</summary>
    Activations,
    /// <summary>Host-written input (tokens, positions, cross-device activations); host visible.</summary>
    Input,
    /// <summary>Host-read output; host visible (cached).</summary>
    Output,
    /// <summary>Runtime scalar block (uint[]), host visible, written before each execution.</summary>
    Runtime,
}
