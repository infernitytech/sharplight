namespace Sharplight.Distributed.Transport;

/// <summary>The message a frame carries (the segment protocol; see docs/multinode.md).</summary>
public enum FrameKind : byte
{
    /// <summary>First frame of a connection: protocol version and role.</summary>
    Hello = 1,
    /// <summary>Answer to <see cref="Hello"/>: the worker's devices, memory and model.</summary>
    HelloAck,
    /// <summary>Heartbeat; answered with <see cref="Pong"/>.</summary>
    Ping,
    /// <summary>Heartbeat answer.</summary>
    Pong,
    /// <summary>A request failed: the error message (and its kind).</summary>
    Error,
    /// <summary>Asks for the worker's current memory figures.</summary>
    MemoryRequest,
    /// <summary>The worker's current memory figures.</summary>
    MemoryReply,
    /// <summary>Asks whether the worker's devices run a node.</summary>
    SupportsRequest,
    /// <summary>Whether the worker's devices run the node, and why not.</summary>
    SupportsReply,
    /// <summary>Compiles a segment on the worker's devices.</summary>
    Compile,
    /// <summary>The compiled segment's handle and figures.</summary>
    CompileReply,
    /// <summary>Runs a compiled segment once: its inputs, symbols and scalars.</summary>
    Execute,
    /// <summary>The outputs of one execution and the symbols its ops produced.</summary>
    ExecuteReply,
    /// <summary>Prepares a compiled segment's work for a shape.</summary>
    Prepare,
    /// <summary>A request without a result succeeded.</summary>
    Ack,
    /// <summary>Clears a compiled segment's persistent state.</summary>
    ResetState,
    /// <summary>Copies rows of a compiled segment's persistent state.</summary>
    CopyStateRows,
    /// <summary>Releases a compiled segment and its memory.</summary>
    Release,
    /// <summary>Sends an output of a compiled segment to a neighbour after every execution.</summary>
    AddForward,
    /// <summary>Takes an input of a compiled segment from a neighbour at every execution.</summary>
    AddForwardedInput,
    /// <summary>The tag a neighbour sends a forwarded input under.</summary>
    ForwardedInputReply,
    /// <summary>A boundary value from a neighbour (worker to worker).</summary>
    PeerValue,
    /// <summary>Fails the executions in progress and the following ones of the connection's segments.</summary>
    Abort,
}
