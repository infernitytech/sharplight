namespace Sharplight.Distributed.Transport;

/// <summary>
/// A persistent, ordered, reliable stream of frames between two processes (a coordinator and a segment worker, or two neighbouring
/// workers). One thread may send while another receives; concurrent sends are serialized.
/// </summary>
public interface ISegmentConnection : IDisposable
{
    /// <summary>The other end, for messages.</summary>
    public string RemoteName { get; }

    /// <summary>Sends one frame (the payload is copied before the call returns).</summary>
    /// <exception cref="IOException">The connection is closed or broken.</exception>
    public void Send(FrameKind kind, uint requestId, ReadOnlySpan<byte> payload);

    /// <summary>Waits for the next frame.</summary>
    /// <exception cref="IOException">The connection was closed or broke, or the frame is malformed.</exception>
    public Frame Receive();
}
