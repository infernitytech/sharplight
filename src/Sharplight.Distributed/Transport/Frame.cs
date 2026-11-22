namespace Sharplight.Distributed.Transport;

/// <summary>One message of the segment protocol.</summary>
/// <param name="Kind">What the message is.</param>
/// <param name="RequestId">Correlates a reply with its request (0 for messages without a reply).</param>
/// <param name="Payload">The message body (owned by the frame).</param>
public readonly record struct Frame(FrameKind Kind, uint RequestId, ReadOnlyMemory<byte> Payload);
