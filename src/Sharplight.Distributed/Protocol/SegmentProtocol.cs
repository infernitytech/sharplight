namespace Sharplight.Distributed.Protocol;

/// <summary>Constants of the segment protocol.</summary>
internal static class SegmentProtocol
{
    /// <summary>"SLSW" in the first word of every <see cref="Transport.FrameKind.Hello"/>.</summary>
    public const uint Magic = 0x57534C53;

    /// <summary>Protocol version; both ends must speak the same one.</summary>
    public const int Version = 1;

    /// <summary><see cref="Transport.FrameKind.Hello"/> role: a coordinator that compiles and runs segments.</summary>
    public const byte CoordinatorRole = 1;

    /// <summary><see cref="Transport.FrameKind.Hello"/> role: a neighbouring worker that sends boundary values.</summary>
    public const byte PeerRole = 2;

    /// <summary>Error kind of an <see cref="Transport.FrameKind.Error"/>: anything else.</summary>
    public const byte GeneralError = 0;

    /// <summary>Error kind: a device ran out of memory.</summary>
    public const byte OutOfMemoryError = 1;

    /// <summary>Error kind: the worker's model copy differs from the coordinator's.</summary>
    public const byte ModelMismatchError = 2;
}
