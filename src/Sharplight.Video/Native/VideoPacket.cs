namespace Sharplight.Video.Native;

/// <summary>A compressed packet returned by the native encoder.</summary>
/// <param name="Data">The packet bytes (copied out of native memory).</param>
/// <param name="Pts">Presentation time in frames.</param>
/// <param name="IsKeyframe">Whether the packet starts a decodable sequence.</param>
internal readonly record struct VideoPacket(byte[] Data, long Pts, bool IsKeyframe);
