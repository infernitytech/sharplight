using System.Runtime.InteropServices;

namespace Sharplight.Audio.Native;

/// <summary>Mirror of <c>sla_stream_stats</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeStreamStats
{
    public ulong FramesTransferred;
    public ulong Underruns;
    public ulong Overruns;
}
