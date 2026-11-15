using System.Runtime.InteropServices;

namespace Sharplight.Audio.Native;

/// <summary>Mirror of <c>sla_audio_info</c>.</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct NativeAudioInfo
{
    public uint SampleRate;
    public uint Channels;
    public long TotalFrames;
    public int Codec;
    public int Reserved;
}
