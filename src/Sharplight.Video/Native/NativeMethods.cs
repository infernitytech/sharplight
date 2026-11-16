using System.Runtime.InteropServices;

namespace Sharplight.Video.Native;

/// <summary>Source-generated bindings of <c>sharplight_video.h</c> (trim and AOT safe).</summary>
internal static unsafe partial class NativeMethods
{
    public const string Library = "sharplight_video";

    public const int AbiVersion = 1;

    [LibraryImport(Library)]
    public static partial int slv_abi_version();

    [LibraryImport(Library)]
    public static partial nint slv_version_string();

    [LibraryImport(Library)]
    public static partial nint slv_last_error();

    [LibraryImport(Library)]
    public static partial int slv_encoder_create(int codec, int width, int height, int fpsNum, int fpsDen, int bitrateKbps, int quality, int threads, int keyframeInterval, out NativeEncoderHandle encoder);

    [LibraryImport(Library)]
    public static partial int slv_encoder_encode(NativeEncoderHandle encoder, byte* y, int yStride, byte* u, int uStride, byte* v, int vStride, long pts, int forceKeyframe);

    [LibraryImport(Library)]
    public static partial int slv_encoder_flush(NativeEncoderHandle encoder);

    [LibraryImport(Library)]
    public static partial int slv_encoder_next_packet(NativeEncoderHandle encoder, out byte* data, out nuint size, out long pts, out int isKeyframe);

    [LibraryImport(Library)]
    public static partial void slv_encoder_destroy(nint encoder);
}
