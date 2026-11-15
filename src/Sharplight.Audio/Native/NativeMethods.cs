using System.Runtime.InteropServices;

namespace Sharplight.Audio.Native;

/// <summary>Source-generated bindings of <c>sharplight_audio.h</c> (blittable, trim and AOT safe). Pointers are raw handles wrapped by the SafeHandles.</summary>
internal static unsafe partial class NativeMethods
{
    public const string Library = "sharplight_audio";

    public const int AbiVersion = 1;

    [LibraryImport(Library)]
    public static partial int sla_abi_version();

    [LibraryImport(Library)]
    public static partial int sla_version();

    [LibraryImport(Library)]
    public static partial nint sla_version_string();

    [LibraryImport(Library)]
    public static partial int sla_context_create(int useNullBackend, out nint context);

    [LibraryImport(Library)]
    public static partial void sla_context_destroy(nint context);

    [LibraryImport(Library)]
    public static partial nint sla_context_backend_name(nint context);

    [LibraryImport(Library)]
    public static partial int sla_device_count(nint context, int kind, out int count);

    [LibraryImport(Library)]
    public static partial int sla_device_info(nint context, int kind, int index, byte* nameBuffer, int nameCapacity, out int isDefault);

    [LibraryImport(Library)]
    public static partial int sla_stream_open(nint context, int kind, int deviceIndex, uint sampleRate, uint channels, uint ringFrames, out nint stream);

    [LibraryImport(Library)]
    public static partial int sla_stream_start(nint stream);

    [LibraryImport(Library)]
    public static partial int sla_stream_stop(nint stream);

    [LibraryImport(Library)]
    public static partial void sla_stream_close(nint stream);

    [LibraryImport(Library)]
    public static partial int sla_stream_write(nint stream, float* frames, uint frameCount, out uint written);

    [LibraryImport(Library)]
    public static partial int sla_stream_read(nint stream, float* frames, uint frameCount, out uint read);

    [LibraryImport(Library)]
    public static partial int sla_stream_available(nint stream, out uint frames);

    [LibraryImport(Library)]
    public static partial int sla_stream_stats_get(nint stream, out NativeStreamStats stats);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int sla_decoder_open_file(string path, out nint decoder);

    [LibraryImport(Library)]
    public static partial int sla_decoder_open_memory(byte* data, nuint size, out nint decoder);

    [LibraryImport(Library)]
    public static partial int sla_decoder_info(nint decoder, out NativeAudioInfo info);

    [LibraryImport(Library)]
    public static partial int sla_decoder_read(nint decoder, float* frames, ulong frameCount, out ulong read);

    [LibraryImport(Library)]
    public static partial int sla_decoder_seek(nint decoder, ulong frameIndex);

    [LibraryImport(Library)]
    public static partial void sla_decoder_close(nint decoder);

    [LibraryImport(Library)]
    public static partial int sla_encoder_open(int format, uint sampleRate, uint channels, int bitrate, out nint encoder);

    [LibraryImport(Library)]
    public static partial int sla_encoder_write(nint encoder, float* frames, ulong frameCount);

    [LibraryImport(Library)]
    public static partial int sla_encoder_finish(nint encoder);

    [LibraryImport(Library)]
    public static partial int sla_encoder_data(nint encoder, out byte* data, out nuint size);

    [LibraryImport(Library, StringMarshalling = StringMarshalling.Utf8)]
    public static partial int sla_encoder_save(nint encoder, string path);

    [LibraryImport(Library)]
    public static partial void sla_encoder_close(nint encoder);
}
