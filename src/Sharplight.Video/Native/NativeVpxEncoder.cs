namespace Sharplight.Video.Native;

/// <summary>Managed wrapper of one native encoder (<c>slv_encoder</c>): feeds I420 frames and collects the packets.</summary>
internal sealed unsafe class NativeVpxEncoder : IDisposable
{
    private readonly NativeEncoderHandle _handle;

    /// <summary>Creates the encoder.</summary>
    /// <exception cref="PlatformNotSupportedException">The native library is unavailable.</exception>
    /// <exception cref="InvalidOperationException">libvpx rejected the configuration.</exception>
    public NativeVpxEncoder(WebMCodec codec, int width, int height, int framesPerSecondNumerator, int framesPerSecondDenominator, int bitrateKilobitsPerSecond, int quality, int threads)
    {
        NativeVideoLibrary.ThrowIfUnavailable();
        int result = NativeMethods.slv_encoder_create(
            codec == WebMCodec.Vp9 ? 1 : 0,
            width,
            height,
            framesPerSecondNumerator,
            framesPerSecondDenominator,
            bitrateKilobitsPerSecond,
            quality,
            threads,
            0,
            out _handle);
        ThrowIfFailed(result, "Creating the video encoder");
    }

    /// <summary>Submits one frame.</summary>
    public void Encode(byte[] y, byte[] u, byte[] v, int width, long pts)
    {
        int chromaStride = (width + 1) / 2;
        fixed (byte* yp = y)
        fixed (byte* up = u)
        fixed (byte* vp = v)
        {
            ThrowIfFailed(NativeMethods.slv_encoder_encode(_handle, yp, width, up, chromaStride, vp, chromaStride, pts, 0), "Encoding a video frame");
        }
    }

    /// <summary>Ends the stream so the remaining packets can be collected.</summary>
    public void Flush() => ThrowIfFailed(NativeMethods.slv_encoder_flush(_handle), "Finishing the video encode");

    /// <summary>Takes every packet that is ready.</summary>
    public void DrainPackets(List<VideoPacket> packets)
    {
        while (true)
        {
            int result = NativeMethods.slv_encoder_next_packet(_handle, out byte* data, out nuint size, out long pts, out int keyframe);
            if (result < 0)
            {
                ThrowIfFailed(-result, "Reading a video packet");
            }

            if (result == 0)
            {
                return;
            }

            packets.Add(new VideoPacket(new ReadOnlySpan<byte>(data, checked((int)size)).ToArray(), pts, keyframe != 0));
        }
    }

    /// <inheritdoc />
    public void Dispose() => _handle.Dispose();

    private static void ThrowIfFailed(int result, string operation)
    {
        if (result != 0)
        {
            throw new InvalidOperationException($"{operation} failed: {NativeVideoLibrary.LastError}");
        }
    }
}
