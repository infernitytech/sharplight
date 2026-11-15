using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Audio;

/// <summary>
/// Reads and writes RIFF WAVE files: integer PCM (8, 16, 24, 32 bit) and IEEE float (32, 64 bit), plain or
/// <c>WAVE_FORMAT_EXTENSIBLE</c>, any channel count. Other encodings (A-law, ADPCM, compressed containers) are the host's job.
/// </summary>
public static class WavFile
{
    private const ushort FormatPcm = 1;
    private const ushort FormatFloat = 3;
    private const ushort FormatExtensible = 0xFFFE;

    /// <summary>Reads a WAVE file.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not a WAVE file or uses an unsupported encoding.</exception>
    public static AudioBuffer Read(string path)
    {
        using var stream = File.OpenRead(path);
        return Read(stream);
    }

    /// <summary>Reads a WAVE stream from its current position.</summary>
    /// <param name="stream">The stream; read up to the end of the data chunk, not disposed.</param>
    /// <exception cref="InvalidDataException">The stream is not a WAVE file or uses an unsupported encoding.</exception>
    public static AudioBuffer Read(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        if (ReadTag(reader) != "RIFF")
        {
            throw new InvalidDataException("Not a RIFF file.");
        }

        reader.ReadUInt32();
        if (ReadTag(reader) != "WAVE")
        {
            throw new InvalidDataException("Not a WAVE file.");
        }

        ushort format = 0;
        int channels = 0;
        int sampleRate = 0;
        int bits = 0;
        while (true)
        {
            string tag;
            uint size;
            try
            {
                tag = ReadTag(reader);
                size = reader.ReadUInt32();
            }
            catch (EndOfStreamException)
            {
                throw new InvalidDataException("WAVE file has no data chunk.");
            }

            if (tag == "fmt ")
            {
                var chunk = reader.ReadBytes(checked((int)size));
                format = BinaryPrimitives.ReadUInt16LittleEndian(chunk);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(chunk.AsSpan(2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(chunk.AsSpan(4));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(chunk.AsSpan(14));
                if (format == FormatExtensible && chunk.Length >= 26)
                {
                    // The sub-format GUID starts with the plain format tag.
                    format = BinaryPrimitives.ReadUInt16LittleEndian(chunk.AsSpan(24));
                }
            }
            else if (tag == "data")
            {
                if (channels == 0)
                {
                    throw new InvalidDataException("WAVE data chunk before the fmt chunk.");
                }

                var data = reader.ReadBytes(checked((int)size));
                return new AudioBuffer(Decode(data, format, bits, channels), sampleRate, channels);
            }
            else
            {
                Skip(reader, size);
            }

            if ((size & 1) != 0)
            {
                Skip(reader, 1);
            }
        }
    }

    /// <summary>Writes a WAVE file.</summary>
    /// <param name="path">The file (created or overwritten).</param>
    /// <param name="audio">The audio.</param>
    /// <param name="format">The sample encoding; integer formats clip samples to [-1, 1].</param>
    public static void Write(string path, AudioBuffer audio, WavSampleFormat format = WavSampleFormat.Pcm16)
    {
        using var stream = File.Create(path);
        Write(stream, audio, format);
    }

    /// <summary>Writes a WAVE stream at the current position.</summary>
    /// <param name="stream">The stream; not disposed.</param>
    /// <param name="audio">The audio.</param>
    /// <param name="format">The sample encoding; integer formats clip samples to [-1, 1].</param>
    public static void Write(Stream stream, AudioBuffer audio, WavSampleFormat format = WavSampleFormat.Pcm16)
    {
        int bytesPerSample = format switch
        {
            WavSampleFormat.Pcm16 => 2,
            WavSampleFormat.Pcm24 => 3,
            _ => 4,
        };
        int blockAlign = bytesPerSample * audio.Channels;
        int dataSize = checked(audio.Samples.Length * bytesPerSample);
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        writer.Write("RIFF"u8);
        writer.Write(36 + dataSize);
        writer.Write("WAVE"u8);
        writer.Write("fmt "u8);
        writer.Write(16);
        writer.Write(format == WavSampleFormat.Float32 ? FormatFloat : FormatPcm);
        writer.Write((ushort)audio.Channels);
        writer.Write(audio.SampleRate);
        writer.Write(audio.SampleRate * blockAlign);
        writer.Write((ushort)blockAlign);
        writer.Write((ushort)(bytesPerSample * 8));
        writer.Write("data"u8);
        writer.Write(dataSize);
        var data = new byte[dataSize];
        Encode(audio.Samples, format, data);
        writer.Write(data);
        if ((dataSize & 1) != 0)
        {
            writer.Write((byte)0);
        }
    }

    private static float[] Decode(byte[] data, ushort format, int bits, int channels)
    {
        int bytesPerSample = bits / 8;
        if (bytesPerSample == 0 || bits % 8 != 0)
        {
            throw new InvalidDataException($"Unsupported WAVE sample width {bits} bits.");
        }

        int count = data.Length / bytesPerSample / channels * channels;
        var samples = new float[count];
        var span = data.AsSpan();
        for (int i = 0; i < count; i++)
        {
            var bytes = span.Slice(i * bytesPerSample, bytesPerSample);
            samples[i] = (format, bits) switch
            {
                (FormatPcm, 8) => (bytes[0] - 128) / 128f,
                (FormatPcm, 16) => BinaryPrimitives.ReadInt16LittleEndian(bytes) / 32768f,
                (FormatPcm, 24) => (((bytes[2] << 24) | (bytes[1] << 16) | (bytes[0] << 8)) >> 8) / 8388608f,
                (FormatPcm, 32) => BinaryPrimitives.ReadInt32LittleEndian(bytes) / 2147483648f,
                (FormatFloat, 32) => BinaryPrimitives.ReadSingleLittleEndian(bytes),
                (FormatFloat, 64) => (float)BinaryPrimitives.ReadDoubleLittleEndian(bytes),
                _ => throw new InvalidDataException($"Unsupported WAVE encoding (format {format}, {bits} bits)."),
            };
        }

        return samples;
    }

    private static void Encode(float[] samples, WavSampleFormat format, Span<byte> data)
    {
        for (int i = 0; i < samples.Length; i++)
        {
            float clipped = Math.Clamp(samples[i], -1f, 1f);
            switch (format)
            {
                case WavSampleFormat.Pcm16:
                    BinaryPrimitives.WriteInt16LittleEndian(data[(i * 2)..], (short)Quantize(clipped, 32768.0));
                    break;
                case WavSampleFormat.Pcm24:
                    WritePcm24(data[(i * 3)..], (int)Quantize(clipped, 8388608.0));
                    break;
                case WavSampleFormat.Pcm32:
                    BinaryPrimitives.WriteInt32LittleEndian(data[(i * 4)..], (int)Quantize(clipped, 2147483648.0));
                    break;
                default:
                    BinaryPrimitives.WriteSingleLittleEndian(data[(i * 4)..], samples[i]);
                    break;
            }
        }
    }

    /// <summary>
    /// Scales a sample in [-1, 1] by <paramref name="scale"/> (2^(bits-1), the divisor reading uses, so a round trip only adds the
    /// rounding error) and clamps to the integer range, where +1 maps to the largest positive value.
    /// </summary>
    private static long Quantize(float sample, double scale) => (long)Math.Clamp(Math.Round(sample * scale), -scale, scale - 1);

    private static void WritePcm24(Span<byte> destination, int value)
    {
        destination[0] = (byte)value;
        destination[1] = (byte)(value >> 8);
        destination[2] = (byte)(value >> 16);
    }

    private static string ReadTag(BinaryReader reader)
    {
        var bytes = reader.ReadBytes(4);
        if (bytes.Length < 4)
        {
            throw new EndOfStreamException();
        }

        return Encoding.ASCII.GetString(bytes);
    }

    private static void Skip(BinaryReader reader, uint count)
    {
        if (reader.BaseStream.CanSeek)
        {
            reader.BaseStream.Seek(count, SeekOrigin.Current);
            return;
        }

        reader.ReadBytes(checked((int)count));
    }
}
