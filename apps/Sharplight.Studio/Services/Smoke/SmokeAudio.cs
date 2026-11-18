using System.Buffers.Binary;

namespace Sharplight.Studio.Services.Smoke;

/// <summary>The short WAV file the smoke run feeds the voice activity detector: silence, a second of a voice-like buzz, silence.</summary>
internal static class SmokeAudio
{
    /// <summary>The sample rate of the file.</summary>
    public const int SampleRate = 16000;

    /// <summary>Creates a mono 16-bit PCM WAV file (1.5 s): 0.25 s silence, 1 s of a 120 Hz harmonic tone with a syllable-rate amplitude, 0.25 s silence.</summary>
    public static byte[] CreateWav()
    {
        int total = SampleRate * 3 / 2;
        int start = SampleRate / 4;
        int end = start + SampleRate;
        var pcm = new short[total];
        var random = new Random(7);
        for (int i = start; i < end; i++)
        {
            double t = (double)i / SampleRate;
            double envelope = 0.5 + 0.5 * Math.Sin(2 * Math.PI * 4 * t);
            double value = 0;
            for (int harmonic = 1; harmonic <= 12; harmonic++)
            {
                value += Math.Sin(2 * Math.PI * 120 * harmonic * t) / harmonic;
            }

            value = (0.25 * envelope * value) + (0.01 * (random.NextDouble() - 0.5));
            pcm[i] = (short)Math.Clamp(value * short.MaxValue, short.MinValue, short.MaxValue);
        }

        var wav = new byte[44 + (pcm.Length * 2)];
        "RIFF"u8.CopyTo(wav);
        BinaryPrimitives.WriteInt32LittleEndian(wav.AsSpan(4), wav.Length - 8);
        "WAVEfmt "u8.CopyTo(wav.AsSpan(8));
        BinaryPrimitives.WriteInt32LittleEndian(wav.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(wav.AsSpan(20), 1);
        BinaryPrimitives.WriteInt16LittleEndian(wav.AsSpan(22), 1);
        BinaryPrimitives.WriteInt32LittleEndian(wav.AsSpan(24), SampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(wav.AsSpan(28), SampleRate * 2);
        BinaryPrimitives.WriteInt16LittleEndian(wav.AsSpan(32), 2);
        BinaryPrimitives.WriteInt16LittleEndian(wav.AsSpan(34), 16);
        "data"u8.CopyTo(wav.AsSpan(36));
        BinaryPrimitives.WriteInt32LittleEndian(wav.AsSpan(40), pcm.Length * 2);
        for (int i = 0; i < pcm.Length; i++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(wav.AsSpan(44 + (i * 2)), pcm[i]);
        }

        return wav;
    }
}
