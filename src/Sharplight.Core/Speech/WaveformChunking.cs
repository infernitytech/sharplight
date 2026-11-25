using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Speech;

/// <summary>
/// Splits a long recording on the waveform before feature extraction (<see cref="SpeechKeys.ChunkSeconds"/>): chunks of at most the chunk
/// length, each cut at the start of the quietest RMS window in the last <see cref="SpeechKeys.ChunkSearchSeconds"/> before the limit (NeMo's
/// and Cohere Transcribe's energy split). Recordings up to the chunk length stay whole. Immutable.
/// </summary>
internal sealed class WaveformChunking
{
    private readonly int _chunkSamples;
    private readonly int _searchSamples;
    private readonly int _energyWindow;
    private readonly HashSet<string> _unspaced;

    private WaveformChunking(int chunkSamples, int searchSamples, int energyWindow, IEnumerable<string> unspaced)
    {
        _chunkSamples = Math.Max(1, chunkSamples);
        _searchSamples = Math.Max(1, searchSamples);
        _energyWindow = Math.Max(1, energyWindow);
        _unspaced = new HashSet<string>(unspaced, StringComparer.OrdinalIgnoreCase);
    }

    /// <summary>The chunking a model's metadata asks for, or null when it has no <see cref="SpeechKeys.ChunkSeconds"/>.</summary>
    /// <param name="metadata">The decoder's metadata.</param>
    /// <param name="sampleRate">The model's sample rate.</param>
    public static WaveformChunking? From(IReadOnlyDictionary<string, object> metadata, int sampleRate)
    {
        if (!metadata.TryGetValue(SpeechKeys.ChunkSeconds, out var chunk))
        {
            return null;
        }

        double seconds = ValueConverter.Convert<double>(chunk, "Metadata", SpeechKeys.ChunkSeconds);
        double search = metadata.TryGetValue(SpeechKeys.ChunkSearchSeconds, out var value) ? ValueConverter.Convert<double>(value, "Metadata", SpeechKeys.ChunkSearchSeconds) : 0;
        int window = TokenizerMetadataReader.GetInt(metadata, SpeechKeys.ChunkEnergyWindow) ?? 1600;
        string[] unspaced = TokenizerMetadataReader.GetStringArray(metadata, SpeechKeys.UnspacedLanguages) ?? [];
        return new WaveformChunking((int)Math.Round(seconds * sampleRate), (int)Math.Round(search * sampleRate), window, unspaced);
    }

    /// <summary>The text between two chunks of a transcription in <paramref name="language"/> (ISO 639-1, or null): a space or nothing.</summary>
    /// <param name="language">The transcription's language.</param>
    public string Separator(string? language) => language is not null && _unspaced.Contains(language) ? "" : " ";

    /// <summary>The chunks of <paramref name="samples"/> as [start, end) sample ranges, in order and without gaps.</summary>
    /// <param name="samples">The recording.</param>
    public IReadOnlyList<(int Start, int End)> Split(ReadOnlySpan<float> samples)
    {
        int total = samples.Length;
        if (total <= _chunkSamples)
        {
            return [(0, total)];
        }

        var chunks = new List<(int Start, int End)>();
        int start = 0;
        while (start < total)
        {
            if (start + _chunkSamples >= total)
            {
                chunks.Add((start, total));
                break;
            }

            int searchStart = Math.Max(start, start + _chunkSamples - _searchSamples);
            int searchEnd = Math.Min(start + _chunkSamples, total);
            int split = searchEnd <= searchStart ? start + _chunkSamples : QuietestPoint(samples, searchStart, searchEnd);
            split = Math.Max(start + 1, Math.Min(split, total));
            chunks.Add((start, split));
            start = split;
        }

        return chunks;
    }

    // The start of the window with the lowest RMS (the first of equal ones); the middle of a stretch shorter than a window.
    private int QuietestPoint(ReadOnlySpan<float> samples, int start, int end)
    {
        int length = end - start;
        if (length <= _energyWindow)
        {
            return (start + end) / 2;
        }

        double lowest = double.PositiveInfinity;
        int quietest = start;
        for (int offset = 0; offset < length - _energyWindow; offset += _energyWindow)
        {
            double sum = 0;
            foreach (float sample in samples.Slice(start + offset, _energyWindow))
            {
                sum += (double)sample * sample;
            }

            double energy = Math.Sqrt(sum / _energyWindow);
            if (energy < lowest)
            {
                lowest = energy;
                quietest = start + offset;
            }
        }

        return quietest;
    }
}
