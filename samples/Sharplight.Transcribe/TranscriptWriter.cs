using System.Globalization;
using System.Text.Json;
using Sharplight.Speech;

namespace Sharplight.Transcribe;

/// <summary>Writes a <see cref="Transcription"/> in one of the <see cref="TranscriptFormat"/>s.</summary>
public static class TranscriptWriter
{
    /// <summary>Writes <paramref name="transcription"/> to <paramref name="output"/>.</summary>
    /// <param name="transcription">The result.</param>
    /// <param name="format">The format.</param>
    /// <param name="output">The destination.</param>
    public static async Task WriteAsync(Transcription transcription, TranscriptFormat format, TextWriter output)
    {
        switch (format)
        {
            case TranscriptFormat.Text:
                await output.WriteLineAsync(transcription.Text);
                break;
            case TranscriptFormat.Segments:
                foreach (var segment in transcription.Segments)
                {
                    await output.WriteLineAsync($"[{Clock(segment.Start, '.')} --> {Clock(segment.End, '.')}] {segment.Text.Trim()}");
                }

                break;
            case TranscriptFormat.Srt:
                for (int i = 0; i < transcription.Segments.Count; i++)
                {
                    var segment = transcription.Segments[i];
                    await output.WriteLineAsync((i + 1).ToString(CultureInfo.InvariantCulture));
                    await output.WriteLineAsync($"{Clock(segment.Start, ',')} --> {Clock(segment.End, ',')}");
                    await output.WriteLineAsync(segment.Text.Trim());
                    await output.WriteLineAsync();
                }

                break;
            case TranscriptFormat.Json:
                await output.WriteLineAsync(Json(transcription));
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(format), format, "Unknown transcript format.");
        }
    }

    private static string Clock(TimeSpan time, char fractionSeparator)
        => string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours:00}:{time.Minutes:00}:{time.Seconds:00}{fractionSeparator}{time.Milliseconds:000}");

    private static string Json(Transcription transcription)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteString("text", transcription.Text);
            writer.WriteString("language", transcription.Language);
            writer.WriteNumber("duration", transcription.Duration.TotalSeconds);
            writer.WriteStartArray("segments");
            foreach (var segment in transcription.Segments)
            {
                writer.WriteStartObject();
                writer.WriteNumber("start", segment.Start.TotalSeconds);
                writer.WriteNumber("end", segment.End.TotalSeconds);
                writer.WriteString("text", segment.Text);
                if (segment.AverageLogProbability is double logProbability)
                {
                    writer.WriteNumber("avg_logprob", logProbability);
                }

                if (segment.NoSpeechProbability is double noSpeech)
                {
                    writer.WriteNumber("no_speech_prob", noSpeech);
                }

                writer.WriteEndObject();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }
}
