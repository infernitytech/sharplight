using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// whisper.cpp model files (f16 and q5_1) against whisper.cpp itself on jfk.wav: the transcript with timestamps (segment texts and times)
/// and without, as whisper.cpp's CPU build prints them (<c>whisper-cli -bs 1 -bo 1 -l en</c>, fixture <c>jfk.whisper.cpp.json</c>), on the CPU
/// and on Vulkan. f16 models must match exactly. Quantized models must match word for word: ggml multiplies quantized weights with
/// activations quantized to 8 bits, Sharplight with the dequantized weights, so punctuation decided by near-equal logits may differ
/// (whisper.cpp's own CPU and CUDA builds disagree on such commas for these files).
/// </summary>
public sealed class WhisperCppParityTests
{
    private static readonly string[] s_models =
    [
        "ggml-tiny", "ggml-tiny.en", "ggml-base", "ggml-base.en", "ggml-small", "ggml-small.en",
        "ggml-tiny-q5_1", "ggml-tiny.en-q5_1", "ggml-base-q5_1", "ggml-base.en-q5_1", "ggml-small-q5_1", "ggml-small.en-q5_1",
    ];

    /// <summary>The model files and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string model in s_models)
        {
            cases.Add(model, "cpu");
            cases.Add(model, "vulkan");
        }

        return cases;
    }

    /// <summary>The segments with timestamps equal whisper.cpp's.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task SegmentsMatchWhisperCpp(string name, string device)
    {
        using var model = Load(name, device);
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        using var fixture = SpeechEnvironment.Fixture("jfk.whisper.cpp.json");
        var expected = fixture.RootElement.GetProperty(name).GetProperty("segments").EnumerateArray()
            .Select(s => (s.GetProperty("from").GetInt32(), s.GetProperty("to").GetInt32(), s.GetProperty("text").GetString()!.Trim())).ToArray();
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Jfk), new SpeechRecognitionOptions { Language = Language(name) });
        var actual = result.Segments.Select(s => ((int)Math.Round(s.Start.TotalMilliseconds), (int)Math.Round(s.End.TotalMilliseconds), s.Text.Trim())).ToArray();
        if (IsQuantized(name))
        {
            Assert.Equal(Words(string.Join(" ", expected.Select(s => s.Item3))), Words(result.Text));
            return;
        }

        Assert.Equal(expected, actual);
    }

    /// <summary>The transcript without timestamps equals whisper.cpp's <c>-nt</c> output.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task TextWithoutTimestampsMatchesWhisperCpp(string name, string device)
    {
        using var model = Load(name, device);
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        using var fixture = SpeechEnvironment.Fixture("jfk.whisper.cpp.json");
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Jfk),
            new SpeechRecognitionOptions { Language = Language(name), Timestamps = TimestampGranularity.None });
        string expected = fixture.RootElement.GetProperty(name).GetProperty("text").GetString()!;
        if (IsQuantized(name))
        {
            Assert.Equal(Words(expected), Words(result.Text));
            return;
        }

        Assert.Equal(expected, result.Text);
    }

    private static bool IsQuantized(string name) => name.Contains("-q", StringComparison.Ordinal);

    /// <summary>The lowercase words of a text, without punctuation.</summary>
    private static string Words(string text)
        => string.Join(" ", new string([.. text.ToLowerInvariant().Where(c => char.IsLetterOrDigit(c) || char.IsWhiteSpace(c))])
            .Split(' ', StringSplitOptions.RemoveEmptyEntries));

    private static string? Language(string name) => name.Contains(".en", StringComparison.Ordinal) ? null : "en";

    private static TaskModel Load(string name, string device)
    {
        string path = SpeechEnvironment.WhisperCpp(name);
        if (!SpeechEnvironment.Available(path, device))
        {
            Assert.Skip($"{path}, jfk.wav or the {device} device is missing.");
        }

        return TaskModel.Load(new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
    }
}
