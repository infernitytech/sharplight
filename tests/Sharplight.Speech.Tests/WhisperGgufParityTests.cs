using Sharplight.Audio;
using Sharplight.Import.Gguf;
using Sharplight.Import.WhisperCpp;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Whisper as GGUF files from the Hub against whisper.cpp on jfk.wav: the transcribe.cpp layout (handy-computer/whisper-*-gguf, the most
/// downloaded Whisper GGUF repositories) imports to the same canonical model as the whisper.cpp file of the same size (decoding keys and
/// suppress lists included), its f16 and f32 files transcribe exactly like whisper.cpp's f16 file, and its quantized files word for word.
/// A whisper.cpp ggml file published with a <c>.gguf</c> extension (oxide-lab) opens through the whisper.cpp importer. Skips when a file is missing.
/// </summary>
public sealed class WhisperGgufParityTests
{
    /// <summary>whisper.cpp's transcript of jfk.wav with large-v3-turbo (CUDA build, <c>-nt -l en</c>; identical for f16, q5_0 and q8_0).</summary>
    private const string LargeV3Turbo = "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

    private static string Gguf(string name) => Path.Combine(Path.GetDirectoryName(SpeechEnvironment.Jfk)!, "..", "whisper-gguf", name);

    /// <summary>transcribe.cpp files of whisper-tiny and the devices.</summary>
    public static TheoryData<string, string> TinyCases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string model in new[] { "whisper-tiny-F16", "whisper-tiny-F32", "whisper-tiny-Q8_0", "whisper-tiny-Q4_K_M" })
        {
            cases.Add(model, "cpu");
            cases.Add(model, "vulkan");
        }

        return cases;
    }

    [Fact]
    public void TranscribeCppGguf_ImportsLikeTheWhisperCppFile()
    {
        string gguf = Gguf("handy-computer/whisper-tiny-F16.gguf");
        string ggml = SpeechEnvironment.WhisperCpp("ggml-tiny");
        Assert.SkipUnless(File.Exists(gguf) && File.Exists(ggml), "whisper-tiny-F16.gguf or ggml-tiny.bin missing");
        var fromGguf = GgufImporter.Open(gguf);
        var fromGgml = WhisperCppImporter.Open(ggml);
        try
        {
            Assert.Equal(fromGgml.Architecture, fromGguf.Architecture);
            Assert.Equal(fromGgml.Weights.Keys.Order(StringComparer.Ordinal), fromGguf.Weights.Keys.Order(StringComparer.Ordinal));
            foreach (var (name, weight) in fromGgml.Weights)
            {
                Assert.Equal(weight.Shape, fromGguf.Weights[name].Shape);
            }

            foreach (var (key, value) in fromGgml.Metadata.Where(pair => pair.Key.StartsWith("encoder_decoder.", StringComparison.Ordinal)
                         || pair.Key.StartsWith("speech.", StringComparison.Ordinal)))
            {
                Assert.True(fromGguf.Metadata.TryGetValue(key, out var actual), $"missing {key}");
                Assert.Equal(Normalize(value), Normalize(actual!));
            }

            Assert.Equal(Bytes(fromGgml.Attachments["feature-extractor/mel-filters"]), Bytes(fromGguf.Attachments["feature-extractor/mel-filters"]));
        }
        finally
        {
            foreach (var resource in fromGguf.OwnedResources.Concat(fromGgml.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }

    [Theory]
    [MemberData(nameof(TinyCases))]
    public async Task TranscribeCppGguf_TranscribesLikeWhisperCpp(string name, string device)
    {
        using var model = Load(Gguf($"handy-computer/{name}.gguf"), device);
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        using var fixture = SpeechEnvironment.Fixture("jfk.whisper.cpp.json");
        var reference = fixture.RootElement.GetProperty("ggml-tiny");
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Jfk), new SpeechRecognitionOptions { Language = "en" });
        var expected = reference.GetProperty("segments").EnumerateArray()
            .Select(s => (s.GetProperty("from").GetInt32(), s.GetProperty("to").GetInt32(), s.GetProperty("text").GetString()!.Trim())).ToArray();
        if (name.Contains("-Q", StringComparison.Ordinal))
        {
            Assert.Equal(Words(string.Join(" ", expected.Select(s => s.Item3))), Words(result.Text));
            return;
        }

        var actual = result.Segments.Select(s => ((int)Math.Round(s.Start.TotalMilliseconds), (int)Math.Round(s.End.TotalMilliseconds), s.Text.Trim())).ToArray();
        Assert.Equal(expected, actual);
    }

    public static TheoryData<string> LargeFiles =>
    [
        "handy-computer/whisper-large-v3-turbo-Q8_0.gguf",
        "handy-computer/whisper-large-v3-turbo-Q4_K_M.gguf",
        "oxide-lab/whisper.cpp/whisper-large-v3-turbo-q5_0.gguf",
    ];

    [Theory]
    [MemberData(nameof(LargeFiles))]
    public async Task LargeV3TurboGguf_TranscribesLikeWhisperCpp(string file)
    {
        using var model = Load(Gguf(file), "vulkan");
        using var recognizer = (ISpeechRecognizer)new EncoderDecoderSpeechRecognizerFactory().Create(model);
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Jfk),
            new SpeechRecognitionOptions { Language = "en", Timestamps = TimestampGranularity.None });
        Assert.Equal(Words(LargeV3Turbo), Words(result.Text));
    }

    [Fact]
    public void CandleGguf_IsRejectedByName()
    {
        string path = Gguf("oxide-lab/whisper-large-v3-turbo-q4_0.gguf");
        Assert.SkipUnless(File.Exists(path), "oxide-lab Candle file missing");
        var error = Assert.Throws<NotSupportedException>(() => GgufImporter.Open(path));
        Assert.Contains("Candle whisper GGUF", error.Message, StringComparison.Ordinal);
    }

    private static byte[] Bytes(IWeightSource source)
    {
        var bytes = new byte[source.ByteLength];
        source.Read(0, bytes);
        return bytes;
    }

    private static object Normalize(object value) => value is Array array ? string.Join(",", array.Cast<object>()) : value;

    /// <summary>The lowercase words of a text, without punctuation.</summary>
    private static string Words(string text)
        => string.Join(" ", new string([.. text.ToLowerInvariant().Where(c => char.IsLetterOrDigit(c) || char.IsWhiteSpace(c))])
            .Split(' ', StringSplitOptions.RemoveEmptyEntries));

    private static TaskModel Load(string path, string device)
    {
        if (!SpeechEnvironment.Available(path, device))
        {
            Assert.Skip($"{path}, jfk.wav or the {device} device is missing.");
        }

        return TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
    }
}
