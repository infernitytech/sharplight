using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// Qwen3-ASR 1.7B (an audio language model: a chunked Whisper-style audio tower whose rows replace the audio placeholders of a Qwen3 decoder's
/// prompt) against transformers (fixtures from <c>tools/make_qwen3_asr_fixtures.py</c>, the <c>-hf</c> checkpoint in float32) on jfk.wav and a
/// 29 s LibriSpeech recording: the projected audio embeddings, and token-exact greedy decoding with the language detected by the model and with
/// English forced, through the speech-recognition task client, on the CPU and on Vulkan. The original release (<c>thinker.*</c>) is the
/// model under test; the <c>-hf</c> conversion and the transcribe.cpp GGUF must give the same tokens.
/// </summary>
public sealed class Qwen3AsrParityTests
{
    private const string Fixture = "qwen3-asr-1.7b.json";

    // <|im_end|> and <asr_text> in the Qwen3 vocabulary.
    private const int QwenEndOfTurn = 151645;
    private const int QwenTranscriptMarker = 151704;

    /// <summary>The checkpoints and the files they load from.</summary>
    public static string ModelPath(string layout) => layout switch
    {
        "hf" => SpeechEnvironment.HuggingFace("Qwen3-ASR-1.7B-hf"),
        "gguf-f16" => SpeechEnvironment.SpeechGguf("Qwen3-ASR-1.7B-F16.gguf"),
        _ => SpeechEnvironment.HuggingFace("Qwen3-ASR-1.7B"),
    };

    /// <summary>The recordings and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    /// <summary>The recordings, devices and request modes ("auto": the model names the language; "en": English forced).</summary>
    public static TheoryData<string, string, string> DecodeCases()
    {
        var cases = new TheoryData<string, string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            foreach (string device in new[] { "cpu", "vulkan" })
            {
                cases.Add(clip, device, "auto");
                cases.Add(clip, device, "en");
            }
        }

        return cases;
    }

    /// <summary>The recordings, devices and request modes of the 0.6B model.</summary>
    public static TheoryData<string, string, string> SmallModelCases()
    {
        var cases = new TheoryData<string, string, string>();
        foreach (string clip in new[] { SpeechEnvironment.JfkName, SpeechEnvironment.LibriSpeechName })
        {
            foreach (string device in new[] { "cpu", "vulkan" })
            {
                cases.Add(clip, device, "auto");
                cases.Add(clip, device, "en");
            }
        }

        return cases;
    }

    /// <summary>Other layouts of the same model on jfk.wav (Vulkan).</summary>
    public static TheoryData<string> Layouts() => ["hf", "gguf-f16"];

    /// <summary>
    /// The log-mel features of the valid frames match transformers' feature extractor (jfk.wav: within 6e-4 of values up to 1.4, the frames
    /// near the recording's peak; the greedy tokens are identical all the same).
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void FeaturesMatchTransformers(string clip, string device)
    {
        string path = ModelPath("original");
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        int frames = expected.GetProperty("frames").GetInt32();
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        var window = new float[features.Channels * frames];
        features.CopyWindow(0, frames, window);
        double error = SpeechEnvironment.MaxError(window, expected.GetProperty("features"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip}: features max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-3 * Math.Max(scale, 1), $"features max abs error {error:G3}");
    }

    /// <summary>The projected audio embeddings match transformers' (relative to their largest value).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void EmbeddingsMatchTransformers(string clip, string device)
    {
        string path = ModelPath("original");
        Skip(path, clip, device);
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }] });
        using var fixture = SpeechEnvironment.Fixture(Fixture);
        var expected = fixture.RootElement.GetProperty(clip);
        var samples = WavFile.Read(SpeechEnvironment.Audio(clip)).ToModelInput(16000).Samples;
        var features = AudioFeatureExtractors.Create(model).Extract(samples);
        using var recognizer = new AudioLanguageSpeechRecognizer(model);
        int frames = expected.GetProperty("frames").GetInt32();
        var embeddings = recognizer.Encode(features, 0, frames);
        double error = SpeechEnvironment.MaxError(embeddings, expected.GetProperty("embeddings"), out double scale);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: embeddings max abs error {error:G3} of max abs {scale:G3}");
        Assert.True(error < 1e-2 * scale, $"embeddings max abs error {error:G3} of max |value| {scale:G3} on {device}");
    }

    /// <summary>Greedy decoding through the task client gives transformers' tokens, text and language.</summary>
    [Theory]
    [MemberData(nameof(DecodeCases))]
    public async Task GreedyTokensMatchTransformers(string clip, string device, string mode)
    {
        await CheckTokensAsync(ModelPath("original"), clip, device, mode);
    }

    /// <summary>The transformers conversion and the transcribe.cpp F16 GGUF give the same tokens as the original release.</summary>
    [Theory]
    [MemberData(nameof(Layouts))]
    public async Task OtherLayoutsGiveTheSameTokens(string layout)
    {
        await CheckTokensAsync(ModelPath(layout), SpeechEnvironment.JfkName, "vulkan", "auto");
    }

    /// <summary>The 0.6B model (transformers' <c>-hf</c> conversion) gives the tokens of transformers' <c>generate</c> on the same recordings.</summary>
    /// <remarks>
    /// On LibriSpeech the comparison stops before the 9th token: after "a sort of" the reference's top two logits are 29.405 (",") and 29.402
    /// ("up"), a 0.003 near-tie (1e-4 relative) that float noise decides: this engine's f32 CPU path has "up" ahead at 29.3995 to 29.3988 (margin 0.0007, so a summation-order change
    /// of a few float ULPs flips it; it resolved to the comma before), and the f16 compute of the GPU path also may resolve it the other way. Everything before the tie is token-exact; the text after the tie is not compared because the two readings
    /// continue differently.
    /// </remarks>
    [Theory]
    [MemberData(nameof(SmallModelCases))]
    public async Task SmallModelTokensMatchTransformers(string clip, string device, string mode)
    {
        // The tie after "a sort of" (the 9th transcript token) on the LibriSpeech clip: only what precedes it is compared.
        await CheckTokensAsync(SpeechEnvironment.HotModel("Qwen3-ASR-0.6B-hf"), clip, device, mode, "qwen3-asr-0.6b.json",
            clip == SpeechEnvironment.LibriSpeechName ? 8 : null);
    }

    /// <summary>
    /// Reson8's Rumble-2-Flash (a Qwen3-ASR 0.6B derivative with 14 decoder layers, an 18-layer audio tower and a vocabulary pruned to 99,630 rows,
    /// so its special tokens have other ids) gives the tokens of transformers' <c>generate</c>.
    /// </summary>
    [Theory]
    [MemberData(nameof(SmallModelCases))]
    public async Task RumbleTokensMatchTransformers(string clip, string device, string mode)
    {
        const int EndOfTurn = 99564;
        const int TranscriptMarker = 99623;
        await CheckTokensAsync(SpeechEnvironment.HotModel("Rumble-2-Flash"), clip, device, mode, "rumble-2-flash.json", null, EndOfTurn, TranscriptMarker);
    }

    /// <summary>
    /// oruk's Fourier 2B (a Qwen3-ASR 1.7B fine-tune with the base model's layout and vocabulary) gives the tokens of transformers' plain greedy
    /// <c>generate</c>. Its generation_config.json asks for beam search (3 beams), an 8-gram repeat block and 32k banned token ids, which this
    /// engine does not apply: the comparison is greedy against greedy.
    /// </summary>
    [Theory]
    [MemberData(nameof(SmallModelCases))]
    public async Task OrukFourierTokensMatchTransformers(string clip, string device, string mode)
    {
        await CheckTokensAsync(SpeechEnvironment.HotModel("oruk-fourier-2b"), clip, device, mode, "oruk-fourier-2b.json");
    }

    private static Task CheckTokensAsync(string path, string clip, string device, string mode) => CheckTokensAsync(path, clip, device, mode, Fixture);

    private static async Task CheckTokensAsync(string path, string clip, string device, string mode, string fixtureName, int? tieAt = null,
        int endOfTurn = QwenEndOfTurn, int transcriptMarker = QwenTranscriptMarker)
    {
        Skip(path, clip, device);
        using var fixture = SpeechEnvironment.Fixture(fixtureName);
        var expected = fixture.RootElement.GetProperty(clip).GetProperty(mode);
        var services = new ServiceCollection().AddSharplight()
            .AddSpeechRecognizer(o =>
            {
                o.ModelPath = path;
                o.Devices = [new DeviceSlot { Device = SpeechEnvironment.Device(device)! }];
            });
        await using var provider = services.BuildServiceProvider();
        var recognizer = provider.GetRequiredService<ISpeechRecognizer>();
        var options = new SpeechRecognitionOptions { Language = mode == "auto" ? null : mode };
        var result = await recognizer.TranscribeAsync(WavFile.Read(SpeechEnvironment.Audio(clip)), options, TestContext.Current.CancellationToken);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device} {mode}: [{result.Language}] {result.Text}");
        int[] ids = [.. expected.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32())];
        int end = ids.Length > 0 && ids[^1] == endOfTurn ? ids.Length - 1 : ids.Length;
        int begin = mode == "auto" ? Array.IndexOf(ids, transcriptMarker) + 1 : 0;
        int[] actual = result.Segments.SelectMany(s => s.Tokens).ToArray();
        string text = expected.GetProperty("text").GetString()!;
        text = text[(text.IndexOf("<asr_text>", StringComparison.Ordinal) is var at and >= 0 ? at + "<asr_text>".Length : 0)..].Trim();
        if (tieAt is int tie)
        {
            Assert.Equal(ids[begin..(begin + tie)], actual[..tie]);
            Assert.StartsWith(text[..text.IndexOf(',')], result.Text, StringComparison.Ordinal);
        }
        else
        {
            Assert.Equal(ids[begin..end], actual);
            Assert.Equal(text, result.Text);
        }

        Assert.Equal("en", result.Language);
    }

    private static void Skip(string path, string clip, string device)
    {
        if (!SpeechEnvironment.Available(path, device) || !File.Exists(SpeechEnvironment.Audio(clip)))
        {
            Assert.Skip($"{path}, {clip}.wav or the {device} device is missing.");
        }
    }
}
