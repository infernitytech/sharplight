using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;
using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// VibeVoice-ASR (the acoustic and the semantic speech tokenizer plus two connectors feeding a Qwen2 language model that writes a speaker-diarized, timestamped JSON transcript) against
/// transformers' <c>VibeVoiceAsrForConditionalGeneration</c> (CPU, f32, weights of the original Microsoft checkpoint renamed to transformers' layout; references from
/// <c>tools/make_vibevoice_asr_reference.py</c>, greedy, the acoustic latents' mean instead of a sample): the prompt, the audio embedding rows, the greedy tokens (teacher forced and free), and
/// the segments with speakers and timestamps through the speech-recognition task client; on the CPU and on Vulkan.
/// </summary>
public sealed class VibeVoiceAsrParityTests
{
    private static string Model => SpeechEnvironment.VibeVoice("VibeVoice-ASR");

    private static string Reference(string clip, string name) => SpeechEnvironment.VibeVoice(Path.Combine("ref", clip, name));

    private static float[] Floats(string clip, string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Reference(clip, name + ".bin"))).ToArray();

    private static int[] Ints(string clip, string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Reference(clip, name + ".bin"))).ToArray();

    private static JsonElement Info(string clip) => JsonDocument.Parse(File.ReadAllText(Reference(clip, "info.json"))).RootElement;

    /// <summary>The recordings (reference folders) and devices.</summary>
    public static TheoryData<string, string> Cases()
    {
        var cases = new TheoryData<string, string>();
        foreach (string clip in new[] { "asr_jfk", "asr_two" })
        {
            cases.Add(clip, "cpu");
            cases.Add(clip, "vulkan");
        }

        return cases;
    }

    private static TaskModel Load(string clip, string device)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Model, "model.safetensors.index.json")) && File.Exists(Path.Combine(Model, "tokenizer.json")), $"{Model} is not present");
        Assert.SkipUnless(File.Exists(Reference(clip, "info.json")), $"the reference of {clip} is not present");
        var gpu = SpeechEnvironment.Device(device);
        Assert.SkipUnless(gpu is not null, "no Vulkan device");
        return TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = Model, Devices = [new DeviceSlot { Device = gpu! }] });
    }

    private static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double error = 0;
        double energy = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            error += d * d;
            energy += (double)expected[i] * expected[i];
        }

        return Math.Sqrt(error / energy);
    }

    /// <summary>The chat prompt (system text, audio placeholders, duration, instruction) is the reference's token for token, and the audio rows match transformers' (relative RMS).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Prompt_and_audio_rows_match_the_reference(string clip, string device)
    {
        using var model = Load(clip, device);
        using var recognizer = new VibeVoiceRecognizer(model);
        var samples = Floats(clip, "audio");
        var embeddings = recognizer.Encode(samples);
        var reference = Floats(clip, "audio_rows");
        var tokens = recognizer.PromptTokens(embeddings.Length / 3584, (double)samples.Length / 24000, null);
        Assert.Equal(Info(clip).GetProperty("input_ids").EnumerateArray().Select(v => v.GetInt32()), tokens);
        Assert.Equal(reference.Length, embeddings.Length);
        double error = RelativeRms(embeddings, reference);
        TestContext.Current.TestOutputHelper?.WriteLine($"{clip} {device}: audio rows relative RMS {error:E2}");
        Console.WriteLine($"VibeVoice-ASR {clip} {device}: {embeddings.Length / 3584} audio rows, relative RMS {error:E2}");
        Assert.True(error < (device == "cpu" ? 1e-3 : 1e-2), $"audio rows relative RMS {error:E2}");
    }

    /// <summary>
    /// Along the reference's greedy tokens the model's argmax agrees with the reference's unless the reference's own margin is small, the top logits agree, and free greedy decoding gives the
    /// reference's tokens up to the end of the turn.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Greedy_tokens_match_the_reference(string clip, string device)
    {
        using var model = Load(clip, device);
        using var recognizer = new VibeVoiceRecognizer(model);
        var embeddings = recognizer.Encode(Floats(clip, "audio"));
        var prompt = recognizer.PromptTokens(embeddings.Length / 3584, (double)Floats(clip, "audio").Length / 24000, null);
        var reference = Ints(clip, "tokens");
        var top2 = Ints(clip, "top2_ids");
        var top2Values = Floats(clip, "top2_values");
        int end = Array.FindIndex(reference, t => t is 151645 or 151643);
        var options = new SpeechRecognitionOptions();
        int hard = 0;
        double worst = 0;
        recognizer.GenerateTokens(embeddings, prompt, options, CancellationToken.None, reference[..end], (i, logits) =>
        {
            int best = Sampler.ArgMax(logits);
            if (best != top2[2 * i] && top2Values[2 * i] - top2Values[(2 * i) + 1] > 0.05f)
            {
                hard++;
            }

            worst = Math.Max(worst, Math.Abs(logits[top2[2 * i]] - top2Values[2 * i]));
        }).ToList();
        var greedy = recognizer.GenerateTokens(embeddings, prompt, options, CancellationToken.None).ToList();
        int same = 0;
        while (same < Math.Min(greedy.Count, end) && greedy[same] == reference[same])
        {
            same++;
        }

        string line = $"VibeVoice-ASR {clip} {device}: teacher forced over {end} tokens, argmax disagreements {hard}, worst top-logit delta {worst:E2}; greedy {greedy.Count} tokens (reference {end}), identical prefix {same}";
        Console.WriteLine(line);
        TestContext.Current.TestOutputHelper?.WriteLine(line);
        Assert.Equal(0, hard);
        Assert.True(worst < (device == "cpu" ? 0.25 : 0.5), $"worst top logit delta {worst:E2}");
        Assert.Equal(end, same);
        Assert.Equal(end, greedy.Count);
    }

    /// <summary>Through the task client: segments with start and end times, speakers and text; the text is the reference's.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public async Task Transcribes_with_speakers_and_timestamps(string clip, string device)
    {
        using var model = Load(clip, device);
        using var recognizer = (ISpeechRecognizer)new VibeVoiceRecognizer(model);
        var samples = Floats(clip, "audio");
        var result = await recognizer.TranscribeAsync(new AudioBuffer(samples, 24000), cancellationToken: TestContext.Current.CancellationToken);
        Assert.NotEmpty(result.Segments);
        foreach (var segment in result.Segments)
        {
            Assert.NotNull(segment.Speaker);
            Assert.True(segment.End >= segment.Start, $"{segment.Start} to {segment.End}");
            Assert.True(segment.End <= TimeSpan.FromSeconds((double)samples.Length / 24000) + TimeSpan.FromSeconds(1));
        }

        string reference = Info(clip).GetProperty("text").GetString()!;
        var parsed = JsonDocument.Parse(reference[reference.IndexOf('[', StringComparison.Ordinal)..(reference.LastIndexOf(']') + 1)]).RootElement;
        Assert.Equal(parsed.GetArrayLength(), result.Segments.Count);
        for (int i = 0; i < result.Segments.Count; i++)
        {
            Assert.Equal(parsed[i].GetProperty("Content").GetString()!.Trim(), result.Segments[i].Text);
            Assert.Equal(parsed[i].GetProperty("Speaker").GetInt32().ToString(System.Globalization.CultureInfo.InvariantCulture), result.Segments[i].Speaker);
            Assert.Equal(parsed[i].GetProperty("Start").GetDouble(), result.Segments[i].Start.TotalSeconds, 0.01);
            Assert.Equal(parsed[i].GetProperty("End").GetDouble(), result.Segments[i].End.TotalSeconds, 0.01);
        }

        Console.WriteLine($"VibeVoice-ASR {clip} {device}: {result.Segments.Count} segments, speakers {string.Join(",", result.Segments.Select(s => s.Speaker).Distinct())}: {result.Text}");
    }

    /// <summary>The model converted to a <c>.slm</c> package and loaded again generates the same tokens and segments (the package is written below <c>SHARPLIGHT_WORK</c> and deleted).</summary>
    [Fact]
    public async Task Package_round_trip_transcribes_the_same()
    {
        const string Clip = "asr_jfk";
        Assert.SkipUnless(File.Exists(Path.Combine(Model, "model.safetensors.index.json")) && File.Exists(Path.Combine(Model, "tokenizer.json")) && File.Exists(Reference(Clip, "info.json")), "the model is not present");
        string folder = Path.Combine(Environment.GetEnvironmentVariable("SHARPLIGHT_WORK") ?? Path.GetTempPath(), "speech-tests");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, $"vibevoice-asr-{Guid.NewGuid():N}.slm");
        try
        {
            TestHost.Converter.Convert(Model, path);
            var package = Sharplight.Package.ModelPackageReader.Open(path);
            Assert.Equal(Sharplight.Models.ModelKinds.SpeechRecognition, package.Definition.Description.Kind);
            var samples = Floats(Clip, "audio");
            var original = await TranscribeAsync(Model, samples);
            var packaged = await TranscribeAsync(path, samples);
            Assert.Equal(original.Text, packaged.Text);
            Assert.Equal(original.Segments.Select(s => (s.Start, s.End, s.Speaker)), packaged.Segments.Select(s => (s.Start, s.End, s.Speaker)));
            Console.WriteLine($"VibeVoice-ASR package round trip: {packaged.Segments.Count} segment(s), identical text and times");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static async Task<Transcription> TranscribeAsync(string path, float[] samples)
    {
        var gpu = SpeechEnvironment.Device("vulkan");
        using var model = TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [new DeviceSlot { Device = gpu ?? SpeechEnvironment.Device("cpu")! }] });
        using var recognizer = (ISpeechRecognizer)new VibeVoiceRecognizer(model);
        return await recognizer.TranscribeAsync(new AudioBuffer(samples, 24000), cancellationToken: TestContext.Current.CancellationToken);
    }
}
