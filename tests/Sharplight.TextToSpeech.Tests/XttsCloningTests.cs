using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// XTTS v2 voice cloning against coqui-tts (reference tensors from <c>tools/make_xtts_clone_reference.py</c>, float32 on the CPU, LibriSpeech
/// 1272-128104-0004 as the recording): the style encoder on the reference's chunk log-mels and the speaker encoder on its mel spectrum, then the front
/// ends on the reference's 22.05 and 16 kHz signals, and speech in the recording's voice end to end.
/// </summary>
public class XttsCloningTests
{
    private static string Model => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "XTTS-v2");

    private static string Reference => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "xtts-clone-ref");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "clone.json"))).RootElement;

    private static void RequireAll() => TtsModels.Require(Path.Combine(Model, "mel_stats.pth"), Path.Combine(Reference, "latents.f32"));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Encoders_match_the_reference_on_its_inputs(bool gpu)
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        Assert.True(XttsVoiceCloning.Supports(model));
        using var cloning = new XttsVoiceCloning(model);
        string device = gpu ? "vulkan" : "cpu";

        // The reference's chunk mels are already divided by the mel statistics; the graph divides again, so multiply them back.
        var frames = Ints("cond_frames.i32");
        var mels = Floats("cond_mels.f32");
        int padded = mels.Length / (frames.Length * 80);
        var stats = Stats(model);
        var styles = Floats("style.f32");
        int perChunk = styles.Length / frames.Length;
        double worst = 0;
        for (int c = 0; c < frames.Length; c++)
        {
            var mel = new float[frames[c] * 80];
            for (int m = 0; m < 80; m++)
            {
                for (int f = 0; f < frames[c]; f++)
                {
                    mel[(f * 80) + m] = mels[(((c * 80) + m) * padded) + f] * stats[m];
                }
            }

            worst = Math.Max(worst, RelativeError(styles.AsSpan(c * perChunk, perChunk).ToArray(), cloning.StyleLatents(mel, frames[c])));
        }

        var spec = Floats("speaker_spec.f32");
        int specFrames = Info().GetProperty("speaker_frames").GetInt32();
        var logMel = new float[spec.Length];
        for (int m = 0; m < 64; m++)
        {
            for (int f = 0; f < specFrames; f++)
            {
                logMel[(f * 64) + m] = MathF.Log(spec[(m * specFrames) + f] + 1e-6f);
            }
        }

        double speakerError = RelativeError(Floats("speaker.f32"), cloning.SpeakerEmbedding(logMel, specFrames));
        ParityReport.Write($"XTTS cloning {device}: style latents worst chunk {worst:E2} over {frames.Length} chunks, speaker embedding {speakerError:E2}");
        Assert.True(worst < 1e-4, $"style latents relative RMS error {worst:E2}");
        Assert.True(speakerError < 1e-4, $"speaker embedding relative RMS error {speakerError:E2}");
    }

    [Fact]
    public void Front_ends_match_the_reference()
    {
        RequireAll();
        using var model = TtsModels.Load(TtsModels.Open(gpu: false), Model);
        using var cloning = new XttsVoiceCloning(model);
        double latentError = RelativeError(Floats("latents.f32"), cloning.Latents(Floats("audio22.f32")));
        double speakerError = RelativeError(Floats("speaker.f32"), cloning.SpeakerEmbedding(Floats("audio16.f32")));
        ParityReport.Write($"XTTS cloning front ends: conditioning latents {latentError:E2}, speaker embedding {speakerError:E2}");
        Assert.True(latentError < 1e-3, $"conditioning latents relative RMS error {latentError:E2}");
        Assert.True(speakerError < 1e-3, $"speaker embedding relative RMS error {speakerError:E2}");
    }

    /// <summary>Speech in the voice of a recording through the task client: closer to the recording's speaker than a built-in speaker's.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Speaks_with_the_voice_of_a_recording(bool gpu)
    {
        string recordingPath = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(TtsModels.Kokoro)!)!, "audio", "librispeech-1272-128104-0004.wav");
        RequireAll();
        TtsModels.Require(recordingPath);
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = TtsModels.Synthesizer(model);
        using var cloning = new XttsVoiceCloning(model);
        var recording = Sharplight.Audio.WavFile.Read(recordingPath);
        const string Text = "The weather today is sunny with a light breeze.";
        var sampling = new Sharplight.Llm.SamplingOptions { Seed = 7 };
        var builtIn = await synthesizer.SynthesizeAsync(Text, new Sharplight.Speech.SpeechSynthesisOptions { Voice = "Claribel Dervla", Sampling = sampling },
            TestContext.Current.CancellationToken);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var cloned = await synthesizer.SynthesizeAsync(Text, new Sharplight.Speech.SpeechSynthesisOptions { ReferenceAudio = recording, Sampling = sampling },
            TestContext.Current.CancellationToken);
        double elapsed = stopwatch.Elapsed.TotalSeconds;
        double seconds = cloned.Audio.Samples.Length / (double)cloned.Audio.SampleRate;

        float[] Embed(Sharplight.Audio.AudioBuffer audio)
            => cloning.SpeakerEmbedding(Sharplight.Audio.AudioResampler.Resample(audio.ToMono().Samples, 1, audio.SampleRate, 16000));
        var target = Embed(recording);
        double clonedSimilarity = Dot(target, Embed(cloned.Audio));
        double builtInSimilarity = Dot(target, Embed(builtIn.Audio));
        ParityReport.Write($"XTTS cloning end to end {(gpu ? "vulkan" : "cpu")}: {seconds:F2} s of audio in {elapsed:F2} s including the voice; speaker "
            + $"similarity to the recording {clonedSimilarity:F3} (built-in speaker {builtInSimilarity:F3})");
        Assert.Equal("reference", cloned.Voice);
        Assert.InRange(seconds, 1.5, 12);
        Assert.True(clonedSimilarity > builtInSimilarity + 0.05, $"cloned similarity {clonedSimilarity:F3}, built-in {builtInSimilarity:F3}");
    }

    private static float[] Stats(Sharplight.Tasks.TaskModel model)
        => WeightConversions.ReadFloats(model.Definition.Weights["gpt.mel_stats"]);

    private static double Dot(float[] a, float[] b) => a.Zip(b, (x, y) => (double)x * y).Sum();

    private static double RelativeError(float[] reference, float[] actual)
    {
        Assert.Equal(reference.Length, actual.Length);
        double error = 0;
        double energy = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            error += d * d;
            energy += (double)reference[i] * reference[i];
        }

        return Math.Sqrt(error / energy);
    }
}
