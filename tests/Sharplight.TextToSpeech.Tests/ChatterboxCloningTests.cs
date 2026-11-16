using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Architectures;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>
/// Chatterbox voice cloning against the chatterbox-tts package (reference tensors from <c>tools/make_chatterbox_clone_reference.py</c>, float32 on the
/// CPU, LibriSpeech 1272-128104-0004 as the recording): every encoder on the reference's own input (voice encoder partials and embedding, S3 tokens,
/// CAM++ x-vector), then the front ends (voice-encoder mel energies, S3 log-mel, Kaldi filter bank, flow prompt log-mel) on the reference's signals.
/// </summary>
public class ChatterboxCloningTests
{
    private static string Model => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "chatterbox");

    private static string Reference => Path.Combine(Path.GetDirectoryName(TtsModels.Kokoro)!, "chatterbox-clone-ref");

    private static float[] Floats(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static int[] Ints(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(Path.Combine(Reference, name))).ToArray();

    private static JsonElement Info() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Reference, "clone.json"))).RootElement;

    private static ChatterboxVoiceCloning Cloning(Sharplight.Tasks.TaskModel model)
    {
        Assert.True(ChatterboxVoiceCloning.Supports(model));
        return new ChatterboxVoiceCloning(model);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Encoders_match_the_reference_on_its_inputs(bool gpu)
    {
        TtsModels.Require(Path.Combine(Model, "ve.safetensors"), Path.Combine(Reference, "xvector.f32"));
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var cloning = Cloning(model);
        var info = Info();
        string device = gpu ? "vulkan" : "cpu";

        int veFrames = info.GetProperty("ve_frames").GetInt32();
        var partials = cloning.PartialEmbeddings(Floats("ve_mel.f32"), veFrames);
        var referencePartials = Floats("ve_partials.f32");
        Assert.Equal(info.GetProperty("ve_partials").GetInt32(), partials.Count);
        double partialError = RelativeError(referencePartials, [.. partials.SelectMany(p => p)]);
        double speakerError = RelativeError(Floats("ve_embed.f32"), cloning.SpeakerEmbedding(Floats("ve_mel.f32"), veFrames));
        ParityReport.Write($"Chatterbox cloning {device}: voice encoder partials {partialError:E2}, speaker embedding {speakerError:E2}");
        Assert.True(partialError < 1e-4, $"voice encoder partial embeddings relative RMS error {partialError:E2}");
        Assert.True(speakerError < 1e-4, $"speaker embedding relative RMS error {speakerError:E2}");

        var t3Tokens = cloning.SpeechTokens(Floats("t3_mel.f32"), info.GetProperty("t3_frames").GetInt32());
        var referenceTokens = Ints("t3_tokens.i32");
        int agree = t3Tokens.Zip(referenceTokens).Count(p => p.First == p.Second);
        ParityReport.Write($"Chatterbox cloning {device}: {agree} of {referenceTokens.Length} T3 prompt tokens identical");
        Assert.Equal(referenceTokens.Length, t3Tokens.Length);
        Assert.True(agree >= referenceTokens.Length - 2, $"{agree} of {referenceTokens.Length} speech tokens identical");

        double xvectorError = RelativeError(Floats("xvector.f32"), cloning.XVector(Floats("cam_fbank.f32"), info.GetProperty("fbank_frames").GetInt32()));
        ParityReport.Write($"Chatterbox cloning {device}: x-vector {xvectorError:E2}");
        Assert.True(xvectorError < 1e-4, $"x-vector relative RMS error {xvectorError:E2}");
    }

    [Fact]
    public void Front_ends_match_the_reference()
    {
        TtsModels.Require(Path.Combine(Model, "ve.safetensors"), Path.Combine(Reference, "xvector.f32"));
        using var model = TtsModels.Load(TtsModels.Open(gpu: false), Model);
        using var cloning = Cloning(model);
        var info = Info();

        var trimmed = ChatterboxVoiceCloning.Trim(Floats("ref16.f32"), 20);
        Assert.Equal(Floats("ve_trimmed.f32").Length, trimmed.Length);

        double speakerError = RelativeError(Floats("ve_embed.f32"), cloning.SpeakerEmbedding(Floats("ref16.f32")));
        var tokens = cloning.SpeechTokens(Floats("ref16.f32").AsSpan(0, 6 * 16000), 150);
        int agree = tokens.Zip(Ints("t3_tokens.i32")).Count(p => p.First == p.Second);
        double xvectorError = RelativeError(Floats("xvector.f32"), cloning.XVector(Floats("dec16.f32")));
        var promptMel = cloning.PromptMel(Floats("dec24.f32"), out int frames);
        Assert.Equal(info.GetProperty("prompt_frames").GetInt32(), frames);
        double melError = RelativeError(Floats("prompt_feat.f32"), promptMel);
        var flowTokens = cloning.SpeechTokens(Floats("dec16.f32"), maxTokens: null);
        int flowAgree = flowTokens.Zip(Ints("prompt_tokens.i32")).Count(p => p.First == p.Second);
        ParityReport.Write($"Chatterbox cloning front ends: speaker embedding {speakerError:E2}, {agree} of 150 T3 tokens, x-vector {xvectorError:E2}, "
            + $"prompt mel {melError:E2}, {flowAgree} of {Ints("prompt_tokens.i32").Length} flow tokens");
        Assert.True(speakerError < 1e-3, $"speaker embedding relative RMS error {speakerError:E2}");
        Assert.True(agree >= 147, $"{agree} of 150 T3 prompt tokens identical");
        Assert.True(xvectorError < 1e-3, $"x-vector relative RMS error {xvectorError:E2}");
        Assert.True(melError < 1e-4, $"prompt mel relative RMS error {melError:E2}");
        Assert.True(flowAgree >= Ints("prompt_tokens.i32").Length - 5, $"{flowAgree} flow tokens identical");
    }

    /// <summary>
    /// Speech in the voice of a recording through the task client: the voice encoder finds the output closer to the recording's speaker than the
    /// built-in voice's output; the real-time factor in the report.
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task Speaks_with_the_voice_of_a_recording(bool gpu)
    {
        string recordingPath = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(TtsModels.Kokoro)!)!, "audio", "librispeech-1272-128104-0004.wav");
        TtsModels.Require(Path.Combine(Model, "ve.safetensors"), recordingPath);
        using var model = TtsModels.Load(TtsModels.Open(gpu), Model);
        using var synthesizer = TtsModels.Synthesizer(model);
        using var cloning = Cloning(model);
        var recording = Sharplight.Audio.WavFile.Read(recordingPath);
        const string Text = "The weather today is sunny with a light breeze.";
        var sampling = new Sharplight.Llm.SamplingOptions { Seed = 7 };
        var builtIn = await synthesizer.SynthesizeAsync(Text, new Sharplight.Speech.SpeechSynthesisOptions { Sampling = sampling }, TestContext.Current.CancellationToken);
        var options = new Sharplight.Speech.SpeechSynthesisOptions { ReferenceAudio = recording, Sampling = sampling };
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var cloned = await synthesizer.SynthesizeAsync(Text, options, TestContext.Current.CancellationToken);
        double elapsed = stopwatch.Elapsed.TotalSeconds;
        double seconds = cloned.Audio.Samples.Length / (double)cloned.Audio.SampleRate;

        float[] Embed(Sharplight.Audio.AudioBuffer audio)
            => cloning.SpeakerEmbedding(Sharplight.Audio.AudioResampler.Resample(audio.Samples, audio.Channels, audio.SampleRate, 16000));
        var target = Embed(recording);
        double clonedSimilarity = Dot(target, Embed(cloned.Audio));
        double builtInSimilarity = Dot(target, Embed(builtIn.Audio));
        ParityReport.Write($"Chatterbox cloning end to end {(gpu ? "vulkan" : "cpu")}: {seconds:F2} s of audio in {elapsed:F2} s including the voice "
            + $"(real-time factor {elapsed / seconds:F2}); speaker similarity to the recording {clonedSimilarity:F3} (built-in voice {builtInSimilarity:F3})");
        Assert.Equal("reference", cloned.Voice);
        Assert.InRange(seconds, 1.5, 12);
        Assert.True(clonedSimilarity > builtInSimilarity + 0.05, $"cloned similarity {clonedSimilarity:F3}, built-in {builtInSimilarity:F3}");
    }

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
