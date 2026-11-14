using Sharplight.Cpu;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Runtime;

namespace Sharplight.Families.Tests;

/// <summary>
/// An encoder-decoder model (T5) behind the serving engine: the engine gets a session decoder, prompt chunks of a new context are
/// collected and encoded together, and a finished context is never continued by the next request (its prompt is a new source sequence).
/// </summary>
public class EncoderDecoderServingTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    // tiny-random-t5-v1_1's greedy continuation (transformers; see MappedFamilyCases).
    private static readonly int[] Expected = [8847, 26621, 8889, 8889, 31376, 731, 731, 28428, 28176, 8298, 27172, 9762, 7827, 7827, 7827, 29378, 28529];

    static EncoderDecoderServingTests() => HuggingFaceImporter.Register();

    [Fact]
    public async Task Engine_encodes_chunked_prompts_and_never_reuses_a_context()
    {
        string path = Path.Combine(Home, "models", "hf", "tiny-random-t5-v1_1");
        Assert.SkipUnless(Directory.Exists(path), $"{path} not present");
        var cpu = new CpuExecutionProvider();
        using var device = cpu.OpenDevice(cpu.GetDevices()[0]);
        using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [new DeviceSlot { Device = device }], ContextLength = 64, MaxBatchTokens = 8 });
        Assert.True(model.IsEncoderDecoder);
        var decoder = model.CreateBatchDecoder(slots: 4);
        Assert.IsType<SessionBatchDecoder>(decoder);
        Assert.False(decoder.ReusesPrefixes);

        var prompt = model.Tokenizer.Encode("The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art", addSpecial: true);
        Assert.True(prompt.Length > 8, "the prompt must span several batches");
        using var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions());
        var ct = TestContext.Current.CancellationToken;
        for (int round = 0; round < 2; round++)
        {
            var handle = engine.Submit(new GenerationRequest
            {
                PromptTokens = prompt, MaxTokens = Expected.Length, StopAtEog = false, Logprobs = true, Sampling = new SamplingParams { Temperature = 0 },
            }, ct);
            var result = await handle.CompleteAsync(ct);
            Assert.Null(result.Error);
            Assert.Equal(Expected, result.Logprobs.Select(l => l.TokenId).ToArray());
        }
    }
}
