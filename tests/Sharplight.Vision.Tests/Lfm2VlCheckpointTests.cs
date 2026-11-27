using Sharplight.Llm;
using Sharplight.Models;

namespace Sharplight.Vision.Tests;

/// <summary>
/// LiquidAI/LFM2-VL-450M read from its Hugging Face safetensors (bf16, <c>lfm2_vl</c>) against the published F16 text GGUF with the mmproj
/// converted by llama.cpp HEAD (<c>mmproj-LFM2-VL-450M-F16-head.gguf</c>, see <see cref="SiglipVlPreprocessTests"/>): the same canonical
/// tower and preprocessing (a tiled 8-bit image: tiles plus thumbnail), nearly equal embeddings (bf16 weights are exact in f16) and the same
/// greedy continuation of an image prompt, on the CPU.
/// </summary>
public class Lfm2VlCheckpointTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    [Fact]
    public void Lfm2_vl_450m_safetensors_match_the_gguf_run()
    {
        string hf = Path.Combine(Home, "models/hf/LFM2-VL-450M");
        string gguf = Path.Combine(Home, "models/gguf/LFM2-VL-450M/LFM2-VL-450M-F16.gguf");
        string mmproj = Path.Combine(Home, "models/gguf/LFM2-VL-450M/mmproj-LFM2-VL-450M-F16-head.gguf");
        Assert.SkipUnless(File.Exists(Path.Combine(hf, "model.safetensors")) && File.Exists(gguf) && File.Exists(mmproj), "LFM2-VL-450M safetensors, GGUF or mmproj not available");

        // 1024×640 exceeds 2·256 tokens, so the processor tiles it (2×1 grid of 512² tiles here) and adds the thumbnail.
        const int width = 1024;
        const int height = 640;
        var rgb = new byte[3 * width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    rgb[(y * width + x) * 3 + c] = (byte)((7 * x + 3 * y + 50 * c + x * y % 31) % 256);
                }
            }
        }

        var image = new ImageInput(width, height, rgb);
        const string prompt = "<|startoftext|><|im_start|>user\n<image>Describe this image.<|im_end|>\n<|im_start|>assistant\n";
        var (hfEmbeddings, hfTiles, hfGreedy, hfText) = Run(LlmModel.Load(TestHost.Loader, hf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 2048 }), image, prompt);
        var (ggufEmbeddings, ggufTiles, ggufGreedy, ggufText) = Run(
            LlmModel.Load(TestHost.Loader, gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 2048, Auxiliary = { [AuxiliaryRoles.Mmproj] = mmproj } }),
            image, prompt);
        Console.Error.WriteLine($"safetensors: {hfText}");
        Console.Error.WriteLine($"GGUF:        {ggufText}");
        Assert.Equal(ggufTiles, hfTiles);
        Assert.True(hfTiles.Columns * hfTiles.Rows > 1, "the image should be tiled");
        Assert.Equal(ggufEmbeddings.Length, hfEmbeddings.Length);
        var (maxAbs, _, rms) = Env.Compare(hfEmbeddings, ggufEmbeddings);
        Console.Error.WriteLine($"embeddings max |diff| {maxAbs:E2}, relative rms {rms:E2}");
        Assert.True(rms < 1e-3, $"embeddings relative rms {rms}");
        Assert.Equal(ggufGreedy, hfGreedy);
    }

    private static (float[] Embeddings, (int Columns, int Rows) Tiles, int[] Greedy, string Text) Run(LlmModel model, ImageInput input, string prompt)
    {
        using (model)
        {
            using var vision = model.CreateVisionEncoder();
            var image = vision.Preprocessor.Process(input);
            var embeddings = vision.Encode(image);
            var ids = model.Tokenizer.Encode(prompt, addSpecial: false, parseSpecial: true);
            Assert.True(model.Architectures.TryFindMultimodal(model.Definition.Architecture, out var multimodal));
            var mm = multimodal.Assemble(new MultimodalContext(model.Definition, t => Single(model, t)), ids, [image], 0);
            using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
            var logits = session.Process(mm, [embeddings]);
            var greedy = new List<int>();
            int next = ArgMax(logits);
            while (greedy.Count < 24)
            {
                greedy.Add(next);
                next = ArgMax(session.Process([next]));
            }

            return (embeddings, (image.TileColumns, image.TileRows), greedy.ToArray(), model.Tokenizer.Decode(greedy.ToArray()));
        }
    }

    private static int ArgMax(float[] v) => Enumerable.Range(0, v.Length).MaxBy(i => v[i]);

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }
}
