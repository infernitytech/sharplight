using System.Diagnostics;
using System.Text.Json;
using Sharplight.Architectures.VideoEnhancement;
using Sharplight.Cuda;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.VideoEnhancement;
using Sharplight.VideoGeneration;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// SeedVR2 (3B: the official seedvr2_ema_3b.pth, ema_vae.pth and pos_emb.pt in <c>{video root}/seedvr2/SeedVR2-3B</c>; 7B: the fp16 safetensors
/// conversion with ema_vae_fp16.safetensors and the same embeddings in <c>{video root}/seedvr2/SeedVR2-7B-fp16</c>) against the official
/// inference path (tools/seedvr2-reference.py; references in <c>{video root}/seedvr2/references/{name}</c>): the VAE encoder's distribution,
/// the DiT's prediction from the same noise and encoder sample, the decoded frames and the colour-corrected output, then the whole
/// <see cref="IVideoEnhancer"/> run from the same noise. Skips when the files or the device are missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class SeedVr2Tests
{
    /// <summary>
    /// The model a reference was made with: the official 3B files, the 3B's fp16 safetensors conversion (ComfyUI, numz/SeedVR2_comfyUI; not a
    /// plain cast of the official weights, 0.8-3 % apart) or the 7B's.
    /// </summary>
    private static string ModelPath(string reference)
    {
        string folder = reference.Contains("-7b-", StringComparison.Ordinal) ? "SeedVR2-7B-fp16"
            : reference.Contains("-comfy-fp16", StringComparison.Ordinal) ? "SeedVR2-3B-fp16"
            : "SeedVR2-3B";
        return Path.Combine(Env.VideoRoot, "seedvr2", folder);
    }

    private static string Reference(string name) => Path.Combine(Env.VideoRoot, "seedvr2", "references", name);

    private static bool Available(string reference) => Directory.Exists(ModelPath(reference)) && File.Exists(Path.Combine(Reference(reference), "index.json"));

    /// <summary>
    /// Reference runs checked on the CPU: the tiny clip in f32 (5 frames of 104 × 56 → 208 × 112) with the official weights and with the fp16
    /// safetensors conversion (the official code on the same weights).
    /// </summary>
    public static TheoryData<string> CpuCases => ["tiny-cpu-fp32", "tiny-cpu-fp32-comfy-fp16"];

    /// <summary>
    /// Reference runs checked on the GPU: the tiny clip, and 17 frames of 416 × 240 → 832 × 480 from the official bf16 runs on CUDA (3B: the
    /// official files; 7B: seedvr2_ema_7b_fp16.safetensors held in bf16).
    /// </summary>
    public static TheoryData<string> GpuCases => ["tiny-cpu-fp32", "fox-480p-cuda-bf16", "fox-480p-7b-cuda-bf16"];

    [Theory]
    [MemberData(nameof(CpuCases))]
    public void GraphsMatchReferenceOnCpu(string reference) => GraphsMatchReference(reference, "cpu");

    [Theory]
    [MemberData(nameof(GpuCases))]
    public void GraphsMatchReferenceOnVulkan(string reference) => GraphsMatchReference(reference, "vulkan");

    [Theory]
    [MemberData(nameof(GpuCases))]
    public void GraphsMatchReferenceOnCuda(string reference) => GraphsMatchReference(reference, "cuda");

    [Theory]
    [MemberData(nameof(CpuCases))]
    public Task EnhanceMatchesReferenceOnCpu(string reference) => EnhanceMatchesReference(reference, "cpu");

    [Theory]
    [MemberData(nameof(GpuCases))]
    public Task EnhanceMatchesReferenceOnVulkan(string reference) => EnhanceMatchesReference(reference, "vulkan");

    [Theory]
    [MemberData(nameof(GpuCases))]
    public Task EnhanceMatchesReferenceOnCuda(string reference) => EnhanceMatchesReference(reference, "cuda");

    private static void GraphsMatchReference(string reference, string device)
    {
        var slot = Available(reference) ? Device(device) : null;
        Assert.SkipWhen(slot is null, $"SeedVR2, the reference '{reference}' or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        using var model = Env.Load(ModelPath(reference), slot!);
        var data = new ReferenceRun(Reference(reference));
        var dimensions = data.Dimensions;

        var clip = FramesFirst(data.Tensor("padded"), data.Shape("padded"));
        var mean = GraphRunner.Run(model, ModelComponentRoles.Vae, SeedVrGraphs.VaeEncode, SeedVrGraphs.Mean, 0, dimensions, (VideoDiffusionGraphs.Frames, clip));
        var deviation = GraphRunner.Run(model, ModelComponentRoles.Vae, SeedVrGraphs.VaeEncode, SeedVrGraphs.Deviation, 0, dimensions, (VideoDiffusionGraphs.Frames, clip));
        // bf16 references ran the VAE in bf16 (the official setting); Sharplight's graphs run it in f32 or with f16 products.
        Check("encoder mean", mean, data.Tensor("latent_mean"), data.Bf16 ? 3e-2 : 1e-3);
        Check("encoder deviation", deviation, data.Tensor("latent_std"), data.Bf16 ? 5e-2 : 1e-3);

        var condition = data.Tensor("latent_mean");
        var std = data.Tensor("latent_std");
        var eps = data.Tensor("latent_eps");
        for (int i = 0; i < condition.Length; i++)
        {
            condition[i] += std[i] * eps[i];
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Transformer, SeedVrGraphs.Restore, VideoDiffusionGraphs.Prediction, 0, dimensions,
            (VideoDiffusionGraphs.Latent, data.Tensor("noise")), (SeedVrGraphs.Condition, condition), (VideoDiffusionGraphs.Timestep, new[] { 1000f }));
        // bf16 references multiply in bf16 throughout (the 7B's weights held in bf16 too); the f32 reference differs only by bf16 attention.
        Check("DiT prediction", prediction, data.Tensor("prediction"), data.Bf16 ? 1e-1 : 2e-2);

        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, SeedVrGraphs.VaeDecode, VideoDiffusionGraphs.Frames, 0, dimensions,
            (VideoDiffusionGraphs.Latent, data.Tensor("restored")));
        Check("decoded frames", decoded, data.Tensor("decoded"), data.Bf16 ? 3e-2 : 1e-3);
    }

    private static async Task EnhanceMatchesReference(string reference, string device)
    {
        var slot = Available(reference) ? Device(device) : null;
        Assert.SkipWhen(slot is null, $"SeedVR2, the reference '{reference}' or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        using var model = Env.Load(ModelPath(reference), slot!);
        using var enhancer = Env.Clients.Create<IVideoEnhancer>(model);
        var data = new ReferenceRun(Reference(reference));
        var frames = data.InputFrames();
        var outputShape = data.Shape("output");
        int height = outputShape[1];
        int width = outputShape[2];
        var options = new VideoEnhancementOptions
        {
            Width = width,
            Height = height,
            InitialNoise = data.Tensor("noise"),
            EncoderNoise = data.Tensor("latent_eps"),
        };
        var stopwatch = Stopwatch.StartNew();
        var result = await enhancer.EnhanceAsync(new VideoEnhancementRequest { Frames = frames, FramesPerSecond = 24 }, options, TestContext.Current.CancellationToken);
        double seconds = stopwatch.Elapsed.TotalSeconds;
        Assert.Equal(frames.Count, result.Frames.Count);
        var expected = File.ReadAllBytes(Path.Combine(Reference(reference), "output.u8"));
        int frameBytes = 3 * width * height;
        double worst = double.MaxValue;
        var perFrame = new List<string>();
        for (int f = 0; f < result.Frames.Count; f++)
        {
            double psnr = Env.Psnr(result.Frames[f].Rgb, expected.AsSpan(f * frameBytes, frameBytes).ToArray());
            perFrame.Add($"{psnr:F1}");
            worst = Math.Min(worst, psnr);
        }

        TestContext.Current.SendDiagnosticMessage($"SeedVR2 {reference} on {device}: {seconds:F1} s (incl. compilation), worst frame {worst:F1} dB (frames: {string.Join(" ", perFrame)})");
        Assert.True(worst > (data.Bf16 ? 30 : 40), $"worst frame PSNR {worst:F1} dB");
    }

    /// <summary>
    /// Batches, images and sizes the network does not take, on the CPU: the tiny clip one frame per batch at 210 × 115 (padded to 224 × 128
    /// and cropped back), the same frames as one batch with AdaIN and without colour correction, and one image with a scale.
    /// </summary>
    [Fact]
    public async Task BatchesImagesAndOddSizes()
    {
        const string reference = "tiny-cpu-fp32";
        Assert.SkipWhen(!Available(reference), "SeedVR2-3B or the tiny reference missing");
        using var model = Env.Load(ModelPath(reference), Env.Cpu());
        using var enhancer = Env.Clients.Create<IVideoEnhancer>(model);
        var defaults = enhancer.Defaults;
        Assert.Equal(16, defaults.SizeMultiple);
        Assert.Equal(4, defaults.FrameMultiple);
        var frames = new ReferenceRun(Reference(reference)).InputFrames();
        var token = TestContext.Current.CancellationToken;
        var batches = new List<int>();
        var perFrame = new List<ImageInput>();
        VideoEnhancementUpdate? last = null;
        await foreach (var update in enhancer.EnhanceStreamingAsync(new VideoEnhancementRequest { Frames = frames, FramesPerSecond = 12 },
            new VideoEnhancementOptions { Width = 210, Height = 115, FramesPerBatch = 1, Seed = 3 }, token))
        {
            if (update.Frames is { } done)
            {
                Assert.Equal(perFrame.Count, update.FirstFrame);
                batches.Add(done.Count);
                perFrame.AddRange(done);
            }

            last = update;
        }

        Assert.Equal(Enumerable.Repeat(1, frames.Count), batches);
        Assert.All(perFrame, f => Assert.True(f.Width == 210 && f.Height == 115));
        Assert.Equal(VideoEnhancementPhase.Completed, last!.Phase);
        Assert.Equal(12, last.FramesPerSecond);
        Assert.Equal(frames.Count, last.TotalFrames);
        Assert.Equal((210, 115), last.Size);
        foreach (var correction in new[] { VideoColorCorrection.AdaIn, VideoColorCorrection.None })
        {
            var whole = await enhancer.EnhanceAsync(new VideoEnhancementRequest { Frames = frames },
                new VideoEnhancementOptions { Height = 112, ColorCorrection = correction, Seed = 3 }, token);
            Assert.Equal(frames.Count, whole.Frames.Count);
            Assert.True(whole.Width == 208 && whole.Height == 112, $"{whole.Width}x{whole.Height}");
        }

        var image = await enhancer.EnhanceImageAsync(frames[0], new VideoEnhancementOptions { Scale = 1.5f, Seed = 3 }, token);
        Assert.True(image.Width == 156 && image.Height == 84, $"{image.Width}x{image.Height}");
    }

    /// <summary>The clips the streaming checks run on: the tiny clip on the CPU, 17 frames of the fox clip at 832 × 480 on a GPU.</summary>
    public static TheoryData<string> StreamingDevices => ["cpu", "vulkan", "cuda"];

    /// <summary>
    /// A frame source gives exactly the frames of the same clip as a list, in one batch and in batches of 5 (the list path and the source path
    /// are one path; batches are independent, so reading them one by one changes nothing).
    /// </summary>
    [Theory]
    [MemberData(nameof(StreamingDevices))]
    public async Task FrameSource_EqualsFrameList(string device)
    {
        var (enhancer, model, frames, height) = Open(device);
        using var owner = model;
        using var client = enhancer;
        var token = TestContext.Current.CancellationToken;
        foreach (int perBatch in new[] { 0, 5 })
        {
            var options = new VideoEnhancementOptions { Height = height, FramesPerBatch = perBatch, Seed = 11 };
            var fromList = await enhancer.EnhanceAsync(new VideoEnhancementRequest { Frames = frames }, options, token);
            var fromSource = await enhancer.EnhanceAsync(VideoEnhancementRequest.FromSource(frames.ToAsyncEnumerable(), 24, null), options, token);
            Assert.Equal(frames.Count, fromSource.Frames.Count);
            for (int f = 0; f < frames.Count; f++)
            {
                Assert.True(fromList.Frames[f].Rgb.AsSpan().SequenceEqual(fromSource.Frames[f].Rgb), $"frame {f} of batches of {perBatch} differs");
            }
        }
    }

    /// <summary>
    /// A long frame source (the clip repeated) with a slow consumer: the frames read from the source and not yet handed to the consumer
    /// never exceed a few batches (the batch being enhanced, the one read ahead and the queue of finished batches), whatever the length.
    /// </summary>
    [Theory]
    [MemberData(nameof(StreamingDevices))]
    public async Task LongFrameSource_HoldsABoundedNumberOfFrames(string device)
    {
        var (enhancer, model, frames, height) = Open(device);
        using var owner = model;
        using var client = enhancer;
        var token = TestContext.Current.CancellationToken;
        const int perBatch = 5;
        int total = device == "cpu" ? 60 : 200;
        long read = 0;
        long received = 0;
        long mostHeld = 0;
        async IAsyncEnumerable<ImageInput> Source()
        {
            for (int i = 0; i < total; i++)
            {
                await Task.Yield();
                long held = Interlocked.Increment(ref read) - Interlocked.Read(ref received);
                mostHeld = Math.Max(mostHeld, held);
                yield return frames[i % frames.Count];
            }
        }

        await foreach (var update in enhancer.EnhanceStreamingAsync(VideoEnhancementRequest.FromSource(Source()),
            new VideoEnhancementOptions { Height = height, FramesPerBatch = perBatch, Seed = 1 }, token))
        {
            if (update.Frames is { } done)
            {
                await Task.Delay(device == "cpu" ? 300 : 100, token);
                Interlocked.Add(ref received, done.Count);
            }
        }

        Assert.Equal(total, received);
        // The group being enhanced (one batch, up to eight on a GPU where the stages take turns), one batch read ahead, four queued
        // updates of finished frames and the batch being read.
        long bound = perBatch * (device == "cpu" ? 7 : 14);
        TestContext.Current.SendDiagnosticMessage($"SeedVR2 streaming on {device}: {total} frames, at most {mostHeld} held (bound {bound})");
        Assert.True(mostHeld <= bound, $"{mostHeld} frames held");
    }

    /// <summary>
    /// The VAE in tiles (forced by <see cref="VideoEnhancementOptions.TileSize"/>) against the whole frame: the layout is taken, and the result
    /// stays close (the tiles see less context, and their seams are crossfaded).
    /// </summary>
    [Theory]
    [MemberData(nameof(StreamingDevices))]
    public async Task TiledVae_IsCloseToTheWholeFrame(string device)
    {
        var (enhancer, model, frames, height) = Open(device);
        using var owner = model;
        using var client = enhancer;
        var token = TestContext.Current.CancellationToken;
        var options = new VideoEnhancementOptions { Height = height, Seed = 5 };
        var whole = await enhancer.EnhanceAsync(new VideoEnhancementRequest { Frames = frames }, options, token);
        int tile = device == "cpu" ? 128 : 384;
        var tiled = await enhancer.EnhanceAsync(new VideoEnhancementRequest { Frames = frames }, options with { TileSize = tile }, token);
        var layouts = ((SeedVrEnhancer)enhancer.GetService(typeof(SeedVrEnhancer))!).TileLayouts;
        Assert.Contains(layouts.Values, layout => layout.Rows.Count * layout.Columns.Count > 1);
        double worst = double.MaxValue;
        double sum = 0;
        for (int f = 0; f < frames.Count; f++)
        {
            double psnr = Env.Psnr(tiled.Frames[f].Rgb, whole.Frames[f].Rgb);
            worst = Math.Min(worst, psnr);
            sum += psnr;
        }

        TestContext.Current.SendDiagnosticMessage(
            $"SeedVR2 tiled VAE ({string.Join(", ", layouts.Select(l => $"{l.Key}: {l.Value}"))}) on {device}: worst frame {worst:F1} dB, mean {sum / frames.Count:F1} dB vs untiled");
        Assert.True(worst > 30, $"worst frame {worst:F1} dB");
    }

    /// <summary>An enhancer on <paramref name="device"/> with the clip the streaming checks use and its output height (skips when either is missing).</summary>
    private static (IVideoEnhancer Enhancer, TaskModel Model, List<ImageInput> Frames, int Height) Open(string device)
    {
        string reference = device == "cpu" ? "tiny-cpu-fp32" : "fox-480p-cuda-bf16";
        var slot = Available(reference) ? Device(device) : null;
        Assert.SkipWhen(slot is null, $"SeedVR2, the reference '{reference}' or the {device} device missing (CUDA runs with SHARPLIGHT_TEST_CUDA=1)");
        var model = Env.Load(ModelPath(reference), slot!);
        var enhancer = Env.Clients.Create<IVideoEnhancer>(model);
        var data = new ReferenceRun(Reference(reference));
        return (enhancer, model, data.InputFrames(), data.Shape("output")[1]);
    }

    /// <summary>The CPU, the Vulkan GPU or (with SHARPLIGHT_TEST_CUDA=1) the first CUDA GPU, or null when unavailable.</summary>
    private static DeviceSlot? Device(string name)
    {
        if (name != "cuda")
        {
            return Env.Device(name);
        }

        if (Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_CUDA") != "1")
        {
            return null;
        }

        var cuda = new CudaProvider();
        return cuda.CheckAvailability().IsAvailable ? new DeviceSlot { Device = cuda.OpenDevice(cuda.GetDevices()[0]) } : null;
    }

    private static void Check(string what, float[] actual, float[] expected, double tolerance)
    {
        var (maxAbs, rms) = Env.Compare(actual, expected);
        TestContext.Current.SendDiagnosticMessage($"SeedVR2 {what}: max abs {maxAbs:E2}, relative RMS {rms:E2}");
        Assert.True(rms < tolerance, $"{what}: max abs {maxAbs}, relative RMS {rms}");
    }

    /// <summary>[3, F, H, W] → [F, 3, H, W].</summary>
    private static float[] FramesFirst(float[] channelsFirst, int[] shape)
    {
        int frames = shape[1];
        int plane = shape[2] * shape[3];
        var result = new float[channelsFirst.Length];
        for (int c = 0; c < 3; c++)
        {
            for (int f = 0; f < frames; f++)
            {
                channelsFirst.AsSpan((c * frames + f) * plane, plane).CopyTo(result.AsSpan((f * 3 + c) * plane));
            }
        }

        return result;
    }

    /// <summary>One reference run of tools/seedvr2-reference.py.</summary>
    private sealed class ReferenceRun(string folder)
    {
        private readonly JsonElement _index = ReadIndex(folder);

        public bool Bf16 => _index.GetProperty("dit_dtype").GetString() == "bf16";

        public Dictionary<string, long> Dimensions
        {
            get
            {
                var latent = Shape("noise");
                var decoded = Shape("decoded");
                return new Dictionary<string, long>
                {
                    [VideoDiffusionGraphs.FramesDimension] = latent[1],
                    [VideoDiffusionGraphs.Height] = decoded[2],
                    [VideoDiffusionGraphs.Width] = decoded[3],
                };
            }
        }

        public int[] Shape(string name) => [.. _index.GetProperty("tensors").GetProperty(name).EnumerateArray().Select(e => e.GetInt32())];

        public float[] Tensor(string name) => Env.ReadFloats(Path.Combine(folder, name + ".f32"));

        public List<ImageInput> InputFrames()
        {
            var shape = Shape("input_frames");
            var bytes = File.ReadAllBytes(Path.Combine(folder, "input_frames.u8"));
            int frameBytes = shape[1] * shape[2] * 3;
            return [.. Enumerable.Range(0, shape[0]).Select(f => new ImageInput(shape[2], shape[1], bytes.AsSpan(f * frameBytes, frameBytes).ToArray()))];
        }

        private static JsonElement ReadIndex(string folder)
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(folder, "index.json")));
            return document.RootElement.Clone();
        }
    }
}
