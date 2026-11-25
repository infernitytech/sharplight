using System.Diagnostics;
using System.Globalization;
using Sharplight.Cuda;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vulkan;

/// <summary>
/// <c>--vae &lt;model&gt; --size N</c>: an image-generation model's VAE decoder at N x N pixels (one image of random latents), compiled once;
/// one warm-up run, then the median wall time of the timed runs (each ends with the device idle) and, with <c>--top</c>, the per-kernel
/// GPU times of one more run. The activation precision is measured first as the image generator does (<c>auto</c>), or left at f16
/// (<c>half</c>).
/// </summary>
internal static class VaeDecodeBench
{
    public static void Run(ModelLoader loader, IComputeDevice device, string modelPath, int size, bool strict, int runs, int top, bool check,
        string activations, int probeSize)
    {
        float[] image = Decode(loader, device, modelPath, size, strict, runs, top, activations, probeSize);
        if (!check || strict)
        {
            return;
        }

        // The same latents decoded with f32 products: how far the reduced-precision decoder is from it (images in [-1, 1]).
        float[] reference = Decode(loader, device, modelPath, size, strict: true, runs: 0, top: 0, "half", probeSize);
        double squared = 0;
        double largest = 0;
        for (int i = 0; i < image.Length; i++)
        {
            double d = image[i] - reference[i];
            squared += d * d;
            largest = Math.Max(largest, Math.Abs(d));
        }

        double psnr = 10 * Math.Log10(4 / Math.Max(squared / image.Length, 1e-30));
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"VAE reduced vs f32 products: PSNR {psnr:F1} dB (range 2), max abs {largest:E2}"));
    }

    private static float[] Decode(ModelLoader loader, IComputeDevice device, string modelPath, int size, bool strict, int runs, int top,
        string activations, int probeSize)
    {
        using var model = TaskModel.Load(loader, new TaskLoadOptions
        {
            ModelPath = modelPath,
            Devices = [new DeviceSlot { Device = device }],
            AllowReducedPrecision = !strict,
            Log = activations == "auto" ? Console.WriteLine : null,
        });
        var dimensions = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        if (activations == "auto")
        {
            var stopwatch = Stopwatch.StartNew();
            var probeDimensions = new Dictionary<string, long> { [DiffusionGraphs.Height] = probeSize, [DiffusionGraphs.Width] = probeSize };
            var precision = model.CalibrateActivationPrecision(ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, probeDimensions, (probe, parameters) =>
            {
                parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
                var input = probe.Graph.Inputs.Single(i => i.Name == DiffusionGraphs.Latent);
                var noise = new Random(5);
                var latents = new float[input.Shape.Resolve(parameters.Symbols).Aggregate(1L, (a, b) => a * b)];
                for (int i = 0; i < latents.Length; i++)
                {
                    // Unit Gaussian latents (Box-Muller), as the denoiser produces.
                    latents[i] = (float)(Math.Sqrt(-2 * Math.Log(1 - noise.NextDouble())) * Math.Cos(2 * Math.PI * noise.NextDouble()));
                }

                probe.SetInput<float>(input, latents);
            });
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"activation precision {precision} (measured in {stopwatch.ElapsedMilliseconds} ms)"));
        }

        using var compiled = model.CompileComponent(ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode,
            new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = 1 }, dimensions);
        var parameters = new RunParameters();
        var latent = compiled.Graph.Inputs.Single(i => i.Name == DiffusionGraphs.Latent);
        if (latent.Shape[0].IsSymbolic)
        {
            parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
        }

        var random = new Random(5);
        var values = new float[latent.Shape.Dims.Skip(1).Aggregate(1L, (a, d) => a * d.Fixed)];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)(random.NextDouble() * 2 - 1);
        }

        compiled.SetInput<float>(latent, values);
        compiled.Run(parameters);
        var image = new float[compiled.OutputShape(DiffusionGraphs.Image, parameters).Aggregate(1L, (a, b) => a * b)];
        compiled.GetOutput<float>(DiffusionGraphs.Image, image);
        var times = new List<double>();
        for (int i = 0; i < runs; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            compiled.Run(parameters);
            compiled.GetOutput<float>(DiffusionGraphs.Image, image);
            times.Add(stopwatch.Elapsed.TotalMilliseconds);
        }

        if (runs > 0)
        {
            times.Sort();
            Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
                $"VAE {Path.GetFileName(modelPath.TrimEnd('/'))} decode {size}x{size}: median {times[times.Count / 2]:F1} ms, min {times[0]:F1} ms ({runs} runs)"));
        }

        if (top <= 0)
        {
            return image;
        }

        foreach (var segment in compiled.Segments)
        {
            switch (segment)
            {
                case CudaSegment cuda:
                    cuda.Profile = true;
                    break;
                case VulkanSegment vulkan:
                    vulkan.Profile = true;
                    break;
            }
        }

        compiled.Run(parameters);
        foreach (var segment in compiled.Segments)
        {
            Console.WriteLine(segment switch
            {
                CudaSegment cuda => cuda.ProfileReport(top),
                VulkanSegment vulkan => vulkan.ProfileReport(top),
                _ => "",
            });
        }

        return image;
    }
}
