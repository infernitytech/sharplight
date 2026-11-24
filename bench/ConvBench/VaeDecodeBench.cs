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
/// GPU times of one more run.
/// </summary>
internal static class VaeDecodeBench
{
    public static void Run(ModelLoader loader, IComputeDevice device, string modelPath, int size, bool strict, int runs, int top)
    {
        using var model = TaskModel.Load(loader, new TaskLoadOptions
        {
            ModelPath = modelPath,
            Devices = [new DeviceSlot { Device = device }],
            AllowReducedPrecision = !strict,
        });
        var dimensions = new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
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
        var times = new List<double>();
        for (int i = 0; i < runs; i++)
        {
            var stopwatch = Stopwatch.StartNew();
            compiled.Run(parameters);
            var image = new float[compiled.OutputShape(DiffusionGraphs.Image, parameters).Aggregate(1L, (a, b) => a * b)];
            compiled.GetOutput<float>(DiffusionGraphs.Image, image);
            times.Add(stopwatch.Elapsed.TotalMilliseconds);
        }

        times.Sort();
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"VAE {Path.GetFileName(modelPath.TrimEnd('/'))} decode {size}x{size}: median {times[times.Count / 2]:F1} ms, min {times[0]:F1} ms ({runs} runs)"));
        if (top <= 0)
        {
            return;
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
    }
}
