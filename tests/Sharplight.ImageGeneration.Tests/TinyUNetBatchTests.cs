using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny SD UNets at batches of one to seven rows: every row on Vulkan matches the CPU at the same batch, and every row matches the
/// same row run alone (odd batches once broke the Vulkan kernels).
/// </summary>
public sealed class TinyUNetBatchTests
{
    public static TheoryData<string, int> Runs()
    {
        var data = new TheoryData<string, int>();
        foreach (string family in new[] { "sd1", "sd2", "sdxl" })
        {
            foreach (int batch in new[] { 1, 2, 3, 4, 5, 7 })
            {
                data.Add(family, batch);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public void VulkanRowsMatchCpu(string family, int batch) => Check(family, batch, 0);

    public static TheoryData<string, int, int> SizedRuns()
    {
        var data = new TheoryData<string, int, int>();
        foreach (int size in new[] { 256, 512 })
        {
            foreach (int batch in new[] { 2, 3, 5 })
            {
                data.Add("sd1", batch, size);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(SizedRuns))]
    public void VulkanRowsMatchCpuAtImageSize(string family, int batch, int size) => Check(family, batch, size);

    private static void Check(string family, int batch, int size)
    {
        Assert.SkipUnless(Directory.Exists(Env.TinyRoot(family)), $"tiny {family} pipeline missing");
        var vulkan = Env.Vulkan();
        Assert.SkipWhen(vulkan is null, "no Vulkan device");
        var inputs = size == 0 ? Inputs(family, batch) : RandomInputs(family, batch, size);
        var dims = size == 0 ? null : new Dictionary<string, long> { [DiffusionGraphs.Height] = size, [DiffusionGraphs.Width] = size };
        float[] expected;
        using (var cpuModel = Env.Load(Env.TinyDiffusersOf(family), Env.Cpu()))
        {
            expected = GraphRunner.Run(cpuModel, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, batch, dims, inputs);
        }

        using var model = Env.Load(Env.TinyDiffusersOf(family), vulkan!);
        var output = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, batch, dims, inputs);
        Assert.Equal(expected.Length, output.Length);
        int rowSize = expected.Length / batch;
        for (int row = 0; row < batch; row++)
        {
            var (maxAbs, rms) = Env.Compare(output.AsSpan(row * rowSize, rowSize).ToArray(), expected.AsSpan(row * rowSize, rowSize).ToArray());
            Assert.True(rms < 1e-3, $"batch {batch}, row {row}: max {maxAbs}, relative RMS {rms}");
        }
    }

    /// <summary>The reference inputs (two rows) repeated to <paramref name="batch"/> rows, every row perturbed differently.</summary>
    private static (string Name, Array Data)[] Inputs(string family, int batch)
    {
        var result = new List<(string, Array)>();
        string[] names = family == "sdxl"
            ? [DiffusionGraphs.Latent, DiffusionGraphs.Timestep, DiffusionGraphs.Context, DiffusionGraphs.Pooled, DiffusionGraphs.SizeConditions]
            : [DiffusionGraphs.Latent, DiffusionGraphs.Timestep, DiffusionGraphs.Context];
        foreach (string name in names)
        {
            var source = Env.TinyTensorOf(family, "unet_" + ReferenceName(name));
            int rowSize = source.Length / 2;
            var values = new float[rowSize * batch];
            for (int row = 0; row < batch; row++)
            {
                for (int i = 0; i < rowSize; i++)
                {
                    float value = source[(row % 2) * rowSize + i];
                    values[row * rowSize + i] = name == DiffusionGraphs.Timestep || name == DiffusionGraphs.SizeConditions ? value + 37 * row : value * (1f + 0.1f * row);
                }
            }

            result.Add((name, values));
        }

        return [.. result];
    }

    /// <summary>Random latents for an image of <paramref name="size"/>² pixels with the reference context rows.</summary>
    private static (string Name, Array Data)[] RandomInputs(string family, int batch, int size)
    {
        var random = new Random(batch * 1000 + size);
        var latent = new float[batch * 4 * (size / 8) * (size / 8)];
        for (int i = 0; i < latent.Length; i++)
        {
            latent[i] = (float)(random.NextDouble() * 2 - 1);
        }

        var reference = Inputs(family, batch);
        return [(DiffusionGraphs.Latent, latent), .. reference.Where(r => r.Name != DiffusionGraphs.Latent)];
    }

    private static string ReferenceName(string input) => input switch
    {
        DiffusionGraphs.Latent => "latent",
        DiffusionGraphs.Timestep => "timesteps",
        DiffusionGraphs.Context => "context",
        DiffusionGraphs.Pooled => "pooled",
        DiffusionGraphs.SizeConditions => "time_ids",
        _ => input,
    };
}
