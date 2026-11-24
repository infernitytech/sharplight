using Sharplight.Cuda;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.VideoGeneration;
using Sharplight.Vulkan;

/// <summary>
/// Per-kernel GPU times of the model's graphs at one size (<c>--profile</c>): each graph is compiled alone, run once to warm up, then once
/// with per-launch timing, and its heaviest launches printed.
/// </summary>
internal static class GraphProfiler
{
    private static readonly (string Role, string Graph)[] s_graphs =
    [
        (ModelComponentRoles.Vae, "vae_encode"),
        (ModelComponentRoles.Transformer, "restore"),
        (ModelComponentRoles.Vae, "vae_decode"),
        (ModelComponentRoles.Vae, "color_correct"),
    ];

    public static void Run(TaskModel model, int frames, int height, int width, int top)
    {
        int latentFrames = (frames - 1) / 4 + 1;
        var dimensions = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = latentFrames,
            [VideoDiffusionGraphs.Height] = height,
            [VideoDiffusionGraphs.Width] = width,
            ["half_caches"] = model.Options.AllowReducedPrecision ? 1 : 0,
        };
        foreach (var (role, graph) in s_graphs)
        {
            using var compiled = model.CompileComponent(role, graph, null, dimensions);
            var random = new Random(1);
            foreach (var input in compiled.Graph.Inputs)
            {
                var values = new float[input.Shape.Elements()];
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)(random.NextDouble() * 2 - 1);
                }

                if (input.Name == VideoDiffusionGraphs.Timestep)
                {
                    values[0] = 1000f;
                }

                compiled.SetInput<float>(input, values);
            }

            compiled.Run(new RunParameters());
            foreach (var segment in compiled.Segments)
            {
                SetProfile(segment, true);
            }

            var stopwatch = System.Diagnostics.Stopwatch.StartNew();
            compiled.Run(new RunParameters());
            Console.WriteLine($"== {graph} {width}x{height}x{frames}: {stopwatch.Elapsed.TotalSeconds:F2} s wall (profiled)");
            foreach (var segment in compiled.Segments)
            {
                Console.WriteLine(Report(segment, top));
            }
        }
    }

    /// <summary>Compiles each graph at one size and prints the device memory it holds (or the allocation failure), without running it.</summary>
    public static void Memory(TaskModel model, int frames, int height, int width)
    {
        var dimensions = new Dictionary<string, long>
        {
            [VideoDiffusionGraphs.FramesDimension] = (frames - 1) / 4 + 1,
            [VideoDiffusionGraphs.Height] = height,
            [VideoDiffusionGraphs.Width] = width,
            ["half_caches"] = model.Options.AllowReducedPrecision ? 1 : 0,
        };
        foreach (var (role, graph) in s_graphs)
        {
            try
            {
                using var compiled = model.CompileComponent(role, graph, null, dimensions);
                Console.WriteLine($"{graph} {width}x{height}x{frames}: {compiled.ResidentMemoryBytes / (1024.0 * 1024 * 1024):F2} GiB");
            }
            catch (Exception error) when (error is DeviceOutOfMemoryException or InsufficientMemoryException)
            {
                Console.WriteLine($"{graph} {width}x{height}x{frames}: {error.Message.Split('.')[0]}");
            }
        }
    }

    private static void SetProfile(ICompiledSegment segment, bool on)
    {
        switch (segment)
        {
            case CudaSegment cuda:
                cuda.ResetProfile();
                cuda.Profile = on;
                break;
            case VulkanSegment vulkan:
                vulkan.ResetProfile();
                vulkan.Profile = on;
                break;
        }
    }

    private static string Report(ICompiledSegment segment, int top) => segment switch
    {
        CudaSegment cuda => cuda.ProfileReport(top),
        VulkanSegment vulkan => vulkan.ProfileReport(top),
        _ => "(no profile for this provider)",
    };
}
