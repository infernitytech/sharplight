using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// Two compiled models on one CUDA device used from two threads, as a server serving two models on one GPU does: one captures a graph per
/// new step shape while the other uploads inputs and reads outputs too large for the page-locked staging buffers. A copy on the legacy
/// stream (<c>cuMemcpyHtoD</c>) would wait for every blocking stream of the context and fail while the other thread captures
/// (CUDA_ERROR_STREAM_CAPTURE_IMPLICIT).
/// </summary>
public sealed class CudaStreamConcurrencyTests
{
    [Fact]
    public void LargeCopiesOfOneModelRunWhileAnotherCapturesGraphs()
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        using var device = provider.OpenDevice(provider.GetDevices()[0]);
        Assert.SkipUnless(((CudaDevice)device).LaunchGraphs, "Launch graphs are disabled.");

        // The capturing model: a few launches per step; each new row count is recorded at its first run and captured at its second.
        const int captureWidth = 256, captureShapes = 400;
        var capturing = new TG("cuda-capturing");
        var cx = capturing.InF("x", TG.T, captureWidth);
        var cy = capturing.B.MatMul(cx, capturing.RandW("w", DataType.F32, 0.05f, captureWidth, captureWidth));
        capturing.B.Output("y", capturing.B.Mul(capturing.B.Silu(cy), cx));
        var captureGraph = capturing.B.Build();

        // The copying model: 2 MiB of input and output, beyond the 1 MiB staging buffers, so both go through the unstaged copies.
        const int rows = 512, width = 1024;
        var copying = new TG("cuda-copying");
        var x = copying.InF("x", TG.T, width);
        copying.B.Output("y", copying.B.Silu(copying.B.MatMul(x, copying.RandW("w", DataType.F32, 0.05f, width, width))));
        var copyGraph = copying.B.Build();

        using var capturer = Compile(captureGraph, device, captureShapes);
        using var copier = Compile(copyGraph, device, rows);
        var input = new float[rows * width];
        for (int i = 0; i < input.Length; i++)
        {
            input[i] = MathF.Sin(i * 0.37f);
        }

        var copyParameters = new RunParameters();
        copyParameters.Symbols["T"] = rows;
        copier.SetInput(copyGraph.Inputs.Single(), MemoryMarshal.AsBytes(input.AsSpan()));
        copier.Run(copyParameters);
        var expected = new float[rows * width];
        copier.GetOutput<float>("y", expected);

        var errors = new System.Collections.Concurrent.ConcurrentQueue<Exception>();
        int capturesDone = 0;
        var capture = new Thread(() =>
        {
            try
            {
                var small = new float[captureShapes * captureWidth];
                var output = new float[captureShapes * captureWidth];
                var parameters = new RunParameters();
                for (int t = 1; t <= captureShapes; t++)
                {
                    parameters.Symbols["T"] = t;
                    for (int run = 0; run < 2; run++)
                    {
                        capturer.SetInput(captureGraph.Inputs.Single(), MemoryMarshal.AsBytes(small.AsSpan(0, t * captureWidth)));
                        capturer.Run(parameters);
                        capturer.GetOutput<float>("y", output.AsSpan(0, t * captureWidth));
                    }
                }
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
            finally
            {
                Volatile.Write(ref capturesDone, 1);
            }
        });
        capture.Start();

        int mismatches = 0;
        var actual = new float[rows * width];
        try
        {
            while (Volatile.Read(ref capturesDone) == 0)
            {
                copier.SetInput(copyGraph.Inputs.Single(), MemoryMarshal.AsBytes(input.AsSpan()));
                copier.Run(copyParameters);
                copier.GetOutput<float>("y", actual);
                mismatches += actual.AsSpan().SequenceEqual(expected) ? 0 : 1;
            }
        }
        catch (Exception e)
        {
            errors.Enqueue(e);
        }

        capture.Join();
        Assert.Empty(errors);
        Assert.Equal(0, mismatches);
    }

    private static CompiledModel Compile(ComputeGraph graph, IComputeDevice device, int maxTokens) => CompiledModel.Compile(graph, new PlacementOptions
    {
        Devices = [new DeviceSlot { Device = device }],
        Compile = new CompileOptions { MaxTokens = maxTokens, ContextLength = maxTokens },
        ReservePerDevice = 0,
    });
}
