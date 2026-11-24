using System.Runtime.InteropServices;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Sharplight.Vulkan.Tests;

namespace Sharplight.Cuda.Tests;

/// <summary>
/// The channels-last convolution (conv_cl) streams its tiles through a ring of asynchronous copies on devices of compute capability
/// 8.0 and later. With the same products in the same order as the register-prefetch form (compiled here by turning off
/// <see cref="DeviceFeatures.NativeGemmF16"/>, which the asynchronous form needs), both must give the same bits; both must also match
/// the f32 CPU convolution to the f16 rounding of their operands. Shapes: partial tiles in both directions, a K that is not a multiple
/// of the 32-value step (24 channels, 5x5), a K split, two images and a fused residual add.
/// </summary>
public sealed class ConvAsyncCopyTests
{
    [Theory]
    [InlineData(64, 20, 96, 3, false)]
    [InlineData(24, 30, 64, 5, false)]
    [InlineData(128, 40, 256, 3, true)]
    [InlineData(256, 9, 128, 3, false)]
    public void AsyncCopiesMatchRegisterPrefetch(int c, int hw, int o, int k, bool residual)
    {
        var provider = new CudaProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "No CUDA device.");
        using var device = provider.OpenDevice(provider.GetDevices()[0]);
        Assert.SkipUnless(device.Info.Features.Contains(DeviceFeatures.NativeGemmF16), "The device has no asynchronous copies (compute capability below 8.0).");
        var tg = new TG("conv-async");
        var x = tg.InF("x", Dim.Of(2), c, hw, hw);
        var r = residual ? tg.InF("r", Dim.Of(2), o, hw, hw) : null;
        var attributes = Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { k / 2, k / 2, k / 2, k / 2 }), ("dilations", new long[] { 1, 1 }),
            ("groups", 1L));
        var y = tg.B.Emit1("Conv", [x, tg.RandW("w", DataType.F32, 1f / MathF.Sqrt(c * k * k), o, c, k, k), tg.RandW("b", DataType.F32, 0.5f, o)], attributes);
        tg.B.Output("y", r is null ? y : tg.B.Add(y, r));
        var graph = tg.B.Build();
        var input = Rnd.F(2L * c * hw * hw, 11);
        var residualInput = Rnd.F(2L * o * hw * hw, 12);

        var asynchronous = Run(device, graph, x, input, r, residualInput, reduced: true, disabled: []);
        var prefetch = Run(device, graph, x, input, r, residualInput, reduced: true, disabled: [DeviceFeatures.NativeGemmF16]);
        var cpuProvider = new CpuExecutionProvider();
        using var cpu = cpuProvider.OpenDevice(cpuProvider.GetDevices()[0]);
        var oracle = Run(cpu, graph, x, input, r, residualInput, reduced: false, disabled: []);
        Assert.Equal(prefetch, asynchronous);
        for (int i = 0; i < oracle.Length; i++)
        {
            Assert.True(Math.Abs(asynchronous[i] - oracle[i]) <= 2e-2 + 1e-2 * Math.Abs(oracle[i]), $"element {i}: {asynchronous[i]} vs {oracle[i]}");
        }
    }

    private static float[] Run(IComputeDevice device, ComputeGraph graph, Value x, float[] input, Value? r, float[] residual, bool reduced, HashSet<string> disabled)
    {
        using var model = CompiledModel.Compile(graph, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = device }],
            Compile = new CompileOptions { MaxTokens = 2, AllowReducedPrecision = reduced, DisabledFeatures = disabled },
            ReservePerDevice = 0,
        });
        model.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan()));
        if (r is not null)
        {
            model.SetInput(r, MemoryMarshal.AsBytes(residual.AsSpan()));
        }

        model.Run(new RunParameters());
        var output = new float[graph.Outputs.Single().Value.Shape.Elements()];
        model.GetOutput<float>("y", output);
        return output;
    }
}
