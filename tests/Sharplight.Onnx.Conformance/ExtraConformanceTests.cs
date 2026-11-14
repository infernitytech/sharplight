using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Conformance;

/// <summary>
/// Extra node-format tests for operators the ONNX backend tests barely cover (ai.onnx.ml, string ops, com.microsoft contrib ops),
/// written by <c>tools/gen-onnx-ml-tests.py</c> and <c>tools/gen-onnx-contrib-tests.py</c> with onnxruntime outputs (see
/// <see cref="ExtraTestSuite"/> for where they are found). Every test must pass on every provider; there is no skip list.
/// </summary>
public sealed class ExtraConformanceTests
{
    public static TheoryData<string> Tests()
    {
        var data = new TheoryData<string>();
        foreach (var name in ExtraTestSuite.Names())
        {
            data.Add(name);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Tests))]
    public void Cpu(string test) => Check(test, Devices.Cpu, null);

    [Theory]
    [MemberData(nameof(Tests))]
    public void Vulkan(string test)
    {
        var (device, reason) = Devices.Vulkan;
        if (device is null)
        {
            Assert.Skip("Vulkan unavailable: " + reason);
        }

        Check(test, device, Devices.Cpu);
    }

    private static void Check(string test, IComputeDevice device, IComputeDevice? fallback)
    {
        var outcome = NodeTestRunner.Run(ExtraTestSuite.DirectoryOf(test), device, fallback);
        Assert.True(outcome.Status == NodeTestStatus.Pass, $"{outcome.Status}: {outcome.Message}");
    }
}
