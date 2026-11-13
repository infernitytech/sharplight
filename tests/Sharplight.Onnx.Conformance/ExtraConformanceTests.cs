using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Conformance;

/// <summary>
/// Extra node-format tests for operators the ONNX backend tests barely cover (ai.onnx.ml, string ops), written by
/// <c>tools/gen-onnx-ml-tests.py</c> with onnxruntime outputs into <c>SHARPLIGHT_ONNX_EXTRA_TESTS</c>, else
/// <c>~/models/onnx/sharplight-ml-tests</c>. Every test must pass on every provider; there is no skip list.
/// </summary>
public sealed class ExtraConformanceTests
{
    private static readonly Lazy<string?> s_root = new(FindRoot);

    public static TheoryData<string> Tests()
    {
        var data = new TheoryData<string>();
        if (s_root.Value is { } root)
        {
            foreach (var directory in Directory.GetDirectories(root).Where(d => File.Exists(Path.Combine(d, "model.onnx"))).Order(StringComparer.Ordinal))
            {
                data.Add(Path.GetFileName(directory));
            }
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
        var outcome = NodeTestRunner.Run(Path.Combine(s_root.Value!, test), device, fallback);
        Assert.True(outcome.Status == NodeTestStatus.Pass, $"{outcome.Status}: {outcome.Message}");
    }

    private static string? FindRoot()
    {
        string root = Environment.GetEnvironmentVariable("SHARPLIGHT_ONNX_EXTRA_TESTS") is { Length: > 0 } configured
            ? configured
            : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx", "sharplight-ml-tests");
        return Directory.Exists(root) ? root : null;
    }
}
