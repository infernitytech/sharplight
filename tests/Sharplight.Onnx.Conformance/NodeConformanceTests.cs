using System.Text.Json;
using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Onnx.Conformance;

/// <summary>
/// The ONNX backend node tests as a conformance suite: every test must pass on every provider unless conformance-skips.txt lists it
/// with a reason. Set <c>SHARPLIGHT_CONFORMANCE_REPORT=&lt;file&gt;</c> to write the outcome of every test (including skipped ones) as JSON lines.
/// </summary>
public sealed class NodeConformanceTests
{
    public static TheoryData<string> Tests()
    {
        var data = new TheoryData<string>();
        foreach (var name in NodeTestSuite.Names())
        {
            data.Add(name);
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Tests))]
    public void Cpu(string test) => Check("cpu", test, Devices.Cpu, null);

    [Theory]
    [MemberData(nameof(Tests))]
    public void Vulkan(string test)
    {
        var (device, reason) = Devices.Vulkan;
        if (device is null)
        {
            Assert.Skip("Vulkan unavailable: " + reason);
        }

        Check("vulkan", test, device, Devices.Cpu);
    }

    /// <summary>Runs every node test on the providers in <c>SHARPLIGHT_CONFORMANCE_PROVIDERS</c> (default "cpu") and writes the report.</summary>
    [Fact]
    public void Report()
    {
        string path = Environment.GetEnvironmentVariable("SHARPLIGHT_CONFORMANCE_REPORT") ?? "";
        if (path.Length == 0)
        {
            Assert.Skip("Set SHARPLIGHT_CONFORMANCE_REPORT to write a conformance report.");
        }

        var providers = (Environment.GetEnvironmentVariable("SHARPLIGHT_CONFORMANCE_PROVIDERS") ?? "cpu").Split(',', StringSplitOptions.RemoveEmptyEntries);
        File.WriteAllLines(path + ".supported", OnnxTranslator.SupportedOperators);
        using var writer = new StreamWriter(path);
        foreach (var provider in providers)
        {
            var (device, fallback) = provider == "cpu" ? (Devices.Cpu, null) : (Devices.Vulkan.Device!, Devices.Cpu);
            foreach (var test in NodeTestSuite.Names())
            {
                var outcome = NodeTestRunner.Run(NodeTestSuite.DirectoryOf(test), device, fallback);
                writer.WriteLine(ReportLine(test, provider, outcome));
                writer.Flush();
            }
        }
    }

    private static string ReportLine(string test, string provider, NodeTestOutcome outcome)
    {
        using var buffer = new MemoryStream();
        using (var json = new Utf8JsonWriter(buffer))
        {
            json.WriteStartObject();
            json.WriteString("test", test);
            json.WriteString("provider", provider);
            json.WriteString("status", outcome.Status.ToString());
            json.WriteString("message", outcome.Message);
            json.WriteString("skip", SkipList.Reason(provider, test));
            json.WriteBoolean("pinned", outcome.Pinned.Length > 0);
            json.WriteStartArray("ops");
            foreach (var op in OpsOf(test))
            {
                json.WriteStringValue(op);
            }

            json.WriteEndArray();
            json.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static void Check(string provider, string test, IComputeDevice device, IComputeDevice? fallback)
    {
        if (SkipList.Reason(provider, test) is { } reason)
        {
            Assert.Skip(reason);
        }

        var outcome = NodeTestRunner.Run(NodeTestSuite.DirectoryOf(test), device, fallback);
        Assert.True(outcome.Status == NodeTestStatus.Pass, $"{outcome.Status}: {outcome.Message}");
    }

    private static string[] OpsOf(string test)
    {
        try
        {
            var model = OnnxParser.Parse(File.ReadAllBytes(Path.Combine(NodeTestSuite.DirectoryOf(test), "model.onnx")));
            return model.Graph.Nodes.Select(n => n.Domain is "" or "ai.onnx" ? n.OpType : $"{n.Domain}.{n.OpType}").Distinct().ToArray();
        }
        catch (Exception)
        {
            return [];
        }
    }
}
