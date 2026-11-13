using System.Runtime.InteropServices;
using Sharplight.Graph;

namespace Sharplight.OnnxRun;

/// <summary>Prints the best-scoring elements of the first batch item of an output, and optionally stores the whole output.</summary>
internal static class OutputReporter
{
    private const int TopCount = 5;

    public static void Report(string name, Value output, float[] result, int batchSize, string? outputFile, TextWriter writer)
    {
        if (outputFile is not null)
        {
            File.WriteAllBytes(outputFile, MemoryMarshal.AsBytes<float>(result).ToArray());
        }

        writer.WriteLine($"output '{name}' {output.Shape}: top-{TopCount} of batch item 0");
        int itemLength = result.Length / batchSize;
        var best = Enumerable.Range(0, itemLength)
            .OrderByDescending(index => result[index])
            .Take(TopCount);
        foreach (int index in best)
        {
            writer.WriteLine($"  [{index}] {result[index]:F5}");
        }
    }
}
