using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Samples;

namespace Sharplight.OnnxRun;

/// <summary>Fills every graph input with float32 data: the first from a file when given, all others (and the first without a file) with reproducible random values.</summary>
internal sealed class InputBinder
{
    private const int RandomSeed = 1234;

    private readonly Random _random = new(RandomSeed);

    public void Bind(CompiledModel compiled, ComputeGraph graph, RunParameters run, string? inputFile)
    {
        bool first = true;
        foreach (var input in graph.Inputs)
        {
            long count = input.Shape.Elements(run.Symbols);
            var data = first && inputFile is not null
                ? ReadFile(inputFile, count, input)
                : CreateRandom(count);
            first = false;
            compiled.SetInput<float>(input, data);
        }
    }

    private static float[] ReadFile(string path, long elementCount, Value input)
    {
        if (!File.Exists(path))
        {
            throw new SampleException($"Input file '{path}' does not exist.", ExitCodes.InvalidInput);
        }

        var bytes = File.ReadAllBytes(path);
        long expectedBytes = elementCount * sizeof(float);
        if (bytes.Length != expectedBytes)
        {
            throw new SampleException($"{path}: {bytes.Length} bytes, expected {expectedBytes} for {input}.", ExitCodes.InvalidInput);
        }

        return [.. MemoryMarshal.Cast<byte, float>(bytes)];
    }

    private float[] CreateRandom(long count)
    {
        var data = new float[count];
        for (long i = 0; i < count; i++)
        {
            // Uniform in [-1, 1).
            data[i] = (float)(_random.NextDouble() * 2 - 1);
        }

        return data;
    }
}
