using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>Runs one component graph of a task model with named inputs.</summary>
internal static class GraphRunner
{
    /// <summary>Compiles the graph, binds the inputs (f32 or i32 arrays), runs with the batch symbol and returns one output.</summary>
    public static float[] Run(TaskModel model, string role, string graph, string output, long batch, params (string Name, Array Data)[] inputs)
        => Run(model, role, graph, output, batch, null, inputs);

    /// <summary>
    /// As <see cref="Run(TaskModel, string, string, string, long, ValueTuple{string, Array}[])"/>, with the graph built for an image size
    /// (<paramref name="dimensions"/>: height and width in pixels) instead of the native one.
    /// </summary>
    public static float[] Run(TaskModel model, string role, string graph, string output, long batch, IReadOnlyDictionary<string, long>? dimensions,
        params (string Name, Array Data)[] inputs)
    {
        using var compiled = model.CompileComponent(role, graph, new Dictionary<string, long> { [DiffusionGraphs.BatchSymbol] = Math.Max(batch, 1) }, dimensions);
        foreach (var (name, data) in inputs)
        {
            var value = compiled.Graph.Inputs.Single(i => i.Name == name);
            if (data is int[] ints)
            {
                compiled.SetInput<int>(value, ints);
            }
            else
            {
                compiled.SetInput<float>(value, (float[])data);
            }
        }

        var parameters = new RunParameters();
        if (compiled.Graph.Inputs.Any(i => i.Shape.Rank > 0 && i.Shape[0].IsSymbolic))
        {
            parameters.Symbols[DiffusionGraphs.BatchSymbol] = batch;
        }

        compiled.Run(parameters);
        var shape = compiled.OutputShape(output, parameters);
        var result = new float[shape.Aggregate(1L, (a, b) => a * b)];
        compiled.GetOutput<float>(output, result);
        return result;
    }

    /// <summary>Runs the text encoder on one token sequence.</summary>
    public static float[] Text(TaskModel model, int[] tokens)
        => Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0, (DiffusionGraphs.Tokens, tokens));
}
