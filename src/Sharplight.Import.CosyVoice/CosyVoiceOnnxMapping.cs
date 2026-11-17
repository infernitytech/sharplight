using System.Runtime.InteropServices;
using Sharplight.Import.Onnx;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.CosyVoice;

/// <summary>
/// Names the parameters of the release's ONNX exports (the S3 tokenizer v3 and CAM++) by the module path of the node that reads them: the exporter
/// gives folded and transposed tensors generated names (<c>onnx::MatMul_3998</c>), but its node names keep the PyTorch module path
/// (<c>/blocks.0/attn/query/MatMul</c>, <c>/head/layer1/layer1.0/conv1/Conv</c>). MatMul weights are stored <c>[in, out]</c> and are transposed to the
/// <c>[out, in]</c> layout of a linear layer; convolutions with a batch norm folded in by the exporter keep the bias it produced.
/// </summary>
internal static class CosyVoiceOnnxMapping
{
    /// <summary>The parameters of an ONNX export under PyTorch module names: initializers named by the exporter keep their names, the others are named by their node.</summary>
    /// <param name="source">The source opened by <see cref="OnnxImporter.Open"/>.</param>
    /// <returns>Module-path names (<c>blocks.0.attn.query.weight</c>) and weights.</returns>
    public static Dictionary<string, WeightInfo> Map(ModelSource source)
    {
        var result = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, weight) in source.Weights)
        {
            if (!IsGenerated(name))
            {
                result[name] = weight;
            }
        }

        foreach (var node in OnnxImporter.NodeParameters(source))
        {
            string module = ModulePath(node.Name);
            if (module.Length == 0)
            {
                continue;
            }

            switch (node.OpType)
            {
                case "Conv":
                    AddGenerated(result, module + ".weight", node.Inputs.ElementAtOrDefault(1), transpose: false);
                    AddGenerated(result, module + ".bias", node.Inputs.ElementAtOrDefault(2), transpose: false);
                    break;
                case "MatMul":
                    AddGenerated(result, module + ".weight", node.FirstTensor, transpose: true);
                    break;
                case "Add":
                    AddGenerated(result, module + ".bias", node.FirstTensor, transpose: false);
                    break;
                case "Mul":
                    AddGenerated(result, module + ".weight", node.FirstTensor, transpose: false);
                    break;
            }
        }

        return result;
    }

    /// <summary>
    /// The module path of a node name: <c>/blocks.0/attn/query/MatMul</c> → <c>blocks.0.attn.query</c>; a segment that the next one repeats as its
    /// prefix (a container and its child, <c>layer1/layer1.0</c>) is dropped.
    /// </summary>
    /// <param name="nodeName">The node name.</param>
    public static string ModulePath(string nodeName)
    {
        var segments = nodeName.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 2)
        {
            return "";
        }

        var kept = new List<string>();
        for (int i = 0; i < segments.Length - 1; i++)
        {
            if (i + 1 < segments.Length - 1 && segments[i + 1].StartsWith(segments[i] + ".", StringComparison.Ordinal))
            {
                continue;
            }

            kept.Add(segments[i]);
        }

        return string.Join('.', kept);
    }

    private static bool IsGenerated(string name) => name.StartsWith("onnx::", StringComparison.Ordinal);

    private static void AddGenerated(Dictionary<string, WeightInfo> result, string name, WeightInfo? weight, bool transpose)
    {
        if (weight is null || !IsGenerated(weight.Name) || weight.Shape.Elements() <= 1 || result.ContainsKey(name))
        {
            return;
        }

        result[name] = transpose && weight.Shape.Rank == 2 ? Transposed(weight, name) : weight.As(name);
    }

    // A [rows, columns] f32 matrix as [columns, rows].
    private static WeightInfo Transposed(WeightInfo weight, string name)
    {
        if (weight.Type != DataType.F32)
        {
            throw new InvalidDataException($"{weight.Name}: only f32 matrices are transposed ({weight.Type}).");
        }

        int rows = (int)weight.Shape[0].Fixed;
        int columns = (int)weight.Shape[1].Fixed;
        return WeightInfo.Derived(name, DataType.F32, new Shape(columns, rows), () =>
        {
            var source = MemoryMarshal.Cast<byte, float>(weight.ReadAll().AsSpan());
            var result = new float[source.Length];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < columns; c++)
                {
                    result[(c * rows) + r] = source[(r * columns) + c];
                }
            }

            return MemoryMarshal.AsBytes(result.AsSpan()).ToArray();
        });
    }
}
