using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// The expert banks of a streamed segment whose experts can be copied apart (<see cref="IStreamedSegment.BeginStreaming(ReadOnlySpan{int})"/>):
/// the weights that are input 1 ([experts, ...]) of its <c>MatMulId</c>s, when every <c>MatMulId</c> of the segment reads the same ids input.
/// </summary>
internal static class StreamedExpertBanks
{
    /// <summary>Per bank weight, its experts; empty when the segment's expert products read more than one ids value or none.</summary>
    public static IReadOnlyDictionary<WeightInfo, int> Of(GraphSegment segment)
    {
        var products = segment.Nodes.Where(n => n.Op.Name == "MatMulId").ToList();
        if (products.Select(n => n.Inputs[2]).Distinct().Count() != 1 || !segment.Inputs.Contains(products[0].Inputs[2]))
        {
            return new Dictionary<WeightInfo, int>();
        }

        var banks = new Dictionary<WeightInfo, int>();
        foreach (var product in products)
        {
            if (product.Inputs[1] is { Weight: { } weight, Shape: { Rank: >= 2 } shape } && !shape[0].IsSymbolic)
            {
                banks[weight] = (int)shape[0].Fixed;
            }
        }

        return banks;
    }
}
