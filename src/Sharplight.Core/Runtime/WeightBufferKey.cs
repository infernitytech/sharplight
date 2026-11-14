using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Identity of a device-resident weight buffer in <see cref="SharedWeights"/>: the tensor's bytes (its <see cref="IWeightSource"/>
/// instance, so renamed or reshaped views of one tensor, e.g. a tied embedding used as the LM head, are the same tensor), its
/// storage type and size, and the provider-defined device layout (empty for the bytes as stored; a repacked or converted copy
/// names its transform).
/// </summary>
/// <param name="Source">Where the bytes come from (compared by reference).</param>
/// <param name="Type">Storage type.</param>
/// <param name="ByteLength">Bytes of the tensor as stored.</param>
/// <param name="Layout">Provider-defined name of the device layout ("" = as stored).</param>
public readonly record struct WeightBufferKey(IWeightSource Source, DataType Type, long ByteLength, string Layout)
{
    /// <summary>The key of <paramref name="weight"/> in device layout <paramref name="layout"/>.</summary>
    public static WeightBufferKey For(WeightInfo weight, string layout = "") => new(weight.Source, weight.Type, weight.ByteSize, layout);
}
