using Sharplight.Tensors;

namespace Sharplight.Graph;

// Random number generation. The generator is deterministic and counter based (Runtime.CounterRandom): element i of the output depends only
// on the attribute "seed" and i, so a seeded graph gives the same values on every provider and in every run.
public static partial class StandardOps
{
    /// <summary>
    /// Random f32 tensor. Without inputs the shape is the attribute <c>shape</c> (long[]); with one input the output has the input's shape
    /// (the input's values are not read). Attributes: <c>distribution</c> ("uniform": values in [low, high); "normal": mean + scale · N(0, 1)),
    /// <c>low</c>/<c>high</c> or <c>mean</c>/<c>scale</c> (float), <c>seed</c> (long).
    /// </summary>
    public static readonly OpSchema Random = new("Random", OpCategory.Data, 0, 1, (i, a) =>
        [new(DataType.F32, i.Count == 1 ? i[0].Shape : new Shape(a.Get<long[]>("shape").Select(d => (Dim)d).ToArray()))]);

    /// <summary>
    /// Draws <c>sample_size</c> class indices per row from the categorical distribution softmax(x): x f32 <c>[batch, classes]</c>
    /// (unnormalized log-probabilities) → i32 <c>[batch, sample_size]</c>. Attributes: <c>sample_size</c> (long), <c>seed</c> (long).
    /// Sample s of row b uses uniform draw <c>b · sample_size + s</c> (inverse CDF).
    /// </summary>
    public static readonly OpSchema Multinomial = new("Multinomial", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank != 2)
        {
            throw new ArgumentException($"Multinomial expects x [batch, classes], got {x}.");
        }

        return [new(DataType.I32, new Shape(x[0], a.Get<long>("sample_size")))];
    });
}
