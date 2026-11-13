using Sharplight.Tensors;

namespace Sharplight.Graph;

// Token sampling over logits rows (see SamplingHead).
public static partial class StandardOps
{
    /// <summary>
    /// Chooses one token per logits row on the device. Inputs: <c>logits</c> [R, V] f32 and <c>params</c> [R, <see cref="SamplingRow.Width"/>] i32,
    /// one encoded <see cref="SamplingRow"/> per row (greedy arg-max, or temperature / top-k / top-p / min-p sampling with a counter-based
    /// random generator keyed by the row's seed and counter, so the choice is a pure function of logits and parameters).
    /// Outputs: <c>tokens</c> [R] i32 (-1 for rows whose parameters say skip) and <c>logprobs</c> [R] f32, the log-softmax of the raw
    /// logits at the chosen token when the row asks for it (0 otherwise). See <see cref="SamplingHead"/> for the exact filter semantics.
    /// </summary>
    public static readonly OpSchema Sample = new("Sample", OpCategory.Reduction, 2, 2, (i, a) =>
    {
        var logits = i[0];
        var parameters = i[1];
        if (logits.Rank != 2 || logits.Type != DataType.F32)
        {
            throw new ArgumentException("Sample logits must be f32 [R, V].");
        }

        if (parameters.Rank != 2 || parameters.Type != DataType.I32 || parameters.Shape[1] != SamplingRow.Width || parameters.Shape[0] != logits.Shape[0])
        {
            throw new ArgumentException($"Sample params must be i32 [R, {SamplingRow.Width}] with the rows of the logits.");
        }

        var rows = new Shape(logits.Shape[0]);
        return [new TensorType(DataType.I32, rows), new TensorType(DataType.F32, rows)];
    });
}
