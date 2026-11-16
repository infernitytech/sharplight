using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// Per-token switched low-rank adapters: a projection <c>W</c> with stacked adapter pairs <c>A</c> [S, R, K] and <c>B</c> [S, N, R] (weights
/// <c>{name}.lora_a</c> / <c>{name}.lora_b</c> next to <c>{name}.weight</c>) adds <c>B[a] (A[a] x)</c> for every row, where <c>a</c> is the row's
/// adapter slot (two expert-indexed products). Slot 0 is conventionally the all-zero adapter, i.e. the base projection.
/// </summary>
internal static class SwitchedLora
{
    /// <summary>
    /// Adds the switched low-rank delta of the projection weight <paramref name="weightName"/> to <paramref name="projected"/>; returns it
    /// unchanged when the projection has no adapter pairs.
    /// </summary>
    /// <param name="b">Graph under construction.</param>
    /// <param name="projected">The base projection W·x [T, N].</param>
    /// <param name="x">The projection's input [T, K].</param>
    /// <param name="weightName">Canonical name of W (ending in <c>.weight</c>).</param>
    /// <param name="slots">Adapter slot of every row, i32 [T, 1].</param>
    public static Value AddDelta(GraphBuilder b, Value projected, Value x, string weightName, Value slots)
    {
        string stem = weightName.EndsWith(".weight", StringComparison.Ordinal) ? weightName[..^".weight".Length] : weightName;
        if (b.TryWeight(stem + ".lora_a") is not { } down || b.TryWeight(stem + ".lora_b") is not { } up)
        {
            return projected;
        }

        var low = b.Emit1("MatMulId", [x, down, slots], null, $"{stem}.lora_down");             // [T, 1, R]
        var delta = b.Emit1("MatMulId", [low, up, slots], null, $"{stem}.lora_up");             // [T, 1, N]
        return b.Add(projected, b.Reshape(delta, 0, up.Shape[1].Fixed), $"{stem}.lora_out");
    }
}
