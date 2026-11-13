using Sharplight.Tensors;

namespace Sharplight.Graph;

// RWKV time-mix recurrences.
public static partial class StandardOps
{
    /// <summary>
    /// RWKV-7 "generalised delta rule" recurrence, one call per layer.
    /// Inputs, in order: <c>r</c>, <c>w</c>, <c>k</c>, <c>v</c>, <c>a</c>, <c>b</c> (all [T, H, N] f32; <c>w</c> is the per-channel decay multiplier already in (0, 1),
    /// <c>a</c> and <c>b</c> the two vectors of the in-context learning-rate update: <c>a = -kk</c>, <c>b = kk · a_gate</c>), <c>state</c> State [H, N, N] f32
    /// (row i is the memory of value channel i, column j runs over key channels; the op reads it, advances it by the T tokens and writes it back).
    /// Per token and head, for every value channel i: <c>sa = Σⱼ a[j]·S[i,j]</c>; <c>S[i,j] = S[i,j]·w[j] + v[i]·k[j] + sa·b[j]</c>; <c>y[i] = Σⱼ S[i,j]·r[j]</c>.
    /// Output y [T, H, N] f32. Multi-sequence form (see <c>MultiSequence</c>): eighth input = sequence runs [U, 4] i32 and state [S, H, N, N]; each run continues
    /// its slot's state and a run starting at position 0 starts from zero state.
    /// </summary>
    public static readonly OpSchema Wkv7 = new("Wkv7", OpCategory.Composite, 7, 8, (i, a) => [new(DataType.F32, i[0].Shape)]);

    /// <summary>
    /// RWKV-6 recurrence (data-dependent decay with a "bonus" for the current token), one call per layer.
    /// Inputs, in order: <c>r</c>, <c>k</c>, <c>v</c>, <c>w</c> (all [T, H, N] f32; <c>w</c> is the per-channel decay multiplier already in (0, 1)), <c>u</c> [H, N] f32
    /// (the bonus, "time_first"), <c>state</c> State [H, N, N] f32 (row i is the memory of value channel i, column j runs over key channels).
    /// Per token and head, for every value channel i: <c>y[i] = Σⱼ r[j]·(u[j]·k[j]·v[i] + S[i,j])</c>; <c>S[i,j] = S[i,j]·w[j] + k[j]·v[i]</c>.
    /// Output y [T, H, N] f32. Multi-sequence form: seventh input = sequence runs [U, 4] i32 and state [S, H, N, N] (as for <c>Wkv7</c>).
    /// </summary>
    public static readonly OpSchema Wkv6 = new("Wkv6", OpCategory.Composite, 6, 7, (i, a) => [new(DataType.F32, i[0].Shape)]);
}
