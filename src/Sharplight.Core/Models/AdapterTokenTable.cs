using Sharplight.Tensors;

namespace Sharplight.Models;

/// <summary>
/// The canonical table of token-activated adapters (Granite Switch), shared by importers: <c>adapter_tokens.weight</c>, f32 [V, <see cref="Width"/>], per token
/// the token it is embedded as (its substitute for an activation token, itself otherwise), its router key (+gain for an activation token,
/// −gain otherwise) and its router value (the slot i + 1 of activation token i, 0 otherwise), then zeros. The number of adapters goes to the
/// <c>adapters.count</c> hyper-parameter; the projections carry their adapter pairs as <c>{projection}.lora_a</c> [S, R, K] and
/// <c>{projection}.lora_b</c> [S, N, R] with the all-zero slot 0 first.
/// </summary>
public static class AdapterTokenTable
{
    /// <summary>Columns per token: the three values, then zero padding (rows of 8 values gather as vectors).</summary>
    public const int Width = 8;

    /// <summary>Canonical weight name of the table.</summary>
    public const string WeightName = "adapter_tokens.weight";

    /// <summary>Builds the table.</summary>
    /// <param name="vocabulary">Vocabulary size V.</param>
    /// <param name="activationTokens">The activation token of every adapter.</param>
    /// <param name="substituteTokens">The token every activation token is embedded as.</param>
    /// <param name="gain">The router gain (the key magnitude).</param>
    /// <exception cref="ArgumentException">The token lists differ in length or name tokens outside the vocabulary.</exception>
    public static WeightInfo Create(int vocabulary, IReadOnlyList<int> activationTokens, IReadOnlyList<int> substituteTokens, float gain)
    {
        if (activationTokens.Count != substituteTokens.Count)
        {
            throw new ArgumentException($"{activationTokens.Count} adapter activation tokens but {substituteTokens.Count} substitute tokens.");
        }

        if (activationTokens.Concat(substituteTokens).Any(t => t < 0 || t >= vocabulary))
        {
            throw new ArgumentException($"An adapter token lies outside the vocabulary of {vocabulary}.");
        }

        return WeightInfo.Derived(WeightName, DataType.F32, new Shape(vocabulary, Width), () =>
        {
            var table = new float[vocabulary * Width];
            for (int token = 0; token < vocabulary; token++)
            {
                table[Width * token] = token;
                table[Width * token + 1] = -gain;
            }

            for (int i = 0; i < activationTokens.Count; i++)
            {
                int token = activationTokens[i];
                table[Width * token] = substituteTokens[i];
                table[Width * token + 1] = gain;
                table[Width * token + 2] = i + 1;
            }

            var bytes = new byte[table.Length * sizeof(float)];
            Buffer.BlockCopy(table, 0, bytes, 0, bytes.Length);
            return bytes;
        });
    }
}
