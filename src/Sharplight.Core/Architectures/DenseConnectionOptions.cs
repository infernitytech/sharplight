using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="DenseConnections"/>, read from the <c>{arch}.dense_connections.*</c> keys: <c>tap_count</c> (n, 0 = off) and
/// <c>taps</c>, n entries per layer (layer_count · n in all) naming the earlier layer inputs a layer mixes, -1 for a layer without dense connections.
/// </summary>
/// <param name="TapCount">Taps n per connected layer.</param>
/// <param name="Taps">The tapped layer indices, n per layer (-1: the layer has none).</param>
internal sealed record DenseConnectionOptions(int TapCount, int[] Taps)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no dense connections.</summary>
    /// <exception cref="InvalidDataException">The tap list does not hold n entries per layer or names a later layer.</exception>
    public static DenseConnectionOptions? Read(ModelSource src, int layerCount)
    {
        int tapCount = src.Arch("dense_connections.tap_count", 0);
        if (tapCount <= 0)
        {
            return null;
        }

        var options = new DenseConnectionOptions(tapCount, src.Arch("dense_connections.taps", Array.Empty<int>()));
        if (options.Taps.Length != layerCount * tapCount)
        {
            throw new InvalidDataException("decoder: dense_connections.taps must hold tap_count entries per layer.");
        }

        for (int il = 0; il < layerCount; il++)
        {
            if (options.TapsOf(il) is { } taps && taps.Any(t => t < 0 || t > il))
            {
                throw new InvalidDataException($"decoder: dense_connections.taps of layer {il} must name layers 0..{il}.");
            }
        }

        return options;
    }

    /// <summary>The layers whose inputs layer <paramref name="il"/> mixes, or null when it has no dense connections.</summary>
    public int[]? TapsOf(int il) => Taps[il * TapCount] < 0 ? null : Taps[(il * TapCount)..((il + 1) * TapCount)];
}
