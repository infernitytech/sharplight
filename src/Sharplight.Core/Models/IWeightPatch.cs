namespace Sharplight.Models;

/// <summary>
/// The <see cref="ModelSource.FormatData"/> of an auxiliary source that changes the primary's weights instead of adding a part of its own
/// (a LoRA: low-rank deltas added to the weights they were trained for). Importers implement it for their formats: they know how the
/// patch's tensor names map to the primary's canonical names and how the primary's stored types decode.
/// </summary>
public interface IWeightPatch
{
    /// <summary>The primary's weights this patch replaces (by canonical name), lazily computed.</summary>
    /// <param name="primary">The primary source.</param>
    /// <param name="scale">The strength (1: as trained).</param>
    /// <exception cref="NotSupportedException">The patch does not fit the primary (another model family, or no weight it targets).</exception>
    public IReadOnlyDictionary<string, WeightInfo> Apply(ModelSource primary, float scale);
}
