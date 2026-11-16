using System.Globalization;

namespace Sharplight.Studio.ViewModels.Inference;

/// <summary>Reads the seed box of the generation pages: empty means a random seed, otherwise a non-negative whole number.</summary>
internal static class SeedParser
{
    /// <summary>Reads the seed from the text of a seed parameter.</summary>
    /// <param name="text">The box's text.</param>
    /// <param name="seed">The seed, or null for a random one.</param>
    /// <returns>False when the text is neither empty nor a non-negative whole number.</returns>
    public static bool TryRead(string text, out long? seed)
    {
        string trimmed = text.Trim();
        seed = null;
        if (trimmed.Length == 0)
        {
            return true;
        }

        if (long.TryParse(trimmed, NumberStyles.Integer, CultureInfo.InvariantCulture, out long parsed) && parsed >= 0)
        {
            seed = parsed;
            return true;
        }

        return false;
    }
}
