namespace Sharplight.DesignTool;

/// <summary>Name conversions between the kebab-case tokens and the generated outputs.</summary>
internal static class Names
{
    /// <summary><c>surface-alt</c> → <c>SurfaceAlt</c>, <c>2xl</c> → <c>2xl</c> (digits stay; a leading letter is capitalized).</summary>
    public static string Pascal(string kebab)
        => string.Concat(kebab.Split('-').Select(part => part.Length == 0 ? part : char.ToUpperInvariant(part[0]) + part[1..]));

    /// <summary>A pixel dimension without its unit (<c>8px</c> → <c>8</c>).</summary>
    public static string Pixels(string value)
        => value.EndsWith("px", StringComparison.Ordinal) ? value[..^2] : throw new InvalidDataException($"'{value}' is not a px dimension.");
}
