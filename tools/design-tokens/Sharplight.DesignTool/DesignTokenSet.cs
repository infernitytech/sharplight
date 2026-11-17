using System.Globalization;
using System.Text.Json;

namespace Sharplight.DesignTool;

/// <summary>design/tokens.json and design/icons.json, read into ordered groups.</summary>
internal sealed class DesignTokenSet
{
    public required IReadOnlyList<TokenValue> Colors { get; init; }
    public required IReadOnlyDictionary<string, IReadOnlyList<TokenValue>> Themes { get; init; }
    public required IReadOnlyList<AccentPaletteDefinition> Palettes { get; init; }
    public required IReadOnlyList<TokenValue> FontFamilies { get; init; }
    public required IReadOnlyList<TokenValue> FontSizes { get; init; }
    public required IReadOnlyList<TokenValue> FontWeights { get; init; }
    public required IReadOnlyList<TokenValue> Radii { get; init; }
    public required IReadOnlyList<TokenValue> Spacing { get; init; }
    public required IReadOnlyList<TokenValue> Shadows { get; init; }
    public required IReadOnlyList<IconSetDefinition> IconSets { get; init; }
    public required IReadOnlyList<IconDefinition> Icons { get; init; }

    /// <summary>The value of a color token.</summary>
    public string Color(string name) => Colors.First(token => token.Name == name).Value;

    /// <summary>Reads the tokens and the icons they reference.</summary>
    /// <exception cref="InvalidDataException">A file is missing a group or has an invalid entry.</exception>
    public static DesignTokenSet Load(string tokensPath)
    {
        using var tokens = JsonDocument.Parse(File.ReadAllText(tokensPath), new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
        var root = tokens.RootElement;
        var iconsFile = Path.Combine(Path.GetDirectoryName(tokensPath)!, Group(root, "icons").GetProperty("$ref").GetString()!);
        using var icons = JsonDocument.Parse(File.ReadAllText(iconsFile));
        var font = Group(root, "font");
        return new DesignTokenSet
        {
            Colors = Values(Group(root, "color")),
            Themes = Group(root, "theme").EnumerateObject().Where(theme => !theme.Name.StartsWith('$'))
                .ToDictionary(theme => theme.Name, theme => (IReadOnlyList<TokenValue>)Values(theme.Value), StringComparer.Ordinal),
            Palettes = [.. Group(root, "accent-palette").EnumerateObject().Where(entry => !entry.Name.StartsWith('$')).Select(Palette)],
            FontFamilies = Values(Group(font, "family")),
            FontSizes = Values(Group(font, "size")),
            FontWeights = Values(Group(font, "weight")),
            Radii = Values(Group(root, "radius")),
            Spacing = Values(Group(root, "spacing")),
            Shadows = Values(Group(root, "shadow")),
            IconSets = [.. Group(icons.RootElement, "sets").EnumerateObject().Select(set =>
                new IconSetDefinition(set.Name, set.Value.GetProperty("size").GetInt32(), set.Value.GetProperty("comment").GetString()!))],
            Icons = [.. Group(icons.RootElement, "icons").EnumerateArray().Select(icon =>
                new IconDefinition(icon.GetProperty("name").GetString()!, icon.GetProperty("set").GetString()!, icon.GetProperty("path").GetString()!))],
        };
    }

    private static JsonElement Group(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var group) ? group : throw new InvalidDataException($"The design tokens have no '{name}' group.");

    private static List<TokenValue> Values(JsonElement group)
        => [.. group.EnumerateObject().Where(entry => !entry.Name.StartsWith('$')).Select(entry => new TokenValue(entry.Name, Text(entry.Name, entry.Value.GetProperty("$value"))))];

    private static string Text(string name, JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.String => value.GetString()!,
        JsonValueKind.Number => value.GetDouble().ToString(CultureInfo.InvariantCulture),
        JsonValueKind.Array => string.Join(", ", value.EnumerateArray().Select(part => part.GetString()!)),
        _ => throw new InvalidDataException($"Token '{name}' has an unsupported value."),
    };

    private static AccentPaletteDefinition Palette(JsonProperty entry)
    {
        var value = entry.Value;
        return new AccentPaletteDefinition(entry.Name, value.GetProperty("name").GetString()!, value.GetProperty("accent").GetString()!,
            [.. value.GetProperty("light").EnumerateArray().Select(step => step.GetString()!)],
            [.. value.GetProperty("dark").EnumerateArray().Select(step => step.GetString()!)],
            value.GetProperty("secondary").GetString()!);
    }
}
