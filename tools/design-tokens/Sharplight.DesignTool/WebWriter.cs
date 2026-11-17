using System.Text;

namespace Sharplight.DesignTool;

/// <summary>The web outputs of the Sharplight.DesignTokens package: CSS custom properties and the SVG icon sprite.</summary>
internal static class WebWriter
{
    private const string Folder = "src/Sharplight.DesignTokens/wwwroot/";
    private const string Note = "Generated from design/tokens.json by tools/design-tokens (docs/design-tokens.md); edit the tokens, not this file.";

    public static IEnumerable<GeneratedFile> Write(DesignTokenSet tokens)
    {
        yield return new GeneratedFile(Folder + "sharplight-tokens.css", Css(tokens));
        yield return new GeneratedFile(Folder + "sharplight-icons.svg", Sprite(tokens));
    }

    private static string Css(DesignTokenSet tokens)
    {
        var text = new StringBuilder();
        text.AppendLine($"/* {Note} */");
        text.AppendLine();
        text.AppendLine(":root {");
        var violet = tokens.Palettes[0];
        foreach (var color in tokens.Colors.Where(color => color.Name != "accent-soft-opacity"))
        {
            text.AppendLine($"  --sl-color-{color.Name}: {color.Value};");
        }

        string percent = (double.Parse(tokens.Color("accent-soft-opacity"), System.Globalization.CultureInfo.InvariantCulture) * 100).ToString("0.##", System.Globalization.CultureInfo.InvariantCulture);
        AppendSteps(text, violet, percent);
        foreach (var family in tokens.FontFamilies)
        {
            text.AppendLine($"  --sl-font-{family.Name}: {CssFontStack(family.Value)};");
        }

        Append(text, "font-size", tokens.FontSizes);
        Append(text, "font-weight", tokens.FontWeights);
        Append(text, "radius", tokens.Radii);
        Append(text, "space", tokens.Spacing);
        Append(text, "shadow", tokens.Shadows);
        text.AppendLine("}");
        text.AppendLine();

        var themes = tokens.Themes.ToList();
        var first = themes[0];
        text.AppendLine($":root, [data-theme=\"{first.Key}\"] {{");
        AppendTheme(text, first.Key, first.Value, "  ");
        text.AppendLine("}");
        foreach (var (name, colors) in themes.Skip(1))
        {
            text.AppendLine();
            text.AppendLine($"[data-theme=\"{name}\"] {{");
            AppendTheme(text, name, colors, "  ");
            text.AppendLine("}");
            text.AppendLine();
            text.AppendLine($"@media (prefers-color-scheme: {name}) {{");
            text.AppendLine("  :root:not([data-theme]) {");
            AppendTheme(text, name, colors, "    ");
            text.AppendLine("  }");
            text.AppendLine("}");
        }

        foreach (var palette in tokens.Palettes)
        {
            text.AppendLine();
            text.AppendLine($"[data-accent=\"{palette.Key}\"] {{");
            text.AppendLine($"  --sl-color-accent: {palette.Accent};");
            text.AppendLine($"  --sl-color-accent-secondary: {palette.Secondary};");
            AppendSteps(text, palette, percent);
            text.AppendLine("}");
        }

        text.AppendLine();
        text.AppendLine("/* Icons of sharplight-icons.svg: <svg class=\"sl-icon\"><use href=\"_content/Sharplight.DesignTokens/sharplight-icons.svg#sl-icon-chat\" /></svg> */");
        text.AppendLine(".sl-icon {");
        text.AppendLine("  width: 1em;");
        text.AppendLine("  height: 1em;");
        text.AppendLine("  fill: none;");
        text.AppendLine("  stroke: currentColor;");
        text.AppendLine("  stroke-width: 1.8;");
        text.AppendLine("  stroke-linecap: round;");
        text.AppendLine("  stroke-linejoin: round;");
        text.AppendLine("}");
        text.AppendLine();
        text.AppendLine(".sl-caption-icon {");
        text.AppendLine("  width: 10px;");
        text.AppendLine("  height: 10px;");
        text.AppendLine("  fill: none;");
        text.AppendLine("  stroke: currentColor;");
        text.AppendLine("  stroke-width: 1;");
        text.AppendLine("}");
        return text.ToString();
    }

    private static string Sprite(DesignTokenSet tokens)
    {
        var sizes = tokens.IconSets.ToDictionary(set => set.Name, set => set.Size, StringComparer.Ordinal);
        var text = new StringBuilder();
        text.AppendLine($"<!-- {Note} -->");
        text.AppendLine("<svg xmlns=\"http://www.w3.org/2000/svg\">");
        foreach (var icon in tokens.Icons)
        {
            int size = sizes[icon.Set];
            text.AppendLine($"  <symbol id=\"sl-{icon.Set}-{icon.Name}\" viewBox=\"0 0 {size} {size}\"><path d=\"{icon.Path}\" /></symbol>");
        }

        text.AppendLine("</svg>");
        return text.ToString();
    }

    private static void Append(StringBuilder text, string prefix, IEnumerable<TokenValue> values)
    {
        foreach (var value in values)
        {
            text.AppendLine($"  --sl-{prefix}-{value.Name}: {value.Value};");
        }
    }

    private static void AppendTheme(StringBuilder text, string name, IEnumerable<TokenValue> colors, string indent)
    {
        text.AppendLine($"{indent}color-scheme: {name};");
        foreach (var color in colors)
        {
            text.AppendLine($"{indent}--sl-color-{color.Name}: {color.Value};");
        }
    }

    /// <summary>
    /// The steps of a palette and the colours derived from the accent (redeclared with every palette: a custom property that uses
    /// var() is resolved where it is declared, so the derived colours follow a <c>data-accent</c> only when declared there too).
    /// </summary>
    private static void AppendSteps(StringBuilder text, AccentPaletteDefinition palette, string softPercent)
    {
        for (int i = 0; i < palette.Light.Count; i++)
        {
            text.AppendLine($"  --sl-color-accent-light-{i + 1}: {palette.Light[i]};");
        }

        for (int i = 0; i < palette.Dark.Count; i++)
        {
            text.AppendLine($"  --sl-color-accent-dark-{i + 1}: {palette.Dark[i]};");
        }

        text.AppendLine($"  --sl-color-accent-soft: color-mix(in srgb, var(--sl-color-accent) {softPercent}%, transparent);");
        text.AppendLine("  --sl-gradient-accent: linear-gradient(135deg, var(--sl-color-accent), var(--sl-color-accent-secondary));");
    }

    /// <summary>A font stack for CSS: names with spaces quoted, generic families bare.</summary>
    private static string CssFontStack(string families)
        => string.Join(", ", families.Split(", ").Select(family => family.Contains(' ') ? $"\"{family}\"" : family));
}
