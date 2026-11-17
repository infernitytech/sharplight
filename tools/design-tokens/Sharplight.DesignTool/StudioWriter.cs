using System.Text;

namespace Sharplight.DesignTool;

/// <summary>The Studio outputs: the Avalonia resource dictionaries of colours, icons and the other tokens, and the accent palettes.</summary>
internal static class StudioWriter
{
    private const string Header = """
        <ResourceDictionary xmlns="https://github.com/avaloniaui"
                            xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
        """;

    private const string GeneratedNote = "  <!-- Generated from design/tokens.json by tools/design-tokens (docs/design-tokens.md); edit the tokens, not this file. -->";

    public static IEnumerable<GeneratedFile> Write(DesignTokenSet tokens)
    {
        yield return new GeneratedFile("apps/Sharplight.Studio/Themes/Colors.axaml", Colors(tokens));
        yield return new GeneratedFile("apps/Sharplight.Studio/Themes/Icons.axaml", Icons(tokens));
        yield return new GeneratedFile("apps/Sharplight.Studio/Themes/Tokens.axaml", Others(tokens));
        yield return new GeneratedFile("apps/Sharplight.Studio/Models/GeneratedAccentPalettes.cs", Palettes(tokens));
    }

    private static string Colors(DesignTokenSet tokens)
    {
        var text = new StringBuilder();
        text.AppendLine(Header);
        text.AppendLine(GeneratedNote);
        text.AppendLine("  <!-- Accent brushes are replaced at runtime by ThemeService (palette from the settings); these are the Violet defaults. -->");
        string accent = tokens.Color("accent");
        string secondary = tokens.Color("accent-secondary");
        text.AppendLine($"""  <SolidColorBrush x:Key="AccentBrush" Color="{accent}" />""");
        text.AppendLine($"""  <SolidColorBrush x:Key="AccentSecondaryBrush" Color="{secondary}" />""");
        text.AppendLine($"""  <SolidColorBrush x:Key="AccentSoftBrush" Color="{accent}" Opacity="{tokens.Color("accent-soft-opacity")}" />""");
        text.AppendLine("""  <LinearGradientBrush x:Key="AccentGradientBrush" StartPoint="0%,0%" EndPoint="100%,100%">""");
        text.AppendLine($"""    <GradientStop Color="{accent}" Offset="0" />""");
        text.AppendLine($"""    <GradientStop Color="{secondary}" Offset="1" />""");
        text.AppendLine("  </LinearGradientBrush>");
        foreach (var color in tokens.Colors.Where(color => !color.Name.StartsWith("accent", StringComparison.Ordinal)))
        {
            text.AppendLine($"""  <SolidColorBrush x:Key="{Names.Pascal(color.Name)}Brush" Color="{color.Value}" />""");
        }

        text.AppendLine();
        text.AppendLine("  <ResourceDictionary.ThemeDictionaries>");
        foreach (var (theme, colors) in tokens.Themes)
        {
            text.AppendLine($"""    <ResourceDictionary x:Key="{Names.Pascal(theme)}">""");
            foreach (var color in colors)
            {
                text.AppendLine($"""      <SolidColorBrush x:Key="{Names.Pascal(color.Name)}Brush" Color="{color.Value}" />""");
            }

            text.AppendLine("    </ResourceDictionary>");
        }

        text.AppendLine("  </ResourceDictionary.ThemeDictionaries>");
        text.AppendLine("</ResourceDictionary>");
        return text.ToString();
    }

    private static string Icons(DesignTokenSet tokens)
    {
        var text = new StringBuilder();
        text.AppendLine(Header);
        text.AppendLine(GeneratedNote);
        foreach (var set in tokens.IconSets)
        {
            text.AppendLine($"  <!-- {set.Comment} -->");
        }

        foreach (var icon in tokens.Icons)
        {
            text.AppendLine($"""  <StreamGeometry x:Key="{Names.Pascal(icon.Set)}{Names.Pascal(icon.Name)}">{icon.Path}</StreamGeometry>""");
        }

        text.AppendLine("</ResourceDictionary>");
        return text.ToString();
    }

    private static string Others(DesignTokenSet tokens)
    {
        var text = new StringBuilder();
        text.AppendLine(Header);
        text.AppendLine(GeneratedNote);
        text.AppendLine("  <!-- Typography and corner radii; Styles.axaml uses them as dynamic resources. The sans family is the embedded Inter (App.axaml). -->");
        foreach (var family in tokens.FontFamilies.Where(family => family.Name != "sans"))
        {
            text.AppendLine($"""  <FontFamily x:Key="{Names.Pascal(family.Name)}FontFamily">{family.Value}</FontFamily>""");
        }

        foreach (var size in tokens.FontSizes)
        {
            text.AppendLine($"""  <x:Double x:Key="FontSize{Names.Pascal(size.Name)}">{Names.Pixels(size.Value)}</x:Double>""");
        }

        foreach (var radius in tokens.Radii)
        {
            text.AppendLine($"""  <CornerRadius x:Key="Radius{Names.Pascal(radius.Name)}">{Names.Pixels(radius.Value)}</CornerRadius>""");
        }

        text.AppendLine("</ResourceDictionary>");
        return text.ToString();
    }

    private static string Palettes(DesignTokenSet tokens)
    {
        var text = new StringBuilder();
        text.AppendLine("// Generated from design/tokens.json by tools/design-tokens (docs/design-tokens.md); edit the tokens, not this file.");
        text.AppendLine("namespace Sharplight.Studio.Models;");
        text.AppendLine();
        text.AppendLine("/// <summary>The accent palettes of the design tokens, in the order the settings offer them.</summary>");
        text.AppendLine("internal static class GeneratedAccentPalettes");
        text.AppendLine("{");
        text.AppendLine("    /// <summary>Every palette.</summary>");
        text.AppendLine("    public static IReadOnlyList<AccentPalette> All { get; } =");
        text.AppendLine("    [");
        foreach (var palette in tokens.Palettes)
        {
            var steps = string.Join(", ", palette.Light.Concat(palette.Dark).Select(step => $"\"{step}\""));
            text.AppendLine($"        new(\"{palette.Name}\", \"{palette.Accent}\", {steps}, \"{palette.Secondary}\"),");
        }

        text.AppendLine("    ];");
        text.AppendLine("}");
        return text.ToString();
    }
}
