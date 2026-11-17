using System.CommandLine;

namespace Sharplight.DesignTool;

/// <summary>
/// The <c>sharplight-design</c> command line: <c>generate</c> writes Studio's resource dictionaries and accent palettes and the
/// Sharplight.DesignTokens web assets from design/tokens.json; <c>generate --check</c> fails when any of them is out of date (CI).
/// </summary>
internal static class DesignToolCommands
{
    public static RootCommand Build()
    {
        var root = new RootCommand("Generates the outputs of the Sharplight design tokens (docs/design-tokens.md).");
        var check = new Option<bool>("--check") { Description = "Do not write: exit with 1 when an output differs from what the tokens generate." };
        var repository = new Option<string?>("--root") { Description = "Repository root (default: found from the working directory)." };
        var generate = new Command("generate", "Writes Studio's Themes/Colors.axaml, Icons.axaml, Tokens.axaml and Models/GeneratedAccentPalettes.cs and the CSS and SVG sprite of src/Sharplight.DesignTokens.");
        generate.Options.Add(check);
        generate.Options.Add(repository);
        generate.SetAction(parse => Generate(parse.GetValue(repository) ?? FindRoot(), parse.GetValue(check)));
        root.Subcommands.Add(generate);
        return root;
    }

    private static int Generate(string root, bool check)
    {
        var tokens = DesignTokenSet.Load(Path.Combine(root, "design", "tokens.json"));
        var outdated = new List<string>();
        foreach (var file in StudioWriter.Write(tokens).Concat(WebWriter.Write(tokens)))
        {
            var path = Path.Combine(root, file.RelativePath);
            var current = File.Exists(path) ? File.ReadAllText(path) : null;
            if (current == file.Content)
            {
                continue;
            }

            outdated.Add(file.RelativePath);
            if (!check)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(path)!);
                File.WriteAllText(path, file.Content);
            }
        }

        foreach (var path in outdated)
        {
            Console.WriteLine(check ? $"out of date: {path}" : $"wrote {path}");
        }

        if (check && outdated.Count > 0)
        {
            Console.Error.WriteLine("Run: dotnet run --project tools/design-tokens/Sharplight.DesignTool -- generate");
            return 1;
        }

        return 0;
    }

    private static string FindRoot()
    {
        for (var directory = new DirectoryInfo(Directory.GetCurrentDirectory()); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "design", "tokens.json")))
            {
                return directory.FullName;
            }
        }

        throw new InvalidOperationException("No design/tokens.json above the working directory; pass --root.");
    }
}
