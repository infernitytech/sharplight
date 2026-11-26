using System.CommandLine;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;

namespace Sharplight.CatalogTool;

/// <summary>The command line of the catalog tool (docs/verified-catalog.md, "Tool").</summary>
internal sealed class CatalogCommands
{
    private readonly Option<string?> _catalog = new("--catalog") { Description = "The catalog file (default: catalog/verified-models.json of the repository).", Recursive = true };
    private readonly Option<string?> _document = new("--doc") { Description = "The document whose tables are generated (default: docs/compatible-models.md).", Recursive = true };
    private readonly Option<bool> _refresh = new("--refresh") { Description = "Resolve every file set again at its pinned commit." };
    private readonly Option<bool> _update = new("--update") { Description = "Pin every file set to the repository's current main (verify the models first)." };
    private readonly Option<string?> _entry = new("--entry") { Description = "Only this entry." };
    private readonly Option<bool> _check = new("--check") { Description = "Do not write: fail when the document differs from the catalog." };

    /// <summary>The root command with its subcommands.</summary>
    public RootCommand Build()
    {
        var root = new RootCommand("Maintains the verified-model catalog: validates it, resolves pinned revisions and files from the Hub, "
            + "generates the tables of docs/compatible-models.md and checks that every pinned file still exists.");
        root.Options.Add(_catalog);
        root.Options.Add(_document);

        var validate = new Command("validate", "Checks the catalog for consistency (no network).");
        validate.SetAction(result => Validate(Load(result)) ? 0 : 1);

        var resolve = new Command("resolve", "Fills in commits, files, sizes, checksums, licenses and memory estimates from the Hub API.");
        resolve.Options.Add(_refresh);
        resolve.Options.Add(_update);
        resolve.Options.Add(_entry);
        resolve.SetAction(ResolveAsync);

        var docs = new Command("docs", "Regenerates the tables of docs/compatible-models.md from the catalog.");
        docs.Options.Add(_check);
        docs.SetAction(Docs);

        var schema = new Command("schema", "Writes catalog/verified-models.schema.json from the catalog types.");
        schema.Options.Add(_check);
        schema.SetAction(Schema);

        var check = new Command("check", "Checks over the network that every pinned file still exists with its size and checksum.");
        check.SetAction(CheckAsync);

        root.Subcommands.Add(validate);
        root.Subcommands.Add(resolve);
        root.Subcommands.Add(docs);
        root.Subcommands.Add(schema);
        root.Subcommands.Add(check);
        return root;
    }

    private async Task<int> ResolveAsync(ParseResult result, CancellationToken cancellationToken)
    {
        var path = CatalogPath(result);
        var catalog = VerifiedCatalogSerializer.Parse(await File.ReadAllTextAsync(path, cancellationToken).ConfigureAwait(false));
        var mode = result.GetValue(_update) ? ResolveMode.Update : result.GetValue(_refresh) ? ResolveMode.Refresh : ResolveMode.Missing;
        using var http = HubClientOptions.CreateHttpClient();
        var resolver = new CatalogResolver(new HubClient(http, HubClientOptions.FromEnvironment()), Console.Error.WriteLine);
        var resolved = await resolver.ResolveAsync(catalog, mode, result.GetValue(_entry), cancellationToken).ConfigureAwait(false);
        await File.WriteAllTextAsync(path, VerifiedCatalogSerializer.Serialize(resolved), cancellationToken).ConfigureAwait(false);
        foreach (var failure in resolver.Failures)
        {
            Console.WriteLine($"not resolved: {failure}");
        }

        return Validate(resolved) && resolver.Failures.Count == 0 ? 0 : 1;
    }

    private int Docs(ParseResult result)
    {
        var catalog = Load(result);
        if (!Validate(catalog))
        {
            // An inconsistent catalog (e.g. an entry duplicated by a merge) would write duplicated rows.
            return 1;
        }

        var documentPath = result.GetValue(_document) ?? Path.Combine(CatalogPaths.FindRepositoryRoot(Environment.CurrentDirectory), CatalogPaths.Document);
        var current = File.ReadAllText(documentPath);
        var generated = CompatibleModelsDocument.Render(current, catalog);
        if (result.GetValue(_check))
        {
            if (generated != current.ReplaceLineEndings("\n"))
            {
                Console.Error.WriteLine($"{documentPath} differs from the catalog; run the docs command.");
                return 1;
            }

            return 0;
        }

        File.WriteAllText(documentPath, generated);
        Console.Error.WriteLine($"Wrote {documentPath}");
        return 0;
    }

    private int Schema(ParseResult result)
    {
        var path = Path.Combine(Path.GetDirectoryName(CatalogPath(result))!, CatalogPaths.SchemaFileName);
        var schema = VerifiedCatalogSerializer.ExportSchema();
        if (result.GetValue(_check))
        {
            bool same = File.Exists(path) && File.ReadAllText(path).ReplaceLineEndings("\n") == schema;
            if (!same)
            {
                Console.Error.WriteLine($"{path} differs from the catalog types; run the schema command.");
            }

            return same ? 0 : 1;
        }

        File.WriteAllText(path, schema);
        Console.Error.WriteLine($"Wrote {path}");
        return 0;
    }

    private async Task<int> CheckAsync(ParseResult result, CancellationToken cancellationToken)
    {
        using var http = HubClientOptions.CreateHttpClient();
        var checker = new CatalogChecker(new HubClient(http, HubClientOptions.FromEnvironment()), Console.Error.WriteLine);
        var problems = await checker.CheckAsync(Load(result), cancellationToken).ConfigureAwait(false);
        foreach (var problem in problems)
        {
            Console.WriteLine(problem);
        }

        return problems.Count == 0 ? 0 : 1;
    }

    private VerifiedCatalog Load(ParseResult result) => VerifiedCatalogSerializer.Parse(File.ReadAllText(CatalogPath(result)));

    private string CatalogPath(ParseResult result)
        => result.GetValue(_catalog) ?? Path.Combine(CatalogPaths.FindRepositoryRoot(Environment.CurrentDirectory), CatalogPaths.Catalog);

    private static bool Validate(VerifiedCatalog catalog)
    {
        var problems = CatalogValidator.Validate(catalog).ToList();

        // Hosts replace their copy only with a catalog whose date is later: an entry verified after it would never reach them.
        var latest = catalog.Entries.Select(entry => entry.Verification.Date).Where(date => date.Length > 0).DefaultIfEmpty("").Max(StringComparer.Ordinal)!;
        if (string.CompareOrdinal(latest, catalog.Updated) > 0)
        {
            problems.Add($"updated '{catalog.Updated}' is older than the latest verification date '{latest}': set it to the day of the change, or hosts keep their older copy");
        }

        foreach (var problem in problems)
        {
            Console.WriteLine(problem);
        }

        Console.Error.WriteLine($"{catalog.Entries.Count} entries, {catalog.Entries.Count(entry => entry.IsInstallable)} installable, {problems.Count} problem(s).");
        return problems.Count == 0;
    }
}
