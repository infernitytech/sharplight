using System.CommandLine;
using System.Text.Json;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Client;
using Sharplight.ModelFeed.Publishing;
using Sharplight.ModelFeed.Signing;

namespace Sharplight.FeedTool;

/// <summary>The command line of the feed tool (docs/model-feed.md, "Building a feed").</summary>
internal sealed class FeedCommands
{
    private readonly Option<string?> _catalog = new("--catalog") { Description = "The verified catalog (default: catalog/verified-models.json of the repository).", Recursive = true };
    private readonly Option<string?> _overlay = new("--overlay") { Description = "The feed overlay (default: tools/model-feed/feed-overlay.json of the repository).", Recursive = true };
    private readonly Option<string> _feedFolder = new("--feed") { Description = "The feed folder.", Required = true };
    private readonly Option<string> _feedLocation = new("--feed") { Description = "The feed: an http(s) URL, a file:// URL or a folder.", Required = true };
    private readonly Option<string?> _buildFeed = new("--feed") { Description = "The feed folder used to choose versions (optional)." };
    private readonly Option<string> _key = new("--key") { Description = "The feed signing key file (private; never in a repository).", Required = true };
    private readonly Option<string> _rootKey = new("--root-key") { Description = "The root signing key file (offline).", Required = true };
    private readonly Option<string[]> _roots = new("--root") { Description = "Trusted root public key file(s).", Required = true, AllowMultipleArgumentsPerToken = true };
    private readonly Option<string[]> _feedKeys = new("--feed-key") { Description = "Public key file(s) of the feed keys to list.", Required = true, AllowMultipleArgumentsPerToken = true };
    private readonly Option<string[]> _revoke = new("--revoke") { Description = "Key ids to list as revoked.", AllowMultipleArgumentsPerToken = true };
    private readonly Option<int> _keySetVersion = new("--version") { Description = "Key set version (increase on every change).", DefaultValueFactory = _ => 1 };
    private readonly Option<int?> _keySetDays = new("--expires-days") { Description = "Days until the key set expires (none by default)." };
    private readonly Option<string> _out = new("--out") { Description = "Output path.", Required = true };
    private readonly Option<string?> _comment = new("--comment") { Description = "Comment stored in the key file." };
    private readonly Option<string[]> _entries = new("--entry") { Description = "Catalog entry ids (with --variant: one entry).", Required = true, AllowMultipleArgumentsPerToken = true };
    private readonly Option<string?> _variant = new("--variant") { Description = "Catalog variant id (default: the recommended variant)." };
    private readonly Option<string?> _store = new("--store") { Description = "Model store for downloads and conversion (default: <out>/.store)." };
    private readonly Option<string?> _feedUrl = new("--feed-url") { Description = "Public feed URL written into manifests." };
    private readonly Option<string> _name = new("--name") { Description = "Feed name.", DefaultValueFactory = _ => "Sharplight verified models" };
    private readonly Option<int> _validityDays = new("--validity-days") { Description = "Days until the index expires (republish before).", DefaultValueFactory = _ => 30 };
    private readonly Option<bool> _files = new("--files") { Description = "Also read and hash every package file." };
    private readonly Option<bool> _allowExpired = new("--allow-expired") { Description = "Accept an expired index with a warning." };
    private readonly Option<string?> _filter = new("--filter") { Description = "Only copy package files whose id starts with this prefix." };
    private readonly Option<string> _reason = new("--reason") { Description = "Why.", Required = true };
    private readonly Option<string> _severity = new("--severity") { Description = "quality, legal or security.", DefaultValueFactory = _ => YankSeverities.Quality };
    private readonly Option<string?> _alternate = new("--alternate") { Description = "Replacement package id." };
    private readonly Argument<string> _id = new("id") { Description = "Package id." };
    private readonly Argument<string> _version = new("version") { Description = "Package version." };
    private readonly Argument<string[]> _folders = new("folders") { Description = "Staged package folders (from build).", Arity = ArgumentArity.OneOrMore };

    /// <summary>The root command.</summary>
    public RootCommand Build()
    {
        var root = new RootCommand("Builds, signs, publishes and verifies Sharplight model feeds (static, signed, paged JSON plus immutable .slm packages).");
        root.Options.Add(_catalog);
        root.Options.Add(_overlay);
        root.Subcommands.Add(Command("keygen", "Generates an ECDSA P-384 signing key: <out>.key.json (private) and <out>.pub.json.", KeyGen, _out, _comment));
        root.Subcommands.Add(Command("keyset", "Writes the feed's key set (feed public keys) signed by the root key.", KeySet, _feedFolder, _rootKey, _feedKeys, _revoke, _keySetVersion, _keySetDays));
        root.Subcommands.Add(Command("plan", "Lists catalog variants without a package for their pinned revision in the feed.", Plan, _buildFeed));
        root.Subcommands.Add(Command("build", "Builds packages from catalog variants into staging folders (download, convert with companions, hash, manifest).", BuildAsync, _entries, _variant, _out, _store, _buildFeed, _feedUrl));
        root.Subcommands.Add(Command("publish", "Adds staged packages to the feed folder, signs manifests and the new index (sequence + 1).", PublishAsync, _folders, _feedFolder, _key, _name, _validityDays));
        root.Subcommands.Add(Command("reindex", "Re-signs the index with the next sequence (refreshes the expiry).", ReindexAsync, _feedFolder, _key, _name, _validityDays));
        root.Subcommands.Add(StateCommand("yank", "Withdraws a version (not installable; installed copies are flagged).", true, false, false));
        root.Subcommands.Add(StateCommand("deprecate", "Deprecates a version (still installable, flagged).", false, true, false));
        root.Subcommands.Add(StateCommand("unlist", "Hides a version from listings.", false, false, true));
        root.Subcommands.Add(Command("verify", "Checks a feed the way a host does (key set, index, digests, signatures; --files: package hashes).", VerifyAsync, _feedLocation, _roots, _files, _allowExpired));
        root.Subcommands.Add(Command("mirror", "Copies a verified feed into a folder (offline bundle).", MirrorAsync, _feedLocation, _roots, _out, _filter, _allowExpired));
        return root;
    }

    private static Command Command(string name, string description, Func<ParseResult, CancellationToken, Task<int>> action, params Symbol[] symbols)
    {
        var command = new Command(name, description);
        foreach (var symbol in symbols)
        {
            if (symbol is Option option)
            {
                command.Options.Add(option);
            }
            else if (symbol is Argument argument)
            {
                command.Arguments.Add(argument);
            }
        }

        command.SetAction(async (result, cancellationToken) =>
        {
            try
            {
                return await action(result, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception exception) when (exception is FeedFormatException or FeedVerificationException or IOException or InvalidDataException or KeyNotFoundException or ArgumentException or Sharplight.Hub.HubException)
            {
                await Console.Error.WriteLineAsync($"error: {exception.Message}").ConfigureAwait(false);
                return 1;
            }
        });
        return command;
    }

    private static Command Command(string name, string description, Func<ParseResult, int> action, params Symbol[] symbols) =>
        Command(name, description, (result, _) => Task.FromResult(action(result)), symbols);

    private Command StateCommand(string name, string description, bool yank, bool deprecate, bool unlist)
    {
        Symbol[] symbols = yank ? [_id, _version, _feedFolder, _key, _reason, _severity, _name, _validityDays]
            : deprecate ? [_id, _version, _feedFolder, _key, _reason, _alternate, _name, _validityDays]
            : [_id, _version, _feedFolder, _key, _name, _validityDays];
        return Command(name, description, async (result, cancellationToken) =>
        {
            var now = DateTimeOffset.UtcNow;
            using var key = FeedSigningKey.Load(result.GetValue(_key)!);
            var sequence = await new FeedPublisher().SetStateAsync(result.GetValue(_feedFolder)!, result.GetValue(_id)!, result.GetValue(_version)!, state =>
            {
                if (yank)
                {
                    return state with { Yanked = new PackageYank { Date = now, Reason = result.GetValue(_reason), Severity = result.GetValue(_severity)! } };
                }

                if (deprecate)
                {
                    var alternate = result.GetValue(_alternate) is { } id ? new PackageAlternate { Id = id } : null;
                    return state with { Deprecated = new PackageDeprecation { Date = now, Reason = result.GetValue(_reason), Alternate = alternate } };
                }

                return state with { Listed = !unlist };
            }, key, PublishOptions(result), cancellationToken).ConfigureAwait(false);
            Console.WriteLine($"{name}: {result.GetValue(_id)} {result.GetValue(_version)}; index sequence {sequence}");
            return 0;
        }, symbols);
    }

    private int KeyGen(ParseResult result)
    {
        var output = result.GetValue(_out)!;
        using var key = FeedSigningKey.Generate(result.GetValue(_comment));
        key.Save(output + ".key.json", output + ".pub.json");
        Console.WriteLine($"{key.KeyId}: private key {output}.key.json (keep it offline / in a secret store), public key {output}.pub.json");
        return 0;
    }

    private int KeySet(ParseResult result)
    {
        var now = DateTimeOffset.UtcNow;
        var revoked = result.GetValue(_revoke) ?? [];
        var keys = (result.GetValue(_feedKeys) ?? []).Select(FeedPublicKey.Load).Select(key => new FeedKeySetEntry
        {
            KeyId = key.KeyId,
            PublicKey = key.Base64,
            NotBefore = now.AddDays(-1),
            Revoked = revoked.Contains(key.KeyId, StringComparer.Ordinal) ? now : null,
        }).ToList();
        var keySet = new FeedKeySet
        {
            Version = result.GetValue(_keySetVersion),
            Issued = now,
            Expires = result.GetValue(_keySetDays) is { } days ? now.AddDays(days) : null,
            Keys = keys,
        };
        using var root = FeedSigningKey.Load(result.GetValue(_rootKey)!);
        FeedPublisher.WriteKeySet(result.GetValue(_feedFolder)!, keySet, root);
        Console.WriteLine($"Key set version {keySet.Version} with {keys.Count} key(s), signed by root {root.KeyId}");
        return 0;
    }

    private int Plan(ParseResult result)
    {
        var catalog = LoadCatalog(result);
        var feed = result.GetValue(_buildFeed);
        var missing = 0;
        foreach (var entry in catalog.Entries.Where(entry => entry.IsInstallable))
        {
            foreach (var variant in entry.Variants)
            {
                var id = ManifestFactory.PackageId(entry, variant);
                var latest = feed is null ? null : LatestManifest(feed, id);
                var pinned = latest?.Provenance?.Sources.Any(source => source.Role == "primary" && source.Revision == variant.Revision) == true;
                if (!pinned)
                {
                    missing++;
                    Console.WriteLine($"{id}\t{(latest is null ? "new" : $"revision changed (feed has {latest.Version})")}\t{variant.TotalBytes / (1 << 20)} MiB");
                }
            }
        }

        Console.Error.WriteLine($"{missing} variant(s) need a package build.");
        return 0;
    }

    private async Task<int> BuildAsync(ParseResult result, CancellationToken cancellationToken)
    {
        var catalog = LoadCatalog(result);
        var overlay = LoadOverlay(result);
        var output = Path.GetFullPath(result.GetValue(_out)!);
        var builder = new PackageBuilder(result.GetValue(_store) ?? Path.Combine(output, ".store"), Console.Error.WriteLine);
        var entries = result.GetValue(_entries) ?? [];
        var variantId = result.GetValue(_variant);
        if (variantId is not null && entries.Length != 1)
        {
            throw new ArgumentException("--variant needs exactly one --entry.");
        }

        foreach (var entryId in entries)
        {
            var entry = catalog.Entries.FirstOrDefault(candidate => candidate.Id == entryId) ?? throw new ArgumentException($"The catalog has no entry '{entryId}'.");
            if (!entry.IsInstallable)
            {
                throw new ArgumentException($"{entryId} is not installable (level {entry.Level}, {entry.Variants.Count} variant(s)).");
            }

            var variant = variantId is null ? entry.DefaultVariant! : entry.Variants.FirstOrDefault(candidate => candidate.Id == variantId)
                ?? throw new ArgumentException($"{entryId} has no variant '{variantId}' ({string.Join(", ", entry.Variants.Select(candidate => candidate.Id))}).");
            var folder = await builder.BuildAsync(CatalogInstallPlan.For(entry, variant), output, result.GetValue(_buildFeed), overlay, result.GetValue(_feedUrl), cancellationToken).ConfigureAwait(false);
            if (folder is not null)
            {
                Console.WriteLine(folder);
            }
        }

        return 0;
    }

    private async Task<int> PublishAsync(ParseResult result, CancellationToken cancellationToken)
    {
        using var key = FeedSigningKey.Load(result.GetValue(_key)!);
        var published = await new FeedPublisher().PublishAsync(result.GetValue(_feedFolder)!, result.GetValue(_folders)!, key, PublishOptions(result), cancellationToken).ConfigureAwait(false);
        foreach (var added in published.Added)
        {
            Console.WriteLine($"added {added}");
        }

        foreach (var unchanged in published.Unchanged)
        {
            Console.WriteLine($"unchanged {unchanged}");
        }

        Console.WriteLine($"Index sequence {published.Sequence}: {published.Packages} package(s), signed by {key.KeyId}");
        return 0;
    }

    private async Task<int> ReindexAsync(ParseResult result, CancellationToken cancellationToken)
    {
        using var key = FeedSigningKey.Load(result.GetValue(_key)!);
        var (sequence, packages) = await new FeedPublisher().ReindexAsync(result.GetValue(_feedFolder)!, key, PublishOptions(result), cancellationToken).ConfigureAwait(false);
        Console.WriteLine($"Index sequence {sequence}: {packages} package(s)");
        return 0;
    }

    private async Task<int> VerifyAsync(ParseResult result, CancellationToken cancellationToken)
    {
        using var http = new HttpClient();
        var client = Client(result, http);
        var problems = await FeedVerifier.VerifyAsync(client, result.GetValue(_files), Console.WriteLine, cancellationToken).ConfigureAwait(false);
        foreach (var problem in problems)
        {
            Console.Error.WriteLine($"problem: {problem}");
        }

        Console.WriteLine(problems.Count == 0 ? "The feed is valid." : $"{problems.Count} problem(s).");
        return problems.Count == 0 ? 0 : 1;
    }

    private async Task<int> MirrorAsync(ParseResult result, CancellationToken cancellationToken)
    {
        using var http = new HttpClient();
        var prefix = result.GetValue(_filter);
        Func<string, bool>? filter = prefix is null ? null : id => id.StartsWith(prefix, StringComparison.Ordinal);
        var copied = await FeedMirror.CopyAsync(Client(result, http), result.GetValue(_out)!, filter, Console.WriteLine, cancellationToken).ConfigureAwait(false);
        Console.WriteLine($"Mirrored {copied} package file(s) to {result.GetValue(_out)}");
        return 0;
    }

    private FeedClient Client(ParseResult result, HttpClient http)
    {
        var options = new FeedClientOptions
        {
            TrustedRoots = [.. (result.GetValue(_roots) ?? []).Select(FeedPublicKey.Load)],
            AllowExpiredIndex = result.GetValue(_allowExpired),
        };
        return new FeedClient(FeedTransports.Create(result.GetValue(_feedLocation)!, http), options);
    }

    private FeedPublishOptions PublishOptions(ParseResult result) => new()
    {
        Name = result.GetValue(_name)!,
        Validity = TimeSpan.FromDays(result.GetValue(_validityDays)),
    };

    private VerifiedCatalog LoadCatalog(ParseResult result) =>
        VerifiedCatalogSerializer.Parse(File.ReadAllText(result.GetValue(_catalog) ?? FeedToolPaths.InRepository(FeedToolPaths.Catalog)));

    private FeedOverlay LoadOverlay(ParseResult result)
    {
        var path = result.GetValue(_overlay) ?? FeedToolPaths.InRepository(FeedToolPaths.Overlay);
        if (!File.Exists(path))
        {
            return new FeedOverlay();
        }

        try
        {
            return JsonSerializer.Deserialize(File.ReadAllBytes(path), FeedToolJsonContext.Default.FeedOverlay) ?? new FeedOverlay();
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException($"{path} is not a valid feed overlay: {exception.Message}", exception);
        }
    }

    private static PackageManifest? LatestManifest(string feedFolder, string id)
    {
        var indexPath = Path.Combine(feedFolder, FeedLayout.PackageIndex(id));
        if (!File.Exists(indexPath))
        {
            return null;
        }

        var index = FeedJson.Read(File.ReadAllBytes(indexPath), FeedJsonContext.Default.FeedPackageIndex, $"The package index of {id}");
        var latest = index.Versions.Select(entry => Sharplight.ModelFeed.Versioning.SemanticVersion.Parse(entry.Version)).Max();
        return latest is null ? null : FeedJson.ReadManifest(File.ReadAllBytes(Path.Combine(feedFolder, FeedLayout.Manifest(id, latest.ToString()))));
    }
}
