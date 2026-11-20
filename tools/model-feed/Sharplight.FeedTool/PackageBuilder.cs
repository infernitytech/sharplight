using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.ModelFeed;
using Sharplight.ModelFeed.Publishing;
using Sharplight.Package;

namespace Sharplight.FeedTool;

/// <summary>
/// Builds one feed package from a verified-catalog variant: downloads exactly the pinned files (checksums verified), converts them
/// with companions merged into one <c>.slm</c> as <see cref="ModelPuller.InstallAsync"/> does, opens the result, hashes and chunks
/// it, chooses the version and writes a staging folder with the package file and <c>manifest.json</c> (signed when published).
/// Unless the store is kept, the converted file is moved (not copied) into the staging folder and the model's store folder is removed,
/// so a build needs disk for the sources plus one package, and nothing of an entry stays behind for the next one.
/// </summary>
internal sealed class PackageBuilder(string storeRoot, Action<string> log, bool keepStore = false)
{
    /// <summary>Builds <paramref name="plan"/> into <paramref name="outputFolder"/>.</summary>
    /// <param name="plan">The catalog variant.</param>
    /// <param name="outputFolder">Staging root; the package goes to <c>&lt;id&gt;-&lt;version&gt;/</c>.</param>
    /// <param name="feedFolder">The feed (for versioning), or <see langword="null"/>.</param>
    /// <param name="overlay">Feed-only data.</param>
    /// <param name="feedUrl">The public feed URL for the manifest, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The staging folder; it holds only the stored manifest when the feed already has an identical package.</returns>
    public async Task<string> BuildAsync(CatalogInstallPlan plan, string outputFolder, string? feedFolder, FeedOverlay overlay, string? feedUrl, CancellationToken cancellationToken)
    {
        var id = ManifestFactory.PackageId(plan.Entry, plan.Variant);
        log($"{id}: building from {plan.Variant.Repository}@{plan.Variant.Revision[..12]}");
        string packagePath;
        string storeDirectory;
        bool ownsStoreCopy;
        using (var services = SharplightDefaults.CreateServiceProvider())
        {
            using var http = HubClientOptions.CreateHttpClient();
            var options = HubClientOptions.FromEnvironment();
            var puller = new ModelPuller(new HubClient(http, options), new FileDownloader(http, options), new ModelStore(storeRoot), services.GetRequiredService<ModelConverter>());
            var result = await puller.InstallAsync(plan, new PullOptions { Log = log }, cancellationToken).ConfigureAwait(false);
            packagePath = result.Manifest.PackagePath;
            storeDirectory = result.Manifest.Directory;

            // A model that was in the store before this build belongs to someone else: copy it and leave it.
            ownsStoreCopy = !keepStore && !result.UpToDate;
        }

        try
        {
            return await StageAsync(plan, id, packagePath, ownsStoreCopy, outputFolder, feedFolder, overlay, feedUrl, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            if (ownsStoreCopy && Directory.Exists(storeDirectory))
            {
                Directory.Delete(storeDirectory, recursive: true);
                log($"{id}: removed the store copy {storeDirectory}");
            }
        }
    }

    private async Task<string> StageAsync(CatalogInstallPlan plan, string id, string packagePath, bool move, string outputFolder, string? feedFolder, FeedOverlay overlay, string? feedUrl, CancellationToken cancellationToken)
    {

        var inspection = PackageInspector.Inspect(packagePath);
        var described = await PackageFileHasher.DescribeAsync(packagePath, "pending.slm", cancellationToken: cancellationToken).ConfigureAwait(false);
        var engineVersion = EngineVersion();
        var draft = ManifestFactory.Create(plan, inspection, described, "1.0.0", overlay.For(plan.Entry.Id, id), engineVersion, feedUrl, DateTimeOffset.UtcNow);
        var versionPlan = VersionPlanner.Plan(feedFolder, draft);
        if (versionPlan is null)
        {
            // Stage the stored manifest alone: publishing lists the version in a channel that does not list it yet (a run interrupted
            // after the package objects were stored) and reports it unchanged otherwise.
            var latest = VersionPlanner.LatestManifest(feedFolder!, id)!;
            var relist = Path.Combine(outputFolder, $"{id}-{latest.Version}");
            Directory.CreateDirectory(relist);
            File.Copy(Path.Combine(feedFolder!, FeedLayout.Manifest(id, latest.Version)), Path.Combine(relist, FeedLayout.ManifestFileName), overwrite: true);
            log($"{id}: the feed already stores this exact package (file and metadata) as {latest.Version}; staged its manifest only, so publishing lists it where it is missing.");
            return relist;
        }

        var version = versionPlan.Version;
        var fileName = FeedLayout.PackageFileName(id, version);
        var folder = Path.Combine(outputFolder, $"{id}-{version}");
        Directory.CreateDirectory(folder);
        var manifest = draft with
        {
            Version = version,
            Files = [described with { Name = fileName }],
        };
        if (versionPlan.ReusesPayloadOf is { } earlier)
        {
            // Same package file, new manifest: the file is stored once, the feed copies it under the new version.
            log($"{id}: the package file is unchanged but the manifest metadata differs; publishing {version} (manifest only) that reuses the file of {earlier.Version}.");
            var reuse = new PayloadReuse { Version = earlier.Version, File = earlier.PrimaryFile!.Name };
            await File.WriteAllBytesAsync(Path.Combine(folder, FeedLayout.PayloadReuseFileName), FeedJson.Write(reuse, FeedJsonContext.Default.PayloadReuse), cancellationToken).ConfigureAwait(false);
        }
        else if (move)
        {
            File.Move(packagePath, Path.Combine(folder, fileName), overwrite: true);
        }
        else
        {
            File.Copy(packagePath, Path.Combine(folder, fileName), overwrite: true);
        }

        await File.WriteAllBytesAsync(Path.Combine(folder, FeedLayout.ManifestFileName), FeedJson.Write(manifest, FeedJsonContext.Default.PackageManifest), cancellationToken).ConfigureAwait(false);
        log($"{id} {version}: {described.Size} bytes, sha256 {described.Sha256[..16]}…, staged in {folder}");
        return folder;
    }

    /// <summary>The engine version (informational version of the package reader's assembly, without build metadata).</summary>
    public static string EngineVersion()
    {
        var attribute = typeof(ModelPackageReader).Assembly.GetCustomAttributes(typeof(System.Reflection.AssemblyInformationalVersionAttribute), false)
            .OfType<System.Reflection.AssemblyInformationalVersionAttribute>().FirstOrDefault();
        var version = attribute?.InformationalVersion ?? "0.0.0";
        var plus = version.IndexOf('+', StringComparison.Ordinal);
        return plus < 0 ? version : version[..plus];
    }
}
