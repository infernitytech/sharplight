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
/// </summary>
internal sealed class PackageBuilder(string storeRoot, Action<string> log)
{
    /// <summary>Builds <paramref name="plan"/> into <paramref name="outputFolder"/>.</summary>
    /// <param name="plan">The catalog variant.</param>
    /// <param name="outputFolder">Staging root; the package goes to <c>&lt;id&gt;-&lt;version&gt;/</c>.</param>
    /// <param name="feedFolder">The feed (for versioning), or <see langword="null"/>.</param>
    /// <param name="overlay">Feed-only data.</param>
    /// <param name="feedUrl">The public feed URL for the manifest, or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancellation.</param>
    /// <returns>The staging folder, or <see langword="null"/> when the feed already has an identical package.</returns>
    public async Task<string?> BuildAsync(CatalogInstallPlan plan, string outputFolder, string? feedFolder, FeedOverlay overlay, string? feedUrl, CancellationToken cancellationToken)
    {
        var id = ManifestFactory.PackageId(plan.Entry, plan.Variant);
        log($"{id}: building from {plan.Variant.Repository}@{plan.Variant.Revision[..12]}");
        string packagePath;
        using (var services = SharplightDefaults.CreateServiceProvider())
        {
            using var http = HubClientOptions.CreateHttpClient();
            var options = HubClientOptions.FromEnvironment();
            var puller = new ModelPuller(new HubClient(http, options), new FileDownloader(http, options), new ModelStore(storeRoot), services.GetRequiredService<ModelConverter>());
            var result = await puller.InstallAsync(plan, new PullOptions { Log = log }, cancellationToken).ConfigureAwait(false);
            packagePath = result.Manifest.PackagePath;
        }

        var inspection = PackageInspector.Inspect(packagePath);
        var described = await PackageFileHasher.DescribeAsync(packagePath, "pending.slm", cancellationToken: cancellationToken).ConfigureAwait(false);
        var engineVersion = EngineVersion();
        var draft = ManifestFactory.Create(plan, inspection, described, "1.0.0", overlay.Packages.GetValueOrDefault(id), engineVersion, feedUrl, DateTimeOffset.UtcNow);
        var versionPlan = VersionPlanner.Plan(feedFolder, draft);
        if (versionPlan is null)
        {
            log($"{id}: the feed already has this exact package (file and metadata); nothing to build.");
            return null;
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
