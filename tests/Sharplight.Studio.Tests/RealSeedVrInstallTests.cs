using Microsoft.Extensions.DependencyInjection;
using Sharplight.Hub;
using Sharplight.Hub.Catalog;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Vision;
using Sharplight.VideoEnhancement;

namespace Sharplight.Studio.Tests;

/// <summary>
/// Installs the official SeedVR2 3B the way Studio does (the committed catalog's <c>seedvr2-3b</c> through <see cref="HubService.InstallAsync"/>,
/// and the Hub browser's pull of <c>ByteDance-Seed/SeedVR2-3B</c> through <see cref="HubService.PullAsync"/>, both with Studio's default
/// precompile after the download), then opens the package as the library does and enhances a still image on the default device as the
/// Enhance page does. Runs only when <c>SHARPLIGHT_STUDIO_SEEDVR_SOURCE</c> names a folder with the official files (they are linked into
/// the store so nothing is downloaded again; missing files are downloaded) and <c>SHARPLIGHT_STUDIO_SEEDVR_STORE</c> a store folder on a
/// volume with about 25 GB free. Opens a GPU: run it under the machine's GPU lock.
/// </summary>
public sealed class RealSeedVrInstallTests
{
    private const string SourceVariable = "SHARPLIGHT_STUDIO_SEEDVR_SOURCE";
    private const string StoreVariable = "SHARPLIGHT_STUDIO_SEEDVR_STORE";
    private const string Repository = "ByteDance-Seed/SeedVR2-3B";

    // ModelStore's layout of a model's downloaded sources (internal to Sharplight.Hub).
    private const string SourceDirectoryName = "source";
    private const string CommitFileName = ".commit";

    private static readonly string[] s_files = ["seedvr2_ema_3b.pth", "ema_vae.pth", "pos_emb.pt", "neg_emb.pt"];

    [Fact]
    public async Task CatalogInstall_And_HubPull_LoadAndEnhance()
    {
        var source = Environment.GetEnvironmentVariable(SourceVariable);
        var storeRoot = Environment.GetEnvironmentVariable(StoreVariable);
        Assert.SkipWhen(string.IsNullOrEmpty(source) || string.IsNullOrEmpty(storeRoot), $"set {SourceVariable} and {StoreVariable} to run");
        var catalog = new VerifiedCatalogSource(null, null);
        var entry = (await catalog.GetAsync(TestContext.Current.CancellationToken)).Find("seedvr2-3b")!;
        var plan = CatalogInstallPlan.For(entry, entry.DefaultVariant!);
        using var container = new ServiceCollection().AddSharplight().AddStudioProviders().BuildServiceProvider();
        var engine = EngineServices.From(container);
        var settings = new FakeSettings(new AppSettings(), storeRoot);
        using var hub = new HubService(settings, engine);
        var store = new ModelStore(storeRoot);
        var log = new List<string>();
        var progress = new SynchronousProgress<PullUpdate>(update => log.Add(update.Message));

        Seed(store.GetModelDirectory(Repository, plan.StoreTag), plan.Variant.Revision, source!);
        var installed = await hub.InstallAsync(plan, progress, TestContext.Current.CancellationToken);
        Assert.Equal(ModelFormat.PyTorch, installed.Format);
        await CheckAsync(engine, installed);

        Seed(store.GetModelDirectory(Repository, ""), plan.Variant.Revision, source!);
        var pulled = await hub.PullAsync(new ModelReference(Repository, null), progress, TestContext.Current.CancellationToken);
        Assert.Equal(ModelFormat.PyTorch, pulled.Format);
        await CheckAsync(engine, pulled);
    }

    /// <summary>Links the official files into the store's source folder of a model, as a finished download at <paramref name="commit"/> leaves them.</summary>
    private static void Seed(string directory, string commit, string source)
    {
        var files = Path.Combine(directory, SourceDirectoryName, "SeedVR2-3B");
        Directory.CreateDirectory(files);
        File.WriteAllText(Path.Combine(directory, SourceDirectoryName, CommitFileName), commit);
        foreach (var name in s_files)
        {
            var target = Path.Combine(files, name);
            if (!File.Exists(target) && File.Exists(Path.Combine(source, name)))
            {
                File.CreateSymbolicLink(target, Path.GetFullPath(Path.Combine(source, name)));
            }
        }
    }

    private static async Task CheckAsync(EngineServices engine, ModelManifest manifest)
    {
        var inspected = ModelInspector.Inspect(manifest.PackagePath);
        Assert.Equal(ModelKinds.VideoEnhancement, inspected.Kind);
        using var runtime = new EngineModelRuntime(engine, () => new ResidencySettings());
        var model = await runtime.LoadAsync(new ModelLoadRequest("seedvr2", manifest.PackagePath, new LoadSettings(), [], Kind: inspected.Kind), null, TestContext.Current.CancellationToken);
        try
        {
            using var lease = await model.AcquireClientAsync<IVideoEnhancer>(TestContext.Current.CancellationToken);
            var pixels = new byte[64 * 48 * 3];
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = (byte)(i * 7 % 251);
            }

            var result = await lease.Client.EnhanceAsync(VideoEnhancementRequest.FromImage(new ImageInput(64, 48, pixels)),
                new VideoEnhancementOptions { Width = 128, Height = 96, Seed = 42 }, TestContext.Current.CancellationToken);

            var frame = Assert.Single(result.Frames);
            Assert.Equal(128, frame.Width);
            Assert.Equal(96, frame.Height);
        }
        finally
        {
            await runtime.UnloadAsync(model.Id);
        }
    }
}
