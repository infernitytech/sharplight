using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary>Runs the steps a <see cref="CliSettings"/> asks for: list devices, load, save, then benchmark, dump logits or generate.</summary>
internal static class CliApplication
{
    public static async Task<int> RunAsync(SampleRuntime runtime, CliSettings settings, CancellationToken cancellationToken)
    {
        if (settings.ListDevices)
        {
            DeviceLister.Print(runtime.Providers, Console.Out);
        }

        if (settings.ModelPath is not { } model)
        {
            return ExitCodes.Success;
        }

        string modelPath;
        using (var hub = new HubContext(runtime, settings.StoreRoot))
        {
            modelPath = await new ModelResolver(hub, settings.Pull).ResolveAsync(model, cancellationToken);
        }

        // Image decoding happens before the (slow) model load so a bad path fails fast.
        var images = ImageLoader.Load(settings.ImagePaths);
        using var loaded = CliModelLoader.Load(runtime, settings, modelPath, cancellationToken);

        if (settings.SavePath is { } savePath)
        {
            await PackageSaver.SaveAsync(runtime.Providers, loaded.Model, savePath, Console.Error, cancellationToken);
        }

        if (settings.BenchmarkTokens is { } benchmarkTokens)
        {
            var summary = new BenchmarkRunner(loaded.Session, benchmarkTokens, Console.Error, loaded.Model.DiskExperts).Run(cancellationToken);
            Console.WriteLine(summary);
        }
        else if (settings.LogitsPath is { } logitsPath)
        {
            LogitsDumper.Dump(loaded, settings.Generation.Prompt, logitsPath, Console.Out);
        }
        else
        {
            new TextGenerator(loaded, settings.Generation).Run(images, Console.Out, Console.Error, cancellationToken);
        }

        if (settings.BenchmarkTokens is null && loaded.Model.DiskExperts is { IsActive: true } disk)
        {
            Console.Error.WriteLine($"experts on disk ({disk.ReaderName}): {disk.Statistics}");
        }

        return ExitCodes.Success;
    }
}
