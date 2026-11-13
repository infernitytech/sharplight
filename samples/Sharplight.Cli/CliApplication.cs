using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary>Runs the steps a <see cref="CliSettings"/> asks for: list devices, load, save, then benchmark, dump logits or generate.</summary>
internal static class CliApplication
{
    public static async Task<int> RunAsync(CliSettings settings, CancellationToken cancellationToken)
    {
        if (settings.ListDevices)
        {
            DeviceLister.Print(Console.Out);
        }

        if (settings.ModelPath is not { } model)
        {
            return ExitCodes.Success;
        }

        string modelPath;
        using (var hub = new HubContext(settings.StoreRoot))
        {
            modelPath = await new ModelResolver(hub, settings.Pull).ResolveAsync(model, cancellationToken);
        }

        // Image decoding happens before the (slow) model load so a bad path fails fast.
        var images = ImageLoader.Load(settings.ImagePaths);
        using var loaded = ModelLoader.Load(settings, modelPath);

        if (settings.SavePath is { } savePath)
        {
            PackageSaver.Save(loaded.Model, savePath, Console.Error);
        }

        if (settings.BenchmarkTokens is { } benchmarkTokens)
        {
            var summary = new BenchmarkRunner(loaded.Session, benchmarkTokens, Console.Error).Run(cancellationToken);
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

        return ExitCodes.Success;
    }
}
