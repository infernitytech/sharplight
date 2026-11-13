using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary><c>pull &lt;owner/name[:tag]&gt;</c>: downloads a model, converts it to a package and stores it.</summary>
public sealed class PullCommand(StoreOptionSet store)
{
    private readonly Argument<string[]> _models = new("model")
    {
        Description = "Hugging Face repository, owner/name[:tag]. The tag picks a GGUF quantization (Q4_K_M, Q8_0, ...) or file; default Q4_K_M, else Q8_0.",
        Arity = ArgumentArity.OneOrMore,
    };

    private readonly PullOptionSet _pull = new(explicitPull: true);
    private readonly RuntimeOptionSet _runtime = new();

    public Command Build()
    {
        var command = new Command("pull", "Downloads a model from Hugging Face (resumable, checksum verified), converts it to a .slm package, "
            + "precompiles it for the local GPU and deletes the downloaded files.");
        command.Arguments.Add(_models);
        _pull.AddTo(command);
        command.Options.Add(_runtime.Cpu);
        command.Options.Add(_runtime.Device);
        command.Options.Add(_runtime.Context);
        command.Options.Add(_runtime.Batch);
        command.SetAction(HubGuard.Guard(RunAsync));
        return command;
    }

    private async Task<int> RunAsync(ParseResult parseResult, CancellationToken cancellationToken)
    {
        var references = parseResult.GetValue(_models)!.Select(Parse).ToList();
        bool cpu = parseResult.GetValue(_runtime.Cpu);
        int device = parseResult.GetValue(_runtime.Device);
        int context = parseResult.GetValue(_runtime.Context);
        int batch = parseResult.GetValue(_runtime.Batch);
        var settings = _pull.Read(parseResult, log => PrecompileTargets.Create(true, cpu, device, context, batch, [], log));

        using var hub = new HubContext(store.ReadRoot(parseResult));
        foreach (var reference in references)
        {
            var printer = new PullProgressPrinter();
            var result = await hub.Puller.PullAsync(reference, settings.ToOptions(printer), cancellationToken);
            Report(result);
        }

        return ExitCodes.Success;
    }

    private static ModelReference Parse(string text)
    {
        return ModelReference.TryParse(text, out var reference)
            ? reference
            : throw new SampleException($"'{text}' is not a Hugging Face model name; expected owner/name[:tag], e.g. Qwen/Qwen3-0.6B-GGUF:Q8_0.", ExitCodes.InvalidInput);
    }

    private static void Report(PullResult result)
    {
        var manifest = result.Manifest;
        if (result.UpToDate)
        {
            Console.WriteLine($"{manifest.Name} ({manifest.ShortName}) is up to date: {manifest.PackagePath}");
            return;
        }

        Console.WriteLine($"{manifest.Name} ({manifest.ShortName}): {manifest.PackagePath}, {ByteSize.Format(manifest.PackageBytes)}");
        var precompile = result.PrecompileTime > TimeSpan.Zero ? $"precompile {result.PrecompileTime.TotalSeconds:F1} s" : "not precompiled";
        Console.WriteLine($"  download {ByteSize.Format(result.DownloadedBytes)} in {result.DownloadTime.TotalSeconds:F1} s, "
            + $"convert {result.ConvertTime.TotalSeconds:F1} s, {precompile}");
    }
}
