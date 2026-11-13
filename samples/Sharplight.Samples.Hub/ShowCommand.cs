using System.CommandLine;
using Sharplight.Hub;
using Sharplight.Package;

namespace Sharplight.Samples.Hub;

/// <summary><c>show &lt;model&gt;</c>: where a stored model came from and what its package contains.</summary>
public sealed class ShowCommand(StoreOptionSet store)
{
    private const int MaxPropertyLength = 80;

    private readonly Argument<string> _model = new("model") { Description = "Stored model: owner/name[:tag] or its short name (see list)." };

    public Command Build()
    {
        var command = new Command("show", "Shows a pulled model: source repository and commit, files, package contents and precompiled artifact sets.");
        command.Arguments.Add(_model);
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var modelStore = new ModelStore(store.ReadRoot(parseResult));
            var name = parseResult.GetValue(_model)!;
            var model = modelStore.Find(name)
                ?? throw new SampleException($"'{name}' is not in the model store {modelStore.Root} (see list).", ExitCodes.InvalidInput);
            Print(model);
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private static void Print(ModelManifest model)
    {
        Console.WriteLine($"Name:         {model.ShortName}");
        Console.WriteLine($"Repository:   {model.Name} ({model.Format.ToString().ToLowerInvariant()}), revision {model.Revision} = commit {model.Commit}");
        Console.WriteLine($"Pulled:       {model.PulledAt.ToLocalTime():yyyy-MM-dd HH:mm}");
        Console.WriteLine($"Package:      {model.PackagePath} ({ByteSize.Format(model.PackageBytes)})");
        Console.WriteLine($"Source files: {ByteSize.Format(model.SourceBytes)}{(model.SourceKept ? " (kept in source/)" : " (deleted after conversion)")}");
        foreach (var file in model.SourceFiles)
        {
            Console.WriteLine($"  {file.Path,-60} {ByteSize.Format(file.Size),10}  {file.Sha256 ?? ""}");
        }

        PrintPackage(model.PackagePath);
    }

    private static void PrintPackage(string path)
    {
        PackageInfo info;
        try
        {
            info = ModelPackageReader.Open(path);
        }
        catch (Exception error) when (error is InvalidDataException or IOException)
        {
            Console.WriteLine($"Package:      cannot be read ({error.Message}); pull the model again with --force.");
            return;
        }

        var definition = info.Definition;
        Console.WriteLine($"Model:        {definition.Name} ({definition.Architecture}), {definition.Graphs.Count} graph(s): {string.Join(", ", definition.Graphs.Select(graph => graph.Name))}");
        Console.WriteLine($"Weights:      {info.Weights.Count}, {ByteSize.Format(info.WeightBytes)}; "
            + string.Join(", ", info.Weights.GroupBy(weight => weight.Type.Name).OrderByDescending(group => group.Sum(weight => weight.ByteSize)).Select(group => $"{group.Key} x{group.Count()}")));
        foreach (var key in definition.Properties.Keys.Order(StringComparer.Ordinal))
        {
            Console.WriteLine($"  {key} = {DefinitionInfo.FormatValue(definition.Properties[key], maxLength: MaxPropertyLength)}");
        }

        Console.WriteLine(info.ArtifactSets.Count == 0 ? "Precompiled:  none (kernels are compiled at the first load)" : $"Precompiled:  {info.ArtifactSets.Count} artifact set(s)");
        foreach (var set in info.ArtifactSets)
        {
            Console.WriteLine($"  {set.Provider}/{set.Tier}: {set.EntryCount} entries, {ByteSize.Format(set.TotalBytes)}");
        }
    }
}
