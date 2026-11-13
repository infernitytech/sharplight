using System.CommandLine;
using Sharplight.Hub;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary><c>list</c>: the models in the store.</summary>
internal sealed class ListCommand(StoreOptionSet store)
{
    public Command Build()
    {
        var command = new Command("list", "Lists the pulled models with their short names, sizes and whether they are precompiled.");
        command.Aliases.Add("ls");
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            Print(new ModelStore(store.ReadRoot(parseResult)));
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private static void Print(ModelStore store)
    {
        var models = store.List();
        if (models.Count == 0)
        {
            Console.WriteLine($"No models in {store.Root}. Find one with 'search', then 'pull owner/name[:tag]'.");
            return;
        }

        int nameWidth = Math.Max("NAME".Length, models.Max(model => model.ShortName.Length));
        int repositoryWidth = Math.Max("REPOSITORY".Length, models.Max(model => model.Name.Length));
        Console.WriteLine($"{"NAME".PadRight(nameWidth)}  {"REPOSITORY".PadRight(repositoryWidth)}  {"SIZE",10}  {"ARCH",-12}  {"PRECOMPILED",-11}  PULLED");
        foreach (var model in models)
        {
            var precompiled = model.Precompiled.Count > 0 ? "yes" : "no";
            Console.WriteLine($"{model.ShortName.PadRight(nameWidth)}  {model.Name.PadRight(repositoryWidth)}  {ByteSize.Format(model.PackageBytes),10}  "
                + $"{model.Architecture,-12}  {precompiled,-11}  {model.PulledAt.ToLocalTime():yyyy-MM-dd HH:mm}");
        }

        Console.WriteLine($"{models.Count} model(s), {ByteSize.Format(models.Sum(model => model.PackageBytes))} in {store.Root}");
    }
}
