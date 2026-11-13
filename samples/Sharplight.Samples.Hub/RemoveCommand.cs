using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary><c>rm &lt;model...&gt;</c>: deletes models from the store.</summary>
public sealed class RemoveCommand(StoreOptionSet store)
{
    private readonly Argument<string[]> _models = new("model")
    {
        Description = "Stored model: owner/name[:tag] or its short name (see list).",
        Arity = ArgumentArity.OneOrMore,
    };

    public Command Build()
    {
        var command = new Command("rm", "Deletes pulled models (package, manifest and any kept source files).");
        command.Aliases.Add("remove");
        command.Arguments.Add(_models);
        command.SetAction(HubGuard.Guard((parseResult, _) =>
        {
            var modelStore = new ModelStore(store.ReadRoot(parseResult));
            foreach (var name in parseResult.GetValue(_models)!)
            {
                var model = modelStore.Find(name)
                    ?? throw new SampleException($"'{name}' is not in the model store {modelStore.Root} (see list).", ExitCodes.InvalidInput);
                modelStore.Remove(model);
                Console.WriteLine($"Removed {model.Name} ({ByteSize.Format(model.PackageBytes)})");
            }

            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }
}
