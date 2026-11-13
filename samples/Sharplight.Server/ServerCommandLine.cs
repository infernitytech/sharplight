using System.CommandLine;

namespace Sharplight.Server;

/// <summary>
/// The <c>sharplight-server</c> command line: serving (the root command and <c>serve</c>) plus model management in the style of
/// ollama (<c>search</c>, <c>pull</c>, <c>list</c>, <c>show</c>, <c>rm</c>, <c>run</c>).
/// </summary>
internal static class ServerCommandLine
{
    public static RootCommand Build()
    {
        var store = new StoreOptionSet();
        var root = new RootCommand("Serves models over the OpenAI and/or Anthropic HTTP API, and pulls, lists and runs models from Hugging Face. "
            + "Without a subcommand it serves the given models (like 'serve').");
        root.Options.Add(store.ModelsDirectory);
        new ServeOptionSet(store).Configure(root, ServeApplication.RunAsync);

        var serve = new Command("serve", "Serves one or several models; models that are not in the store are pulled first.");
        new ServeOptionSet(store).Configure(serve, ServeApplication.RunAsync);
        root.Subcommands.Add(serve);
        root.Subcommands.Add(new RunCommand(store).Build());
        root.Subcommands.Add(new PullCommand(store).Build());
        root.Subcommands.Add(new SearchCommand(store).Build());
        root.Subcommands.Add(new ListCommand(store).Build());
        root.Subcommands.Add(new ShowCommand(store).Build());
        root.Subcommands.Add(new RemoveCommand(store).Build());
        return root;
    }
}
