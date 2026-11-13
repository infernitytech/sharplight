using System.CommandLine;
using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Samples;

namespace Sharplight.Inspect;

/// <summary>The <c>Inspect</c> command line: load a model through the adapter layer and dump the resulting compute graph.</summary>
internal sealed class InspectCommand
{
    private readonly Argument<string> _model = new("model")
    {
        Description = "Model to load: a .gguf file or a directory the registered importers understand.",
    };

    private readonly Argument<string> _graph = new("graph")
    {
        Description = "Name of the graph to dump.",
        Arity = ArgumentArity.ZeroOrOne,
        DefaultValueFactory = _ => WellKnown.DecodeGraph,
    };

    public RootCommand Build()
    {
        var command = new RootCommand("Loads a model through the adapter layer and prints its compute graph (op histogram, first and last nodes).");
        command.Arguments.Add(_model);
        command.Arguments.Add(_graph);
        command.SetAction(CommandRunner.Guard((parseResult, _) =>
        {
            Run(parseResult.GetValue(_model)!, parseResult.GetValue(_graph)!);
            return Task.FromResult(ExitCodes.Success);
        }));
        return command;
    }

    private static void Run(string modelPath, string graphName)
    {
        if (!File.Exists(modelPath) && !Directory.Exists(modelPath))
        {
            throw new SampleException($"Model '{modelPath}' does not exist.");
        }

        BuiltinAdapters.RegisterAll();
        GgufImporter.Register();
        var definition = BuildDefinition(modelPath);
        if (!definition.Graphs.ContainsKey(graphName))
        {
            throw new SampleException($"The model has no graph '{graphName}'; available graphs: {string.Join(", ", definition.Graphs.Keys)}.");
        }

        new GraphReport(definition, definition.Graph(graphName), Console.Out).Write();
    }

    private static ModelDefinition BuildDefinition(string modelPath)
    {
        try
        {
            return ModelAdapterRegistry.Build(ModelImporterRegistry.OpenSet(modelPath));
        }
        catch (NotSupportedException error)
        {
            throw new SampleException(error.Message, error);
        }
    }
}
