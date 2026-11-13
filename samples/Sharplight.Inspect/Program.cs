using Sharplight.Architectures;
using Sharplight.Gguf;
using Sharplight.Models;

// Loads a model through the adapter layer and dumps the resulting compute graph.
ModelAdapterRegistry.Register(new Qwen35Adapter());
ModelAdapterRegistry.Register(new NemotronHAdapter());
GgufImporter.Register();
var def = ModelAdapterRegistry.Build(ModelImporterRegistry.OpenSet(args[0]));
var g = def.Graph(args.Length > 1 ? args[1] : WellKnown.DecodeGraph);
Console.WriteLine($"{def.Name}: {def.Weights.Count} weights, {def.States.Count} states, graphs: {string.Join(", ", def.Graphs.Keys)}");
Console.WriteLine($"{g.Nodes.Count} nodes; ops: " + string.Join(", ", g.Nodes.GroupBy(n => n.Op.Name).OrderByDescending(x => x.Count()).Select(x => $"{x.Key}×{x.Count()}")));
var dump = g.Dump().Split('\n');
foreach (var l in dump.Take(80)) Console.WriteLine(l);
Console.WriteLine("...");
foreach (var l in dump.TakeLast(8)) Console.WriteLine(l);
