// Loads an ONNX model, runs it on the CPU or Vulkan provider and prints the top-5 of the first output.
//   dotnet run --project samples/Sharplight.OnnxRun -- model.onnx [--provider cpu|vulkan] [--input x.bin]
//       [--batch 1] [--shape name=-1x3x224x224] [--runs 5] [--device 0] [--dump-graph]
using System.Diagnostics;
using System.Globalization;
using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Runtime;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string name, string? def = null) { int i = argv.IndexOf(name); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : def; }
bool Flag(string name) => argv.Contains(name);
if (argv.Count == 0 || argv[0].StartsWith("--"))
{
    Console.Error.WriteLine("usage: OnnxRun <model.onnx> [--provider cpu|vulkan] [--input f32.bin] [--batch N] [--shape name=-1x3x224x224]... [--runs R] [--device D] [--dump-graph]");
    return 1;
}

var path = argv[0];
string providerName = Opt("--provider", "cpu")!;
int batch = int.Parse(Opt("--batch", "1")!);
int runs = int.Parse(Opt("--runs", "5")!);

var options = new OnnxImportOptions();
for (int i = 0; i < argv.Count - 1; i++)
    if (argv[i] == "--shape")
    {
        var kv = argv[i + 1].Split('=', 2);
        options.InputShapes[kv[0]] = kv[1].Split('x').Select(s => long.Parse(s, CultureInfo.InvariantCulture)).ToArray();
    }

var sw = Stopwatch.StartNew();
OnnxImporter.Register();
var source = OnnxImporter.Open(path, options);
var def = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = source });
var graph = def.Graph(OnnxGraphAdapter.MainGraph);
Console.WriteLine($"imported {Path.GetFileName(path)} in {sw.ElapsedMilliseconds} ms: {graph.Nodes.Count} nodes, {def.Weights.Count} weights, " +
                  $"producer {source.Get("onnx.producer_name", "?")}, opset {source.Get("onnx.opset", 0L)}");
if (Flag("--dump-graph")) Console.WriteLine(graph.Dump());

IExecutionProvider provider = providerName switch
{
    "cpu" => new CpuExecutionProvider(),
    "vulkan" => new VulkanProvider(),
    _ => throw new ArgumentException($"unknown provider '{providerName}'"),
};
var avail = provider.CheckAvailability();
if (!avail.IsAvailable) { Console.Error.WriteLine($"{providerName} unavailable: {avail.Reason}"); return 2; }
var devInfo = provider.GetDevices()[int.Parse(Opt("--device", "0")!)];
Console.WriteLine($"device: {devInfo}");
using var device = provider.OpenDevice(devInfo);

var compileOpts = new CompileOptions { SymbolBounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = batch } };
sw.Restart();
using var compiled = CompiledModel.Compile(graph, new PlacementOptions
{
    Devices = [new DeviceSlot { Device = device }],
    Compile = compileOpts,
    ReservePerDevice = 256L << 20,
});
Console.WriteLine($"compiled in {sw.ElapsedMilliseconds} ms ({compiled.Plan.Segments.Count} segment(s))");

var run = new RunParameters();
run.Symbols[OnnxGraphAdapter.BatchSymbol] = batch;
var rng = new Random(1234);
var inputFile = Opt("--input");
bool first = true;
foreach (var input in graph.Inputs)
{
    long count = input.Shape.Elements(run.Symbols);
    var data = new float[count];
    if (first && inputFile is not null)
    {
        var bytes = File.ReadAllBytes(inputFile);
        if (bytes.Length != count * 4) { Console.Error.WriteLine($"{inputFile}: {bytes.Length} bytes, expected {count * 4} for {input}"); return 3; }
        Buffer.BlockCopy(bytes, 0, data, 0, bytes.Length);
    }
    else for (long i = 0; i < count; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
    first = false;
    compiled.SetInput<float>(input, data);
}

compiled.Run(run);   // warm-up
var times = new List<double>();
for (int r = 0; r < runs; r++)
{
    sw.Restart();
    compiled.Run(run);
    var (outName0, outV0) = graph.Outputs.First();
    var tmp = new float[outV0.Shape.Elements(run.Symbols)];
    compiled.GetOutput<float>(outName0, tmp);   // forces completion on async providers
    times.Add(sw.Elapsed.TotalMilliseconds);
}
Console.WriteLine($"run: min {times.Min():F2} ms, median {times.OrderBy(t => t).ElementAt(times.Count / 2):F2} ms, mean {times.Average():F2} ms over {runs} runs (batch {batch})");

var (outName, outVal) = graph.Outputs.First();
var result = new float[outVal.Shape.Elements(run.Symbols)];
compiled.GetOutput<float>(outName, result);
if (Opt("--out") is { } outPath) { var ob = new byte[result.Length * 4]; Buffer.BlockCopy(result, 0, ob, 0, ob.Length); File.WriteAllBytes(outPath, ob); }
Console.WriteLine($"output '{outName}' {outVal.Shape}: top-5 of batch item 0");
int per = result.Length / batch;
foreach (var i in Enumerable.Range(0, per).OrderByDescending(i => result[i]).Take(5))
    Console.WriteLine($"  [{i}] {result[i]:F5}");
return 0;
