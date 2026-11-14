// Imports an ONNX model described by a reference manifest, runs it on cpu|vulkan, compares the outputs with the
// onnxruntime reference and times it. Prints exactly one line starting with "RESULT " followed by a JSON object.
using System.Text.Json;
using OnnxBench;

var arguments = BenchArguments.Parse(args);
if (arguments is null)
{
    Console.Error.WriteLine(BenchArguments.Usage);
    return 2;
}

string? modelRoot = Environment.GetEnvironmentVariable("SHARPLIGHT_ONNX_MODELS");
if (string.IsNullOrEmpty(modelRoot))
{
    Console.Error.WriteLine("set SHARPLIGHT_ONNX_MODELS to the ONNX model directory");
    return 2;
}

var result = new Dictionary<string, object?> { ["manifest"] = Path.GetFileName(arguments.ManifestPath), ["provider"] = arguments.Provider };
int exitCode = new ModelBench(arguments, ReferenceManifest.Load(arguments.ManifestPath, modelRoot), result).Run();
Console.Out.Flush();
Console.WriteLine("RESULT " + JsonSerializer.Serialize(result));
return exitCode;
