// Survey harness: imports an ONNX model described by a reference manifest (same format as the ZooTests fixtures),
// runs it on cpu|vulkan, compares with the onnxruntime reference outputs and times it.
//   SurveyRunner <manifest.json> <cpu|vulkan> [--runs N] [--budget-s S] [--device D] [--batch B] [--bound name=value]
//                [--profile] [--reduced-precision] [--dump DIR]
// Prints exactly one line starting with "RESULT " followed by a JSON object.
using System.Text.Json;

var options = SurveyOptions.Parse(args);
var result = new Dictionary<string, object?>
{
    ["manifest"] = options.ManifestPath,
    ["provider"] = options.ProviderName,
    ["stage"] = "start",
};

int exitCode;
try
{
    exitCode = new SurveyRun(options, result).Execute();
}
catch (Exception e)
{
    result["error"] = SurveyRun.Truncate(e.GetType().Name + ": " + e.Message, 1500);
    exitCode = 1;
}

Console.Out.Flush();
Console.WriteLine("RESULT " + JsonSerializer.Serialize(result));
return exitCode;
