using Sharplight.Convert;
using Sharplight.Llm;
using Sharplight.Onnx;
using Sharplight.Samples;

SampleRuntime.RegisterProvidersAndImporters();
OnnxImporter.Register();
BuiltinAdapters.RegisterAll();

return await CommandRunner.RunAsync(ConvertCommandLine.Build(), args);
