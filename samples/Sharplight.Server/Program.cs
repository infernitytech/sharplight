using Sharplight.Llm;
using Sharplight.Onnx;
using Sharplight.Samples;
using Sharplight.Server;

SampleRuntime.RegisterProvidersAndImporters();
OnnxImporter.Register();
BuiltinAdapters.RegisterAll();

return await CommandRunner.RunAsync(ServerCommandLine.Build(), args);
