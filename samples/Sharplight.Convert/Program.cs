using Sharplight.Convert;
using Sharplight.Import.Onnx;
using Sharplight.Llm;
using Sharplight.Samples;

SampleRuntime.RegisterProvidersAndImporters();
OnnxImporter.Register();
BuiltinAdapters.RegisterAll();

return await CommandRunner.RunAsync(ConvertCommandLine.Build(), args);
