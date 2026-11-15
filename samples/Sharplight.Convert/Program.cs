using Sharplight.Convert;
using Sharplight.Samples;

SampleRuntime.RegisterProvidersAndImporters();

return await CommandRunner.RunAsync(ConvertCommandLine.Build(), args);
