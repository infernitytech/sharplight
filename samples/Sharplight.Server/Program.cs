using Sharplight.Samples;
using Sharplight.Server;

SampleRuntime.RegisterProvidersAndImporters();

return await CommandRunner.RunAsync(ServerCommandLine.Build(), args);
