using Sharplight.Samples;
using Sharplight.Server;

using var runtime = new SampleRuntime();
return await CommandRunner.RunAsync(ServerCommandLine.Build(runtime), args);
