using Sharplight.Cli;
using Sharplight.Samples;

using var runtime = new SampleRuntime();
var options = new CliOptionSet(runtime);
var command = options.BuildCommand((settings, cancellationToken) => CliApplication.RunAsync(runtime, settings, cancellationToken));
return await CommandRunner.RunAsync(command, args);
