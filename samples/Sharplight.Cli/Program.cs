using Sharplight.Cli;
using Sharplight.Samples;

SampleRuntime.RegisterProvidersAndImporters();

var options = new CliOptionSet();
var command = options.BuildCommand(CliApplication.RunAsync);
return await CommandRunner.RunAsync(command, args);
