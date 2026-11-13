using Sharplight.Samples;
using Sharplight.Serve;

SampleRuntime.RegisterProvidersAndImporters();

var options = new ServeOptionSet();
var command = options.BuildCommand(ServeApplication.RunAsync);
return await CommandRunner.RunAsync(command, args);
