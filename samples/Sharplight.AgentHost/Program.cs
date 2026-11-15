using Sharplight.AgentHost;
using Sharplight.Samples;

var options = new AgentHostOptionSet();
var command = options.BuildCommand(AgentHostApplication.RunAsync);
return await CommandRunner.RunAsync(command, args);
