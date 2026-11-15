using Sharplight.Samples;
using Sharplight.Vision.Sample;

var options = new VisionOptionSet();
var command = options.BuildCommand(VisionApplication.ClassifyAsync, VisionApplication.DetectAsync);
return await CommandRunner.RunAsync(command, args);
