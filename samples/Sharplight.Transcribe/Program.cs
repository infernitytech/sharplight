using Sharplight.Samples;
using Sharplight.Transcribe;

var options = new TranscribeOptionSet();
var command = options.BuildCommand(TranscribeApplication.RunAsync);
return await CommandRunner.RunAsync(command, args);
