using Sharplight.Convert;
using Sharplight.Samples;

using var runtime = new SampleRuntime();
return await CommandRunner.RunAsync(ConvertCommandLine.Build(runtime), args);
