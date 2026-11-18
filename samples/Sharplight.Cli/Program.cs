using Microsoft.Extensions.DependencyInjection;
using Sharplight.Cli;
using Sharplight.Samples;

// The CUDA provider is registered after Vulkan: Vulkan stays the default GPU provider, CUDA runs with --provider cuda (docs/cuda.md).
using var runtime = new SampleRuntime(services => services.AddCudaProvider());
var options = new CliOptionSet(runtime);
var command = options.BuildCommand((settings, cancellationToken) => CliApplication.RunAsync(runtime, settings, cancellationToken));
return await CommandRunner.RunAsync(command, args);
