using Sharplight.OnnxRun;
using Sharplight.Samples;

var command = new OnnxRunCommand().Build((settings, cancellationToken) =>
{
    new OnnxRunner(settings, Console.Out).Run(cancellationToken);
    return Task.FromResult(ExitCodes.Success);
});
return await CommandRunner.RunAsync(command, args);
