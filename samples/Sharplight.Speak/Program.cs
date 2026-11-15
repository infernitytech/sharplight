using Sharplight.Samples;
using Sharplight.Speak;

SpeakSettings? settings;
try
{
    settings = SpeakSettings.Parse(args);
}
catch (ArgumentException error)
{
    Console.Error.WriteLine(error.Message);
    Console.Error.WriteLine(SpeakSettings.Usage);
    return ExitCodes.UsageError;
}

if (settings is null)
{
    Console.Error.WriteLine(SpeakSettings.Usage);
    return ExitCodes.UsageError;
}

using var cancellation = new CancellationTokenSource();
Console.CancelKeyPress += (_, e) =>
{
    e.Cancel = true;
    cancellation.Cancel();
};
try
{
    return await SpeakApplication.RunAsync(settings, cancellation.Token);
}
catch (OperationCanceledException)
{
    return ExitCodes.Cancelled;
}
catch (SampleException error)
{
    Console.Error.WriteLine(error.Message);
    return ExitCodes.RuntimeError;
}
