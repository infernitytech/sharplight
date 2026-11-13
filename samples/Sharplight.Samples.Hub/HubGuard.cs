using System.CommandLine;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary>Like <see cref="CommandRunner.Guard"/>, and also reports <see cref="HubException"/>s as one-line errors.</summary>
public static class HubGuard
{
    public static Func<ParseResult, CancellationToken, Task<int>> Guard(Func<ParseResult, CancellationToken, Task<int>> action)
    {
        return CommandRunner.Guard(async (parseResult, cancellationToken) =>
        {
            try
            {
                return await action(parseResult, cancellationToken);
            }
            catch (HubException error)
            {
                throw new SampleException(error.Message, error);
            }
        });
    }
}
