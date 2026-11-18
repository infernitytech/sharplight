using System.Runtime.CompilerServices;
using Sharplight.Extensions.AI;

namespace Sharplight.Serving.Tests;

/// <summary>A <see cref="ITextGenerationBackend"/> that streams a fixed output one character per piece (like one token per step).</summary>
internal sealed class ScriptedTextBackend(string template, string output) : ITextGenerationBackend
{
    public string ModelName => "scripted";

    public string? ChatTemplate => template;

    public string? BosToken => null;

    public string? EosToken => null;

    public bool SupportsImages => false;

    public int ContextLength => 4096;

    /// <summary>The prompt of the last request.</summary>
    public string? LastPrompt { get; private set; }

    public async IAsyncEnumerable<string> GenerateAsync(TextGenerationRequest request, TextGenerationResult result, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        LastPrompt = request.Prompt;
        foreach (char c in output)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await Task.Yield();
            yield return c.ToString();
        }

        result.GeneratedTokens = output.Length;
    }

    public void Dispose()
    {
    }
}
