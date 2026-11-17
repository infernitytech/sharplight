using Microsoft.AspNetCore.Http;
using Sharplight.AspNetCore;

namespace Sharplight.Serving.Tests;

/// <summary>Refuses everything of one scope with 429 and counts the others' held slots.</summary>
public sealed class ScopeAdmission(string refusedScope) : IRequestAdmission
{
    private int _held;

    public int Held => Volatile.Read(ref _held);

    public int Admitted { get; private set; }

    public ValueTask<IAsyncDisposable?> AdmitAsync(SharplightRequestContext context, string model, HttpContext http, CancellationToken cancellationToken)
    {
        if (context.Scope == refusedScope)
        {
            throw ApiException.TooManyRequests($"Scope '{context.Scope}' is over its quota.", 7);
        }

        Admitted++;
        Interlocked.Increment(ref _held);
        return ValueTask.FromResult<IAsyncDisposable?>(new Slot(this));
    }

    private sealed class Slot(ScopeAdmission owner) : IAsyncDisposable
    {
        public ValueTask DisposeAsync()
        {
            Interlocked.Decrement(ref owner._held);
            return ValueTask.CompletedTask;
        }
    }
}
