using System.Collections.Concurrent;

namespace Sharplight.Graph;

/// <summary>Global, thread-safe registry of operator schemas.</summary>
public static class OpRegistry
{
    private static readonly ConcurrentDictionary<string, OpSchema> s_ops = new(StringComparer.Ordinal);

    static OpRegistry() => StandardOps.RegisterAll();

    /// <summary>
    /// Adds an op to the registry. Registering the identical schema instance again is a no-op.
    /// </summary>
    /// <param name="schema">The op to register.</param>
    /// <returns><paramref name="schema"/>, so registrations can initialize static fields.</returns>
    /// <exception cref="InvalidOperationException">A different op with the same name is already registered.</exception>
    public static OpSchema Register(OpSchema schema)
    {
        if (!s_ops.TryAdd(schema.Name, schema) && !ReferenceEquals(s_ops[schema.Name], schema))
        {
            throw new InvalidOperationException($"Op '{schema.Name}' is already registered.");
        }

        return schema;
    }

    /// <summary>
    /// Looks up an op by name.
    /// </summary>
    /// <param name="name">The op name.</param>
    /// <exception cref="KeyNotFoundException">No op with that name is registered.</exception>
    public static OpSchema Get(string name) =>
        s_ops.TryGetValue(name, out var s) ? s : throw new KeyNotFoundException($"Unknown op '{name}'.");

    /// <summary>
    /// Looks up an op by name without throwing.
    /// </summary>
    /// <param name="name">The op name.</param>
    /// <param name="schema">The op, or <see langword="null"/> when not found.</param>
    /// <returns>Whether the op was found.</returns>
    public static bool TryGet(string name, out OpSchema schema) => s_ops.TryGetValue(name, out schema!);

    /// <summary>All registered ops, in no particular order.</summary>
    public static IEnumerable<OpSchema> All => s_ops.Values;
}
