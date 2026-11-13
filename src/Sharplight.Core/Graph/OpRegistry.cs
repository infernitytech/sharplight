using System.Collections.Concurrent;

namespace Sharplight.Graph;

/// <summary>Global, thread-safe registry of operator schemas.</summary>
public static class OpRegistry
{
    private static readonly ConcurrentDictionary<string, OpSchema> s_ops = new(StringComparer.Ordinal);

    static OpRegistry() => StandardOps.RegisterAll();

    public static OpSchema Register(OpSchema schema)
    {
        if (!s_ops.TryAdd(schema.Name, schema) && !ReferenceEquals(s_ops[schema.Name], schema))
        {
            throw new InvalidOperationException($"Op '{schema.Name}' is already registered.");
        }

        return schema;
    }

    public static OpSchema Get(string name) =>
        s_ops.TryGetValue(name, out var s) ? s : throw new KeyNotFoundException($"Unknown op '{name}'.");

    public static bool TryGet(string name, out OpSchema schema) => s_ops.TryGetValue(name, out schema!);

    public static IEnumerable<OpSchema> All => s_ops.Values;
}
