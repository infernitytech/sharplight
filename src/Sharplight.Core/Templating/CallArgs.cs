namespace Sharplight.Templating;

/// <summary>Positional and keyword arguments of a call.</summary>
public sealed class CallArgs
{
    /// <summary>A call without arguments.</summary>
    public static readonly CallArgs Empty = new(new List<object?>(), null);

    /// <summary>Creates the argument set; <paramref name="named"/> may be null when there are no keyword arguments.</summary>
    public CallArgs(List<object?> positional, Dictionary<string, object?>? named)
    {
        Positional = positional;
        Named = named;
    }

    /// <summary>Positional arguments in call order, in the template value model.</summary>
    public List<object?> Positional { get; }
    /// <summary>Keyword arguments, or null when none were given.</summary>
    public Dictionary<string, object?>? Named { get; }

    /// <summary>Total number of positional and keyword arguments.</summary>
    public int Count => Positional.Count + (Named?.Count ?? 0);

    /// <summary>True if the argument was supplied positionally at <paramref name="index"/> or by keyword.</summary>
    public bool Has(int index, string name) => index < Positional.Count || (Named?.ContainsKey(name) ?? false);

    /// <summary>Argument at <paramref name="index"/> or keyword <paramref name="name"/> (positional wins); <paramref name="defaultValue"/> when absent.</summary>
    public object? Get(int index, string name, object? defaultValue = null)
    {
        if (index >= 0 && index < Positional.Count)
        {
            return Positional[index];
        }

        if (Named != null && Named.TryGetValue(name, out var v))
        {
            return v;
        }

        return defaultValue;
    }

    /// <summary>Keyword argument <paramref name="name"/>, or <paramref name="defaultValue"/> when absent.</summary>
    public object? GetNamed(string name, object? defaultValue = null) =>
        Named != null && Named.TryGetValue(name, out var v) ? v : defaultValue;
}
