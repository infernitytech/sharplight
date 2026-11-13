using System.Collections;

namespace Sharplight.Templating;

/// <summary>
/// Value of a name or attribute that does not exist. Renders as empty text, is falsy and iterates as empty
/// (unless <see cref="Strict"/>, in which case any use except <c>is defined</c>-style tests throws).
/// </summary>
public sealed class Undefined
{
    internal Undefined(string? name, bool strict)
    {
        Name = name;
        Strict = strict;
    }

    public string? Name { get; }
    public bool Strict { get; }
    public static Undefined Default { get; } = new(null, false);

    internal TemplateRuntimeException Fail() => new($"'{Name ?? "value"}' is undefined");

    public override string ToString() => "";
}

/// <summary>Insertion-ordered dictionary (Python dict semantics). Keys are strings, longs or bools.</summary>
public sealed class JinjaDict : IEnumerable<KeyValuePair<object, object?>>
{
    private readonly Dictionary<object, object?> _map = new();
    private readonly List<object> _keys = new();

    public int Count => _keys.Count;
    public IReadOnlyList<object> Keys => _keys;

    public object? this[object key]
    {
        get => _map[key];
        set
        {
            key = NormalizeKey(key);
            if (!_map.ContainsKey(key)) _keys.Add(key);
            _map[key] = value;
        }
    }

    internal static object NormalizeKey(object key) => key switch
    {
        int i => (long)i,
        short s => (long)s,
        byte b => (long)b,
        _ => key,
    };

    public bool TryGetValue(object key, out object? value) => _map.TryGetValue(NormalizeKey(key), out value);
    public bool ContainsKey(object key) => _map.ContainsKey(NormalizeKey(key));

    public bool Remove(object key)
    {
        key = NormalizeKey(key);
        if (!_map.Remove(key)) return false;
        _keys.Remove(key);
        return true;
    }

    public void Clear()
    {
        _map.Clear();
        _keys.Clear();
    }

    public JinjaDict Clone()
    {
        var d = new JinjaDict();
        foreach (var k in _keys) d[k] = _map[k];
        return d;
    }

    public IEnumerator<KeyValuePair<object, object?>> GetEnumerator()
    {
        foreach (var k in _keys.ToArray()) yield return new KeyValuePair<object, object?>(k, _map[k]);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>Result of <c>namespace(...)</c>: a mutable bag of attributes.</summary>
public sealed class JinjaNamespace
{
    public Dictionary<string, object?> Attributes { get; } = new();
}

/// <summary>Positional and keyword arguments of a call.</summary>
public sealed class CallArgs
{
    public static readonly CallArgs Empty = new(new List<object?>(), null);

    public CallArgs(List<object?> positional, Dictionary<string, object?>? named)
    {
        Positional = positional;
        Named = named;
    }

    public List<object?> Positional { get; }
    public Dictionary<string, object?>? Named { get; }

    public int Count => Positional.Count + (Named?.Count ?? 0);

    /// <summary>True if the argument was supplied positionally at <paramref name="index"/> or by keyword.</summary>
    public bool Has(int index, string name) => index < Positional.Count || (Named?.ContainsKey(name) ?? false);

    public object? Get(int index, string name, object? defaultValue = null)
    {
        if (index >= 0 && index < Positional.Count) return Positional[index];
        if (Named != null && Named.TryGetValue(name, out var v)) return v;
        return defaultValue;
    }

    public object? GetNamed(string name, object? defaultValue = null) =>
        Named != null && Named.TryGetValue(name, out var v) ? v : defaultValue;
}

/// <summary>Something that can be called from a template: macros, builtin functions, bound methods.</summary>
public abstract class JinjaCallable
{
    public abstract string Name { get; }
    public abstract object? Invoke(CallArgs args);
}

/// <summary>A host function exposed to templates.</summary>
public sealed class NativeFunction : JinjaCallable
{
    private readonly Func<CallArgs, object?> _fn;

    public NativeFunction(string name, Func<CallArgs, object?> fn)
    {
        Name = name;
        _fn = fn;
    }

    public override string Name { get; }
    public override object? Invoke(CallArgs args) => _fn(args);
}

internal sealed class BoundMethod : JinjaCallable
{
    private readonly object _target;
    private readonly string _name;

    public BoundMethod(object target, string name)
    {
        _target = target;
        _name = name;
    }

    public override string Name => _name;
    public override object? Invoke(CallArgs args) => Methods.Call(_target, _name, args);
}

/// <summary>Result of <c>|safe</c> / <c>|escape</c>: a string that is not escaped again, but escapes other operands when used with <c>+</c>.</summary>
public sealed class Markup
{
    public Markup(string value) => Value = value;
    public string Value { get; }
    public override string ToString() => Value;

    internal static string Escape(string s)
    {
        var sb = new System.Text.StringBuilder(s.Length);
        foreach (char c in s)
        {
            switch (c)
            {
                case '&': sb.Append("&amp;"); break;
                case '<': sb.Append("&lt;"); break;
                case '>': sb.Append("&gt;"); break;
                case '\'': sb.Append("&#39;"); break;
                case '"': sb.Append("&#34;"); break;
                default: sb.Append(c); break;
            }
        }
        return sb.ToString();
    }
}

/// <summary>Python tuple: a list that prints with parentheses.</summary>
public sealed class JinjaTuple : List<object?>
{
    public JinjaTuple() { }
    public JinjaTuple(IEnumerable<object?> items) : base(items) { }
}

internal sealed class SliceValue
{
    public object? Start, Stop, Step;
}

internal sealed class LoopContext
{
    public List<object?> Items = new();
    public int Index0 = -1;
    public int Depth0 => 0;
    private object? _changedLast = Unset;
    private static readonly object Unset = new();

    public int Length => Items.Count;

    public object? Get(string name, bool strict)
    {
        switch (name)
        {
            case "index": return (long)(Index0 + 1);
            case "index0": return (long)Index0;
            case "revindex": return (long)(Length - Index0);
            case "revindex0": return (long)(Length - Index0 - 1);
            case "first": return Index0 == 0;
            case "last": return Index0 == Length - 1;
            case "length": return (long)Length;
            case "depth": return (long)(Depth0 + 1);
            case "depth0": return (long)Depth0;
            case "previtem": return Index0 > 0 ? Items[Index0 - 1] : new Undefined("loop.previtem", strict);
            case "nextitem": return Index0 < Length - 1 ? Items[Index0 + 1] : new Undefined("loop.nextitem", strict);
            case "cycle":
            case "changed": return new BoundMethod(this, name);
            default: return new Undefined("loop." + name, strict);
        }
    }

    public object? Call(string name, CallArgs args)
    {
        if (name == "cycle")
        {
            if (args.Positional.Count == 0) throw new TemplateRuntimeException("loop.cycle() requires at least one argument");
            return args.Positional[Index0 % args.Positional.Count];
        }
        // changed
        var val = (object?)new List<object?>(args.Positional);
        bool changed = ReferenceEquals(_changedLast, Unset) || !Ops.Eq(_changedLast, val);
        _changedLast = val;
        return changed;
    }
}
