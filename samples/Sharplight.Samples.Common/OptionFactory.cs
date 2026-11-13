using System.CommandLine;
using System.CommandLine.Parsing;
using System.Globalization;

namespace Sharplight.Samples;

/// <summary>Builds typed <c>System.CommandLine</c> options with the validation the samples share.</summary>
public static class OptionFactory
{
    /// <summary>An integer option that must be at least <paramref name="minimum"/>.</summary>
    public static Option<int> Integer(string name, string description, int defaultValue, int minimum = 1, string? helpName = null, params string[] aliases)
    {
        var option = new Option<int>(name, aliases)
        {
            Description = description,
            HelpName = helpName,
            DefaultValueFactory = _ => defaultValue,
        };
        option.Validators.Add(result =>
        {
            int value = result.GetValueOrDefault<int>();
            if (value < minimum)
            {
                result.AddError($"{name} must be at least {minimum} (got {value}).");
            }
        });
        return option;
    }

    /// <summary>An optional integer option (null when absent) that must be at least <paramref name="minimum"/>.</summary>
    public static Option<int?> OptionalInteger(string name, string description, int minimum = 1, string? helpName = null)
    {
        var option = new Option<int?>(name) { Description = description, HelpName = helpName };
        option.Validators.Add(result =>
        {
            int? value = result.GetValueOrDefault<int?>();
            if (value < minimum)
            {
                result.AddError($"{name} must be at least {minimum} (got {value}).");
            }
        });
        return option;
    }

    /// <summary>A floating-point option parsed with the invariant culture and a lower bound.</summary>
    public static Option<float> Single(string name, string description, float defaultValue, float minimum = 0f, string? helpName = null)
    {
        var option = new Option<float>(name)
        {
            Description = description,
            HelpName = helpName,
            DefaultValueFactory = _ => defaultValue,
            CustomParser = result => ParseSingle(result, name, minimum),
        };
        return option;
    }

    /// <summary>An option whose value is a comma-separated list; it may also be repeated, and all items are merged.</summary>
    public static Option<string[]> CommaSeparatedList(string name, string description, string helpName = "list")
    {
        return new Option<string[]>(name)
        {
            Description = description,
            HelpName = helpName,
            DefaultValueFactory = _ => [],
            CustomParser = result => Split(result.Tokens.Select(token => token.Value)),
        };
    }

    /// <summary>A memory-size option (<c>512M</c>, <c>1.5G</c>, ...) converted to bytes; null when absent unless <paramref name="defaultValue"/> is given.</summary>
    public static Option<long?> Size(string name, string description, string? defaultValue = null, string helpName = "size")
    {
        var option = new Option<long?>(name)
        {
            Description = description,
            HelpName = helpName,
            CustomParser = result => ParseSize(result, name),
        };
        if (defaultValue is not null)
        {
            option.DefaultValueFactory = _ => ByteSize.Parse(defaultValue);
        }

        return option;
    }

    /// <summary>Splits comma-separated items, trimming whitespace and dropping empty entries.</summary>
    public static string[] Split(IEnumerable<string> values)
    {
        return [.. values.SelectMany(value => value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))];
    }

    private static float ParseSingle(ArgumentResult result, string name, float minimum)
    {
        var text = result.Tokens.Single().Value;
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out float value) || value < minimum)
        {
            result.AddError($"{name} expects a number >= {minimum.ToString(CultureInfo.InvariantCulture)}, got '{text}'.");
        }

        return value;
    }

    private static long? ParseSize(ArgumentResult result, string name)
    {
        var text = result.Tokens.Single().Value;
        try
        {
            return ByteSize.Parse(text);
        }
        catch (FormatException error)
        {
            result.AddError($"{name}: {error.Message}");
            return null;
        }
    }
}
