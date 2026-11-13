using System.Globalization;
using Sharplight.Samples;

namespace Sharplight.Server;

/// <summary>Parses the value of <c>--model</c>: <c>NAME=PATH[,key=value...]</c>.</summary>
internal static class ModelSpecParser
{
    private const char NameSeparator = '=';
    private const char SettingSeparator = ',';

    /// <exception cref="SampleException">The text is malformed or names an unknown setting.</exception>
    public static ModelSpec Parse(string text)
    {
        int separator = text.IndexOf(NameSeparator);
        if (separator <= 0)
        {
            throw new SampleException($"--model expects NAME=PATH[,key=value...], got '{text}'.");
        }

        var parts = text[(separator + 1)..].Split(SettingSeparator);
        var spec = new ModelSpec { Name = text[..separator], Path = parts[0] };
        foreach (var setting in parts.Skip(1))
        {
            Apply(spec, setting, text);
        }

        return spec;
    }

    private static void Apply(ModelSpec spec, string setting, string fullText)
    {
        var keyValue = setting.Split(NameSeparator, 2);
        var key = keyValue[0].ToLowerInvariant();

        // A bare key such as "pin" means "pin=true".
        var value = keyValue.Length > 1 ? keyValue[1] : "true";
        try
        {
            switch (key)
            {
                case "ctx":
                    spec.Ctx = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "slots":
                    spec.Slots = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "batch":
                    spec.Batch = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "device":
                    spec.Device = value;
                    break;
                case "mmproj":
                    spec.Mmproj = value;
                    break;
                case "template":
                    spec.Template = value;
                    break;
                case "lazy":
                    spec.Lazy = bool.Parse(value);
                    break;
                case "pin":
                    spec.Pin = bool.Parse(value);
                    break;
                case "offload":
                    spec.Offload = bool.Parse(value);
                    break;
                case "idle":
                    spec.Idle = double.Parse(value, CultureInfo.InvariantCulture);
                    break;
                default:
                    throw new SampleException($"Unknown model setting '{keyValue[0]}' in --model {fullText}.");
            }
        }
        catch (FormatException)
        {
            throw new SampleException($"Invalid value '{value}' for model setting '{key}' in --model {fullText}.");
        }
    }
}
