using System.Globalization;
using Sharplight.Hosting;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Samples;

namespace Sharplight.Cli;

/// <summary>Parses the value of <c>--model</c>: <c>NAME=PATH[,key=value...]</c> into a hosted model.</summary>
internal static class ModelSpecParser
{
    private const char NameSeparator = '=';
    private const char SettingSeparator = ',';

    /// <exception cref="SampleException">The text is malformed or names an unknown setting.</exception>
    public static HostedModelOptions Parse(string text)
    {
        int separator = text.IndexOf(NameSeparator);
        if (separator <= 0)
        {
            throw new SampleException($"--model expects NAME=PATH[,key=value...], got '{text}'.");
        }

        var parts = text[(separator + 1)..].Split(SettingSeparator);
        var model = new HostedModelOptions { Name = text[..separator], Path = parts[0] };
        foreach (var setting in parts.Skip(1))
        {
            Apply(model, setting, text);
        }

        return model;
    }

    private static void Apply(HostedModelOptions model, string setting, string fullText)
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
                    model.ContextLength = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "slots":
                    model.Slots = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "batch":
                    model.BatchTokens = int.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "device":
                    model.Device = value.Replace('+', ',');
                    break;
                case "mmproj":
                    model.Projector = value;
                    break;
                case "template":
                    model.ChatTemplate = value;
                    break;
                case "lazy":
                    model.Lazy = bool.Parse(value);
                    break;
                case "pin":
                    model.Pinned = bool.Parse(value);
                    break;
                case "offload":
                    model.CpuOffload = bool.Parse(value);
                    break;
                case "idle":
                    model.IdleTimeoutSeconds = double.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "kind":
                    model.Kind = Enum.Parse<ModelKind>(value, ignoreCase: true);
                    break;
                case "task":
                    model.Kind = ModelKind.Task;
                    model.Task = value;
                    break;
                default:
                    throw new SampleException($"Unknown model setting '{keyValue[0]}' in --model {fullText}.");
            }
        }
        catch (Exception error) when (error is FormatException or ArgumentException)
        {
            throw new SampleException($"Invalid value '{value}' for model setting '{key}' in --model {fullText}.");
        }
    }
}
