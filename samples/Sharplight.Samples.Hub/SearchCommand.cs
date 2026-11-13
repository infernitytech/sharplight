using System.CommandLine;
using System.Globalization;
using Sharplight.Hub;

namespace Sharplight.Samples.Hub;

/// <summary><c>search &lt;query&gt; [--format gguf|safetensors|onnx] [--limit N]</c>: lists Hub repositories, most downloaded first.</summary>
public sealed class SearchCommand(StoreOptionSet store)
{
    private const int MaxVariantsShown = 6;
    private const string FormatGguf = "gguf";
    private const string FormatSafetensors = "safetensors";
    private const string FormatOnnx = "onnx";

    private readonly Argument<string> _query = new("query") { Description = "Text to search for in repository names." };

    private readonly Option<string?> _format = new("--format")
    {
        HelpName = "format",
        Description = "Only repositories with files of this format: gguf, safetensors or onnx.",
    };

    private readonly Option<int> _limit = OptionFactory.Integer("--limit", "Maximum number of results.", HubSearchQuery.DefaultLimit, helpName: "count");

    public Command Build()
    {
        _format.AcceptOnlyFromAmong(FormatGguf, FormatSafetensors, FormatOnnx);
        var command = new Command("search", "Searches Hugging Face for models (most downloaded first) with parameter counts, formats and the variants (GGUF quantizations) that pull accepts as tags.");
        command.Arguments.Add(_query);
        command.Options.Add(_format);
        command.Options.Add(_limit);
        command.SetAction(HubGuard.Guard(RunAsync));
        return command;
    }

    private async Task<int> RunAsync(ParseResult parseResult, CancellationToken cancellationToken)
    {
        var query = new HubSearchQuery
        {
            Text = parseResult.GetValue(_query)!,
            Format = ParseFormat(parseResult.GetValue(_format)),
            Limit = parseResult.GetValue(_limit),
        };
        using var hub = new HubContext(store.ReadRoot(parseResult));
        var results = await hub.Hub.SearchAsync(query, cancellationToken);
        if (results.Count == 0)
        {
            Console.WriteLine($"No models match '{query.Text}'.");
            return ExitCodes.Success;
        }

        int width = Math.Max("REPOSITORY".Length, results.Max(result => result.Id.Length + (result.Gated ? 2 : 0)));
        Console.WriteLine($"{"REPOSITORY".PadRight(width)}  {"DOWNLOADS",10}  {"LIKES",6}  {"PARAMS",7}  {"FORMATS",-18}  VARIANTS");
        foreach (var result in results)
        {
            var id = result.Gated ? result.Id + " *" : result.Id;
            var formats = string.Join(",", result.Formats.Select(format => format.ToString().ToLowerInvariant()));
            Console.WriteLine($"{id.PadRight(width)}  {result.Downloads,10:N0}  {result.Likes,6:N0}  {FormatParameters(result.Parameters),7}  {formats,-18}  {FormatVariants(result.Variants)}");
        }

        if (results.Any(result => result.Gated))
        {
            Console.WriteLine($"* gated: accept the terms on the repository page and set {HubClientOptions.TokenVariable}.");
        }

        return ExitCodes.Success;
    }

    private static ModelFormat? ParseFormat(string? text) => text switch
    {
        FormatGguf => ModelFormat.Gguf,
        FormatSafetensors => ModelFormat.Safetensors,
        FormatOnnx => ModelFormat.Onnx,
        _ => null,
    };

    private static string FormatParameters(long? parameters)
    {
        return parameters switch
        {
            null => "-",
            >= 1_000_000_000 => string.Create(CultureInfo.InvariantCulture, $"{parameters / 1e9:F1}B"),
            _ => string.Create(CultureInfo.InvariantCulture, $"{parameters / 1e6:F0}M"),
        };
    }

    private static string FormatVariants(IReadOnlyList<string> variants)
    {
        var shown = string.Join(" ", variants.Take(MaxVariantsShown));
        return variants.Count > MaxVariantsShown ? $"{shown} (+{variants.Count - MaxVariantsShown})" : shown;
    }
}
