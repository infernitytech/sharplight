using System.Globalization;

/// <summary>The command line of VideoEnhancementBench.</summary>
internal sealed record Options(string Model, string Input, string? InputShape, int Width, int Height, string Provider, int? Batch, int Runs, int Warmup,
    bool ReducedPrecision, string? OutputU8, string? OutputVideo, string? Json, string? Noise, string? Profile, string? Memory)
{
    public static Options Parse(string[] args)
    {
        string? Get(string name) => Array.IndexOf(args, name) is int i and >= 0 && i + 1 < args.Length ? args[i + 1] : null;
        return new Options(Get("--model")!, Get("--input") ?? "", Get("--input-shape"), int.Parse(Get("--width") ?? "0", CultureInfo.InvariantCulture),
            int.Parse(Get("--height") ?? "0", CultureInfo.InvariantCulture), Get("--provider") ?? "cuda",
            Get("--batch") is { } batch ? int.Parse(batch, CultureInfo.InvariantCulture) : null,
            int.Parse(Get("--runs") ?? "5", CultureInfo.InvariantCulture), int.Parse(Get("--warmup") ?? "1", CultureInfo.InvariantCulture),
            args.Contains("--reduced-precision"), Get("--output-u8"), Get("--output-video"), Get("--json"), Get("--noise"), Get("--profile"), Get("--memory"));
    }
}
