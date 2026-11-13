using System.Runtime.InteropServices;

namespace Sharplight.Cli;

/// <summary>Processes a prompt, prints its token ids and the most likely next tokens, and stores the raw last-token logits.</summary>
internal static class LogitsDumper
{
    private const int TopTokenCount = 5;

    public static void Dump(LoadedModel loaded, string prompt, string outputPath, TextWriter output)
    {
        var tokenizer = loaded.Model.Tokenizer;
        var ids = tokenizer.Encode(prompt, addSpecial: true, parseSpecial: true);
        float[] logits = loaded.Session.Process(ids);
        File.WriteAllBytes(outputPath, MemoryMarshal.AsBytes<float>(logits).ToArray());

        output.WriteLine("ids: " + string.Join(",", ids));
        var best = Enumerable.Range(0, logits.Length)
            .OrderByDescending(id => logits[id])
            .Take(TopTokenCount);
        foreach (int id in best)
        {
            output.WriteLine($"  {id} {logits[id]:F4} {tokenizer.Decode([id])}");
        }
    }
}
