using System.Diagnostics;
using Sharplight.Llm.Embeddings;

namespace Sharplight.Embeddings.Tests;

/// <summary>
/// Throughput of 64 short texts (one batch) per model. Opt-in (SHARPLIGHT_EMBED_BENCH=1; device via SHARPLIGHT_EMBED_BENCH_DEVICE,
/// default vulkan); results go to accuracy-summary.txt. Minimum of several runs after a warm-up, as the GPU may be shared.
/// </summary>
public class ThroughputTests
{
    static readonly string[] Words = ["search", "vector", "model", "query", "document", "fast", "local", "inference", "text", "embedding",
        "rank", "the", "a", "of", "with", "graph", "token", "batch", "shader", "memory"];

    static List<string> Texts()
    {
        var rng = new Random(42);
        // 64 texts of 8-24 words (~10-30 tokens each).
        return [.. Enumerable.Range(0, 64).Select(_ => string.Join(' ', Enumerable.Range(0, rng.Next(8, 25)).Select(_ => Words[rng.Next(Words.Length)])))];
    }

    public static TheoryData<string, string> Models => new()
    {
        { "all-MiniLM-L6-v2", "hf" }, { "bge-small-en-v1.5", "hf" }, { "nomic-embed-text-v1.5", "hf" },
        { "multilingual-e5-small", "hf" }, { "Qwen3-Embedding-0.6B", "hf" },
        { "bge-small-en-v1.5", "gguf:bge-small-en-v1.5/bge-small-en-v1.5-f16.gguf" }, { "nomic-embed-text-v1.5", "gguf:nomic-embed-text-v1.5/nomic-embed-text-v1.5.f16.gguf" },
        { "bge-reranker-base", "rerank" }, { "bge-reranker-v2-m3", "rerank" }, { "Qwen3-Reranker-0.6B", "rerank" },
    };

    [Theory, MemberData(nameof(Models))]
    public void Throughput(string model, string kind)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_EMBED_BENCH") == "1", "Set SHARPLIGHT_EMBED_BENCH=1 to run.");
        string path = kind.StartsWith("gguf:") ? Env.Gguf(kind[5..].Split('/')[0], kind[5..].Split('/')[1]) : Env.Hf(model);
        Env.RequireModel(path);
        string device = Environment.GetEnvironmentVariable("SHARPLIGHT_EMBED_BENCH_DEVICE") ?? "vulkan";
        var dev = Env.Device(device);
        var texts = Texts();
        Func<int> run;
        IDisposable owner;
        if (kind == "rerank")
        {
            var r = Reranker.Load(path, new RerankerLoadOptions { Devices = [dev] });
            owner = r;
            run = () => r.Score("what is a fast local vector search model?", texts).TotalTokens;
        }
        else
        {
            var m = EmbeddingModel.Load(path, new EmbeddingLoadOptions { Devices = [dev] });
            owner = m;
            run = () => m.Embed(texts).TotalTokens;
        }
        using (owner)
        {
            int tokens = run();
            double best = double.MaxValue;
            for (int i = 0; i < 5; i++)
            {
                var sw = Stopwatch.StartNew();
                run();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            Env.Summary[$"throughput {kind.Split('/')[^1]} {model} {device}"] = $"64 texts, {tokens} tokens: {best:F1} ms = {64 / best * 1000:F0} texts/s, {tokens / best * 1000:F0} tokens/s";
        }
    }
}
