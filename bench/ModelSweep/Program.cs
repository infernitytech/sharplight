// Real-checkpoint sweep: header-only compatibility check vs the real import, without downloading weights.
//
//   ModelSweep <scratch dir> <repo>[:<variant filter>] ...      Hub repositories (GGUF variants or a safetensors checkpoint)
//   ModelSweep <scratch dir> --local <file.gguf | hf dir> ...   local models (same steps, no download)
//
// For every GGUF variant (all shards) the header (metadata + tensor table) is read with HTTP range requests; for a safetensors
// checkpoint the small files and every shard's JSON header. They are written into sparse files of the real size under the
// scratch dir, so the real import (LlmModel.Load: importer mapping, adapter graph build, tokenizer) runs on them; tensor data
// reads as zeros and is never needed for that. Then the chat template is rendered and a test string tokenized.
// One tab-separated line per variant goes to stdout:
//   repo  variant  arch  types  check  import  tokens  template
// "check" is the header-only verdict as the compatibility check computes it (architecture mapped, every tensor type known and with
// kernels); "import" is what the real import does.
using System.Diagnostics;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Hub;
using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Templating;
using ModelSweep;

GgufImporter.Register();
HuggingFaceImporter.Register();

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: ModelSweep <scratch dir> <repo>[:<filter>] ... | --local <path> ...");
    return 2;
}

string scratch = Path.GetFullPath(args[0]);
Directory.CreateDirectory(scratch);
bool local = args[1] == "--local";
var targets = args.Skip(local ? 2 : 1).ToList();
string? token = Environment.GetEnvironmentVariable("HF_TOKEN");
string tokenFile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "huggingface", "token");
if (token is null && File.Exists(tokenFile))
{
    token = File.ReadAllText(tokenFile).Trim();
}

var options = new HubClientOptions { Token = token, RequestTimeout = TimeSpan.FromMinutes(3) };
using var http = HubClientOptions.CreateHttpClient();
var hub = new HubClient(http, options);
bool keep = Environment.GetEnvironmentVariable("SWEEP_KEEP") == "1";

foreach (var target in targets)
{
    if (local)
    {
        Sweep.Report("local", Path.GetFileName(target.TrimEnd('/')), target, null);
        continue;
    }

    string repoId = target;
    string? filter = null;
    int colon = target.IndexOf(':', StringComparison.Ordinal);
    if (colon > 0)
    {
        repoId = target[..colon];
        filter = target[(colon + 1)..];
    }

    HubRepository repo;
    try
    {
        repo = await hub.GetRepositoryAsync(repoId);
    }
    catch (Exception error)
    {
        Console.WriteLine($"{repoId}\t-\t-\t-\t-\tHUB: {Sweep.OneLine(error.Message)}\t-\t-");
        continue;
    }

    string dir = Path.Combine(scratch, repoId.Replace('/', '_'));
    var ggufs = repo.Files.Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && !f.Name.Contains("mmproj", StringComparison.OrdinalIgnoreCase))
        .ToList();
    if (ggufs.Count > 0)
    {
        foreach (var variant in ggufs.GroupBy(f => Sweep.VariantKey(f.Path)).OrderBy(g => g.Key, StringComparer.Ordinal))
        {
            if (filter is not null && !variant.Key.Contains(filter, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var shards = variant.OrderBy(f => f.Path, StringComparer.Ordinal).ToList();
            string variantDir = Path.Combine(dir, variant.Key.Replace('/', '_'));
            try
            {
                var header = await Sweep.FetchGgufHeadersAsync(hub, repo, shards, variantDir);
                Sweep.Report(repoId, variant.Key, Path.Combine(variantDir, shards[0].Name), header);
            }
            catch (Exception error)
            {
                Console.WriteLine($"{repoId}\t{variant.Key}\t-\t-\t-\tFETCH: {Sweep.OneLine(error.Message)}\t-\t-");
            }
            finally
            {
                if (!keep && Directory.Exists(variantDir))
                {
                    Directory.Delete(variantDir, recursive: true);
                }
            }
        }

        continue;
    }

    if (repo.Files.Any(f => f.Path.EndsWith(".safetensors", StringComparison.Ordinal)))
    {
        try
        {
            await Sweep.FetchCheckpointAsync(hub, repo, dir);
            Sweep.Report(repoId, "safetensors", dir, null);
        }
        catch (Exception error)
        {
            Console.WriteLine($"{repoId}\tsafetensors\t-\t-\t-\tFETCH: {Sweep.OneLine(error.Message)}\t-\t-");
        }
        finally
        {
            if (!keep && Directory.Exists(dir))
            {
                Directory.Delete(dir, recursive: true);
            }
        }

        continue;
    }

    Console.WriteLine($"{repoId}\t-\t-\t-\t-\tno GGUF or safetensors files\t-\t-");
}

return 0;

namespace ModelSweep
{
    /// <summary>The sweep steps.</summary>
    internal static class Sweep
    {
        private const string TestText = "The capital of France is Paris. 1234567 +-*/ ünïcödé 日本語 \n\n  spaces\t<tab>";

        /// <summary>Header-derived facts of a GGUF variant.</summary>
        internal sealed record GgufHeader(IReadOnlyDictionary<string, object> Metadata, Dictionary<int, int> Types, List<GgufTensorEntry> Tensors);

        /// <summary>The variant of a GGUF path: the path without the shard suffix and the extension.</summary>
        public static string VariantKey(string path)
        {
            string name = path[..^".gguf".Length];
            var match = System.Text.RegularExpressions.Regex.Match(name, @"-\d{5}-of-\d{5}$");
            return match.Success ? name[..match.Index] : name;
        }

        public static string OneLine(string text)
        {
            string line = text.Replace('\n', ' ').Replace('\t', ' ').Replace('\r', ' ');
            return line.Length > 400 ? line[..400] + "..." : line;
        }

        /// <summary>Reads each shard's header (growing the range until the tensor table is complete) into a sparse file of the real size.</summary>
        public static async Task<GgufHeader> FetchGgufHeadersAsync(HubClient hub, HubRepository repo, IReadOnlyList<HubFile> shards, string dir)
        {
            Directory.CreateDirectory(dir);
            IReadOnlyDictionary<string, object>? metadata = null;
            var types = new Dictionary<int, int>();
            var tensors = new List<GgufTensorEntry>();
            foreach (var shard in shards)
            {
                int count = 1 << 21;
                byte[] prefix;
                IReadOnlyDictionary<int, int>? shardTypes;
                while (true)
                {
                    prefix = await hub.GetFilePrefixAsync(repo, shard.Path, (int)Math.Min(count, shard.Size));
                    shardTypes = GgufHeaderProbe.ReadTensorTypes(prefix);
                    if (shardTypes is not null || prefix.Length >= shard.Size || count >= 1 << 30)
                    {
                        break;
                    }

                    count *= 4;
                }

                if (shardTypes is null)
                {
                    throw new InvalidDataException($"{shard.Path}: tensor table not complete within {prefix.Length} bytes");
                }

                metadata ??= GgufHeaderProbe.ReadMetadata(prefix);
                tensors.AddRange(GgufHeaderProbe.ReadTensorTable(prefix) ?? []);
                foreach (var (type, n) in shardTypes)
                {
                    types[type] = types.GetValueOrDefault(type) + n;
                }

                using var file = new FileStream(Path.Combine(dir, shard.Name), FileMode.Create, FileAccess.Write);
                file.SetLength(shard.Size);
                file.Write(prefix);
            }

            return new GgufHeader(metadata!, types, tensors);
        }

        /// <summary>Small files fully, every safetensors shard as its JSON header in a sparse file of the real size.</summary>
        public static async Task FetchCheckpointAsync(HubClient hub, HubRepository repo, string dir)
        {
            Directory.CreateDirectory(dir);
            foreach (var file in repo.Files)
            {
                if (file.Path.Contains('/', StringComparison.Ordinal) && !file.Path.EndsWith(".safetensors", StringComparison.Ordinal))
                {
                    continue;
                }

                string target = Path.Combine(dir, file.Path);
                Directory.CreateDirectory(Path.GetDirectoryName(target)!);
                if (file.Path.EndsWith(".safetensors", StringComparison.Ordinal))
                {
                    var head = await hub.GetFilePrefixAsync(repo, file.Path, 8);
                    long length = BitConverter.ToInt64(head);
                    var header = await hub.GetFilePrefixAsync(repo, file.Path, checked((int)(8 + length)));
                    using var stream = new FileStream(target, FileMode.Create, FileAccess.Write);
                    stream.SetLength(file.Size);
                    stream.Write(header);
                }
                else if (file.Size < 64 << 20 && IsSmallModelFile(file.Name))
                {
                    var bytes = await hub.GetFilePrefixAsync(repo, file.Path, (int)Math.Max(1, file.Size));
                    await File.WriteAllBytesAsync(target, bytes);
                }
            }
        }

        private static bool IsSmallModelFile(string name)
            => name.EndsWith(".json", StringComparison.Ordinal) || name.EndsWith(".jinja", StringComparison.Ordinal) || name.EndsWith(".model", StringComparison.Ordinal)
               || name.EndsWith(".txt", StringComparison.Ordinal) || name.EndsWith(".tiktoken", StringComparison.Ordinal) || name.EndsWith(".py", StringComparison.Ordinal)
               || name == "tokenizer.model";

        /// <summary>The header-only verdict of a GGUF: architecture mapped, every tensor type known and with CPU kernels.</summary>
        private static string Check(GgufHeader header)
        {
            string? arch = header.Metadata.GetValueOrDefault("general.architecture") as string;
            if (arch is null)
            {
                return "unknown (no architecture)";
            }

            if (GgufSupport.IsProjector(arch))
            {
                return "projector";
            }

            var level = GgufSupport.GetArchitectureSupport(arch);
            if (level == SupportLevel.Unsupported)
            {
                return "unsupported architecture";
            }

            foreach (var id in header.Types.Keys.Order())
            {
                if (GgufSupport.GetTensorType(id) is not { } type)
                {
                    return $"unsupported type id {id}";
                }

                if (!CpuExecutionProvider.SupportsWeightType(type))
                {
                    return $"unsupported type {type.Name}";
                }
            }

            if (GgufSupport.IsDraftModel(arch))
            {
                return "draft model";
            }

            // The importer mapping run on the header (hyper-parameters, features, every tensor name).
            var validation = GgufSupport.Validate(header.Metadata, header.Tensors);
            if (validation.IsDraftModel)
            {
                return "draft model";
            }

            if (validation.Problems.Count > 0)
            {
                return "mapping: " + OneLine(string.Join("; ", validation.Problems));
            }

            return (level == SupportLevel.Partial ? "partial" : "supported") + (validation.IsComplete ? "" : " (incomplete)");
        }

        private static string CheckConfig(string dir)
        {
            string config = Path.Combine(dir, "config.json");
            if (!File.Exists(config))
            {
                return "no config.json";
            }

            string json = File.ReadAllText(config);
            string? quant = Path.Combine(dir, "hf_quant_config.json") is var q && File.Exists(q) ? File.ReadAllText(q) : null;
            if (HfSupport.GetModelType(json) is not { } modelType)
            {
                return "unknown (no model_type)";
            }

            var level = HfSupport.GetArchitectureSupport(modelType);
            if (level == SupportLevel.Unsupported)
            {
                return $"unsupported model_type {modelType}";
            }

            var quantization = HfSupport.GetQuantizationSupport(json, quant);
            if (quantization.Level == SupportLevel.Unsupported)
            {
                return $"unsupported quantization {quantization.Method}";
            }

            if (quantization.Method is not null && quantization.Level is null)
            {
                return $"unknown quantization {quantization.Method}";
            }

            if (HfSupport.IsDraftModel(modelType))
            {
                return "draft model";
            }

            if (HfSupport.Validate(json) is { Count: > 0 } problems)
            {
                return "mapping: " + OneLine(string.Join("; ", problems));
            }

            return (level == SupportLevel.Partial ? "partial" : "supported") + (quantization.Method is { } m ? $" ({m})" : "");
        }

        /// <summary>Runs the real import on the (sparse) model and prints the result line.</summary>
        public static void Report(string repo, string variant, string path, GgufHeader? header)
        {
            string arch;
            string types;
            string check;
            if (header is not null)
            {
                arch = header.Metadata.GetValueOrDefault("general.architecture") as string ?? "?";
                types = string.Join(",", header.Types.OrderByDescending(p => p.Value).Select(p => $"{TypeName(p.Key)}:{p.Value}"));
                check = Check(header);
            }
            else if (Directory.Exists(path))
            {
                string config = Path.Combine(path, "config.json");
                arch = File.Exists(config) ? HfSupport.GetModelType(File.ReadAllText(config)) ?? "?" : "?";
                types = "-";
                check = CheckConfig(path);
            }
            else
            {
                using var stream = File.OpenRead(path);
                var prefix = new byte[(int)Math.Min(stream.Length, 1L << 28)];
                stream.ReadExactly(prefix);
                var meta = GgufHeaderProbe.ReadMetadata(prefix);
                var t = GgufHeaderProbe.ReadTensorTypes(prefix) ?? new Dictionary<int, int>();
                header = new GgufHeader(meta, new Dictionary<int, int>(t), [.. GgufHeaderProbe.ReadTensorTable(prefix) ?? []]);
                arch = meta.GetValueOrDefault("general.architecture") as string ?? "?";
                types = string.Join(",", header.Types.OrderByDescending(p => p.Value).Select(p => $"{TypeName(p.Key)}:{p.Value}"));
                check = Check(header);
            }

            string import;
            string tokens = "-";
            string template = "-";
            var sw = Stopwatch.StartNew();
            try
            {
                using var model = LlmModel.Load(path, new LlmLoadOptions { Devices = [], ContextLength = 4096 });
                var kernels = model.Definition.Weights.Values.Select(w => w.Type).Distinct().Where(type => !CpuExecutionProvider.SupportsWeightType(type)).ToList();
                import = kernels.Count == 0 ? $"ok {model.Definition.Architecture}" : $"NO KERNEL {string.Join(",", kernels.Select(k => k.Name))}";
                try
                {
                    var ids = model.Tokenizer.Encode(TestText, addSpecial: false);
                    string back = model.Tokenizer.Decode(ids);
                    tokens = $"{ids.Length}{(back == TestText ? "" : " ROUNDTRIP-DIFF")}";
                    if (header is not null && LlamaTokenize(path) is { } reference)
                    {
                        tokens += reference.SequenceEqual(ids) ? " =llama.cpp" : $" LLAMA-DIFF ours [{string.Join(' ', ids)}] llama.cpp [{string.Join(' ', reference)}]";
                    }
                    else if (header is null && HfTokenize(path) is { } hf)
                    {
                        tokens += hf.StartsWith("ERROR", StringComparison.Ordinal) ? $" (transformers: {hf})"
                            : hf == "[" + string.Join(", ", ids) + "]" ? " =transformers" : $" HF-DIFF ours [{string.Join(", ", ids)}] transformers {hf}";
                    }
                }
                catch (Exception error)
                {
                    tokens = "TOKENIZE: " + OneLine(error.Message);
                }

                template = RenderTemplate(model);
            }
            catch (Exception error)
            {
                import = $"FAIL {error.GetType().Name}: {OneLine(error.Message.Replace(path, "<file>", StringComparison.Ordinal))}";
            }

            Console.WriteLine($"{repo}\t{variant}\t{arch}\t{types}\t{check}\t{import}\t{tokens}\t{template}\t{sw.ElapsedMilliseconds}ms");
        }

        private static string RenderTemplate(LlmModel model)
        {
            if (model.ChatTemplate is not { } source)
            {
                return "none";
            }

            try
            {
                string? bos = model.Tokenizer.BosId is { } b ? model.Tokenizer.TokenString(b) : null;
                string? eos = model.Tokenizer.EosId is { } e ? model.Tokenizer.TokenString(e) : null;
                var template = new ChatTemplate(source, bos, eos);
                string text = template.Render([new ChatMessage("system", "You are helpful."), new ChatMessage("user", "Hi"), new ChatMessage("assistant", "Hello!"),
                    new ChatMessage("user", "Bye")], addGenerationPrompt: true);
                return $"ok {text.Length}";
            }
            catch (Exception error)
            {
                try
                {
                    var template = new ChatTemplate(source, null, null);
                    template.Render([new ChatMessage("user", "Hi")], addGenerationPrompt: true);
                    return "ok-user-only (" + OneLine(error.Message)[..Math.Min(120, error.Message.Length)] + ")";
                }
                catch (Exception second)
                {
                    return "FAIL " + OneLine(second.Message);
                }
            }
        }

        /// <summary>The ids llama.cpp's tokenizer gives for the test text (vocabulary only, so a sparse file works); null without LLAMA_BIN.</summary>
        private static int[]? LlamaTokenize(string path)
        {
            if (Environment.GetEnvironmentVariable("LLAMA_BIN") is not { } bin)
            {
                return null;
            }

            var start = new ProcessStartInfo(Path.Combine(bin, "llama-tokenize"))
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
            };
            foreach (var arg in new[] { "-m", path, "-p", TestText, "--ids", "--no-bos", "--log-disable" })
            {
                start.ArgumentList.Add(arg);
            }

            start.Environment["LD_LIBRARY_PATH"] = bin;
            using var process = Process.Start(start)!;
            string output = process.StandardOutput.ReadToEnd();
            process.StandardError.ReadToEnd();
            process.WaitForExit();
            string? line = output.Split('\n').LastOrDefault(l => l.StartsWith('['));
            if (process.ExitCode != 0 || line is null)
            {
                return [-1];
            }

            return [.. line.Trim('[', ']', ' ', '\r').Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(int.Parse)];
        }

        /// <summary>transformers' ids for the test text (HF_PYTHON: a python with transformers), as its JSON list; null without HF_PYTHON.</summary>
        private static string? HfTokenize(string dir)
        {
            if (Environment.GetEnvironmentVariable("HF_PYTHON") is not { } python)
            {
                return null;
            }

            var start = new ProcessStartInfo(python) { RedirectStandardOutput = true, RedirectStandardError = true };
            foreach (var arg in new[] { "-I", Path.Combine(AppContext.BaseDirectory, "hf_tokenize.py"), dir, TestText })
            {
                start.ArgumentList.Add(arg);
            }

            start.Environment["HF_HUB_OFFLINE"] = "1";
            start.Environment["TRANSFORMERS_VERBOSITY"] = "error";
            using var process = Process.Start(start)!;
            string output = process.StandardOutput.ReadToEnd();
            process.StandardError.ReadToEnd();
            process.WaitForExit();
            return output.Split('\n').LastOrDefault(l => l.StartsWith('[') || l.StartsWith("ERROR", StringComparison.Ordinal))?.Trim() ?? "ERROR no output";
        }

        private static string TypeName(int id) => GgufSupport.GetTensorType(id)?.Name ?? $"id{id}";
    }
}
