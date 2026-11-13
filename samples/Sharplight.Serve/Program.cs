// sharplight-serve: load a model and serve it over the OpenAI and/or Anthropic HTTP API.
//
//   sharplight-serve <model.gguf | model.slm | hf-checkpoint-dir> [options]
//
//   --listen http://host:port   address to listen on (alias --urls; default http://127.0.0.1:8080)
//   --api openai|anthropic|both which API to expose (default both)
//   --api-key KEY               require this key (Authorization: Bearer / x-api-key); repeatable
//   --model-name NAME           model id reported by the API (default: file/directory name)
//   --device N | --cpu          GPU device index (default 0) / run on the CPU
//   --ctx N --batch N           context length per sequence (4096) / rows per forward pass (512)
//   --slots N                   concurrent sequences batched per forward pass, each with --ctx context (default 4)
//   --max-queue N               waiting requests before answering 429 (default 64)
//   --prefill-chunk N           prompt rows per step and sequence, interleaved with decoding (default 512)
//   --cpu-offload --disable-features a,b   as in the CLI
//   --mmproj vision.gguf        vision projector (images are not served yet)
//   --cors ORIGIN               allow a CORS origin ("*" for any); repeatable
//   --chat-template FILE        use this Jinja chat template instead of the model's
//   --temp T                    default temperature for requests that give none (default 0.7)
//
// Examples:
//   dotnet run -c Release --project samples/Sharplight.Serve -- ~/models/gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf --listen http://0.0.0.0:8080
//   curl localhost:8080/v1/chat/completions -H 'content-type: application/json' \
//        -d '{"model":"m","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
//   curl -N localhost:8080/v1/chat/completions -H 'content-type: application/json' \
//        -d '{"model":"m","stream":true,"messages":[{"role":"user","content":"Count to 5"}]}'
//   curl localhost:8080/v1/messages -H 'content-type: application/json' -H 'x-api-key: KEY' -H 'anthropic-version: 2023-06-01' \
//        -d '{"model":"m","max_tokens":64,"messages":[{"role":"user","content":"Say hi"}]}'
//   curl localhost:8080/health ; curl localhost:8080/metrics
//
// Python: openai.OpenAI(base_url="http://localhost:8080/v1", api_key="x") / anthropic.Anthropic(base_url="http://localhost:8080", api_key="x")

using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Server;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string name, string? def = null) { int i = argv.IndexOf(name); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : def; }
List<string> Opts(string name) { var l = new List<string>(); for (int i = 0; i + 1 < argv.Count; i++) if (argv[i] == name) l.Add(argv[i + 1]); return l; }
bool Flag(string name) => argv.Contains(name);

if (argv.Count == 0 || argv[0].StartsWith("--"))
{
    Console.Error.WriteLine("usage: sharplight-serve <model.gguf | model.slm | hf-dir> [--listen url] [--api openai|anthropic|both] [--api-key k] [--device n] [--ctx n] [--batch n] [--slots n] [--mmproj f] ...");
    return 1;
}

ProviderRegistry.Register(() => new VulkanProvider());
CpuExecutionProvider.Register();
GgufImporter.Register();
HuggingFaceImporter.Register();
ModelPackageImporter.Register();

// --- devices (as in the CLI) ---
var modelPath = argv[0];
var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && (x.Provider.Name == "cpu") == Flag("--cpu")).Provider;
var devOptions = new DeviceOptions();
foreach (var f in (Opt("--disable-features") ?? "").Split(',', StringSplitOptions.RemoveEmptyEntries)) devOptions.DisabledFeatures.Add(f.Trim());
var device = provider.OpenDevice(provider.GetDevices()[int.Parse(Opt("--device", "0")!)], devOptions);
DeviceSlot? fallback = null;
if (Flag("--cpu-offload")) { var cpu = new CpuExecutionProvider(); fallback = new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) }; }

// --- model, decoder, engine ---
using var model = LlmModel.Load(modelPath, new LlmLoadOptions
{
    Auxiliary = Opt("--mmproj") is { } mm ? [("mmproj", mm)] : [],
    Devices = [new DeviceSlot { Device = device }],
    Fallback = fallback,
    ContextLength = int.Parse(Opt("--ctx", "4096")!),
    MaxBatchTokens = int.Parse(Opt("--batch", "512")!),
    Log = Console.Error.WriteLine,
});
int slots = int.Parse(Opt("--slots", "4")!);
var decoder = CreateDecoder(model, slots);
var engine = new InferenceEngine(decoder, model.Tokenizer, new EngineOptions
{
    MaxQueue = int.Parse(Opt("--max-queue", "64")!),
    PrefillChunk = int.Parse(Opt("--prefill-chunk", "512")!),
});
string name = Opt("--model-name") ?? Path.GetFileNameWithoutExtension(modelPath.TrimEnd('/', '\\'));
var service = new ModelService(engine, model.ChatTemplate, new ModelServiceOptions
{
    ModelName = name,
    ChatTemplate = Opt("--chat-template") is { } tf ? File.ReadAllText(tf) : null,
});

// --- HTTP ---
string api = Opt("--api", "both")!;
var builder = WebApplication.CreateBuilder();
builder.WebHost.UseUrls(Opt("--listen") ?? Opt("--urls") ?? "http://127.0.0.1:8080");
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = null);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
builder.Services.AddSharplight(service, o =>
{
    o.OpenAi = api is "openai" or "both";
    o.Anthropic = api is "anthropic" or "both";
    o.ApiKeys.AddRange(Opts("--api-key"));
    o.CorsOrigins.AddRange(Opts("--cors"));
    o.DefaultSampling = new SamplingParams { Temperature = float.Parse(Opt("--temp", "0.7")!, System.Globalization.CultureInfo.InvariantCulture) };
});
var app = builder.Build();
app.UseSharplight();
app.MapSharplight();
Console.Error.WriteLine($"Serving '{name}' ({api} API, {decoder.SlotCount} slot(s), ctx {decoder.ContextLength}) on {string.Join(", ", builder.WebHost.GetSetting("urls")?.Split(';') ?? [])}");
await app.RunAsync();

engine.Dispose();
device.Dispose();
fallback?.Device.Dispose();
return 0;

// --slots N > 1: rows of up to N sequences run in one forward pass (continuous batching), each with --ctx tokens of
// context. Models whose stateful ops lack a multi-sequence form fall back to one slot (requests are queued).
static IBatchDecoder CreateDecoder(LlmModel model, int slots) => model.CreateBatchDecoder(slots);
