// Serves a local Sharplight model as a Microsoft Agent Framework agent over OpenAI-compatible HTTP endpoints.
//
//   dotnet run --project samples/Sharplight.AgentHost -- <model.gguf | model.slm | hf-dir> [--cpu] [--ctx 4096] [--urls http://localhost:5100]
//
//   Chat Completions:  POST /assistant/v1/chat/completions   {"model":"assistant","messages":[...],"stream":true}
//   Responses:         POST /assistant/v1/responses          {"input":"hello"}
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Hosting.OpenAI;
using Microsoft.Extensions.AI;
using Sharplight.Cpu;
using Sharplight.Extensions.AI;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Vulkan;

var argv = args.ToList();
string? Opt(string name, string? def = null) { int i = argv.IndexOf(name); return i >= 0 && i + 1 < argv.Count ? argv[i + 1] : def; }
bool Flag(string name) => argv.Contains(name);

if (argv.Count == 0 || argv[0].StartsWith("--"))
{
    Console.Error.WriteLine("usage: AgentHost <model> [--cpu] [--ctx N] [--urls url]");
    return 1;
}
string modelPath = argv[0];

ProviderRegistry.Register(() => new VulkanProvider());
CpuExecutionProvider.Register();
GgufImporter.Register();
HuggingFaceImporter.Register();
ModelPackageImporter.Register();

var builder = WebApplication.CreateBuilder(args.Where(a => a != modelPath).ToArray());
if (Opt("--urls") is { } urls) builder.WebHost.UseUrls(urls);

builder.Services.AddSharplightChatClient(modelPath, _ =>
{
    var provider = ProviderRegistry.Discover().First(x => x.Availability.IsAvailable && (x.Provider.Name == "cpu") == Flag("--cpu")).Provider;
    var slot = new DeviceSlot { Device = provider.OpenDevice(provider.GetDevices()[0]) };
    return new LlmLoadOptions { Devices = [slot], ContextLength = int.Parse(Opt("--ctx", "4096")!), Log = Console.Error.WriteLine };
}).UseFunctionInvocation();

builder.AddOpenAIChatCompletions();
builder.AddOpenAIResponses();
builder.AddOpenAIConversations();
var agent = builder.AddAIAgent("assistant", instructions: "You are a helpful assistant.");

var app = builder.Build();
app.MapOpenAIChatCompletions(agent);
app.MapOpenAIResponses(agent);
app.MapOpenAIConversations();
app.Run();
return 0;
