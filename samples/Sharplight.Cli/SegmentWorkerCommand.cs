using System.CommandLine;
using Sharplight.Cpu;
using Sharplight.Distributed.Worker;
using Sharplight.Runtime;
using Sharplight.Samples;
using Sharplight.Samples.Hub;

namespace Sharplight.Cli;

/// <summary>
/// <c>segment-worker &lt;model&gt;</c>: serves the layers of a model to coordinators on other machines (multi-node pipeline parallel
/// inference, docs/multinode.md) until interrupted.
/// </summary>
internal sealed class SegmentWorkerCommand(StoreOptionSet store, SampleRuntime sampleRuntime)
{
    private readonly Argument<string> _model = new("model")
    {
        Description = "This node's copy of the model the coordinator runs: a .slm package, a .gguf file, a checkpoint directory or a stored model.",
    };

    private readonly RuntimeOptionSet _runtime = new();
    private readonly PullOptionSet _pull = new(explicitPull: false);

    private readonly Option<string> _listen = new("--listen")
    {
        HelpName = "host:port",
        Description = "Address to listen on (0.0.0.0:7070 for every interface; port 0 picks a free one).",
        DefaultValueFactory = _ => "0.0.0.0:7070",
    };

    private readonly Option<string?> _advertise = new("--advertise")
    {
        HelpName = "host:port",
        Description = "Address the neighbouring workers use to reach this one, when it differs from the address the coordinator connects to.",
    };

    private readonly Option<int> _gpus = OptionFactory.Integer("--gpus",
        "Place this worker's segments over this many GPUs, starting at --device.", 1, minimum: 1, helpName: "count");

    private readonly Option<long?> _memory = OptionFactory.Size("--memory",
        "Memory this worker offers per device (K/M/G suffixes; default: what the device reports as available). Coordinators weight their split by it.");

    public Command Build()
    {
        var command = new Command("segment-worker", "Serves the layers of a model to coordinators on other machines (multi-node pipeline parallel inference): "
            + "loads only the layers placed here, keeps their KV cache, and exchanges boundary activations with its neighbours. Runs until interrupted.");
        command.Arguments.Add(_model);
        _runtime.AddTo(command);
        _pull.AddTo(command);
        command.Options.Add(_listen);
        command.Options.Add(_advertise);
        command.Options.Add(_gpus);
        command.Options.Add(_memory);
        command.SetAction(HubGuard.Guard(async (parseResult, cancellationToken) =>
        {
            var runtime = _runtime.Read(parseResult);
            using var hub = new HubContext(sampleRuntime, store.ReadRoot(parseResult));
            var pull = _pull.Read(parseResult, _ => null);
            var path = await new ModelResolver(hub, pull).ResolveAsync(parseResult.GetValue(_model)!, cancellationToken);
            return await RunAsync(path, runtime, parseResult, cancellationToken);
        }));
        return command;
    }

    private async Task<int> RunAsync(string path, RuntimeSettings runtime, ParseResult parseResult, CancellationToken cancellationToken)
    {
        var devices = OpenDevices(runtime, parseResult.GetValue(_gpus), parseResult.GetValue(_memory));
        var fallback = runtime.CpuOffload ? OpenCpu() : null;
        try
        {
            var model = sampleRuntime.Loader.Build(path);
            using var worker = new SegmentWorker(model, new SegmentWorkerOptions
            {
                Listen = parseResult.GetValue(_listen)!,
                Advertise = parseResult.GetValue(_advertise),
                Devices = [.. devices],
                Fallback = fallback,
                ExpertsOnDisk = runtime.ExpertsOnDisk,
                ExpertCacheBytes = runtime.ExpertCacheBytes,
                Log = line => Console.Error.WriteLine($"[{DateTime.Now:HH:mm:ss}] {line}"),
            });
            worker.Start();
            Console.WriteLine($"listening on {worker.Endpoint}");
            try
            {
                await worker.Completion.WaitAsync(cancellationToken);
            }
            catch (OperationCanceledException)
            {
                Console.Error.WriteLine("segment worker stopping");
            }

            return ExitCodes.Success;
        }
        finally
        {
            foreach (var slot in devices)
            {
                slot.Device.Dispose();
            }

            fallback?.Device.Dispose();
        }
    }

    private List<DeviceSlot> OpenDevices(RuntimeSettings runtime, int gpus, long? memory)
    {
        var provider = sampleRuntime.SelectProvider(runtime);
        var options = new DeviceOptions();
        foreach (var feature in runtime.DisabledFeatures)
        {
            options.DisabledFeatures.Add(feature);
        }

        int count = provider is CpuExecutionProvider ? 1 : gpus;
        return [.. Enumerable.Range(runtime.DeviceIndex, count).Select(i => new DeviceSlot
        {
            Device = provider.OpenDevice(SampleRuntime.GetDevice(provider, i), options),
            MemoryLimit = memory,
        })];
    }

    private static DeviceSlot OpenCpu()
    {
        var cpu = new CpuExecutionProvider();
        return new DeviceSlot { Device = cpu.OpenDevice(cpu.GetDevices()[0]) };
    }
}
