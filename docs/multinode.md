# Multi-node inference (`Sharplight.Distributed`)

A model too large for one machine runs **pipeline parallel across several nodes**: the graph is split into per-device segments in
pipeline order exactly as over local GPUs ([hardware.md](hardware.md), `PlacementOptions.Devices`), and some of the devices are
**segment workers** on other machines. A worker loads only the layers placed on it from its own copy of the model, keeps the KV cache
and recurrent state of those layers, and exchanges only boundary activations with its neighbours over persistent TCP streams. Single-stream
decode gains capacity, not speed; throughput comes from keeping several micro-batches in flight so the stages overlap (below).

```
coordinator (tokenizer, scheduler,         worker A (layers 12-23,           worker B (layers 24-35, LM head,
 embeddings, layers 0-11)                   their KV cache)                    sampling head, their KV cache)
   │  hidden rows of the step ─────────────▶ │  hidden rows ────────────────────▶ │
   │◀──────────────────────────────────────────────────────── token ids (or logits) of the step
```

## Quick start

On every worker node, with that node's copy of the same model (any format the loader reads: `.slm`, GGUF, a checkpoint directory):

```
sharplight segment-worker ~/models/qwen3-8b-q4_k_m.gguf --listen 0.0.0.0:7070              # GPU 0 (Vulkan; --provider cuda)
sharplight segment-worker ~/models/qwen3-8b-q4_k_m.gguf --listen 0.0.0.0:7070 --cpu        # the CPU
sharplight segment-worker ~/models/big-moe.gguf --listen 0.0.0.0:7070 --cpu-offload --experts-on-disk
```

On the coordinator (it holds the first stage, the tokenizer and the scheduler; `--remote-only` places no layers on it):

```
sharplight ~/models/qwen3-8b-q4_k_m.gguf --workers 10.0.0.2:7070,10.0.0.3:7070 -p "The capital of France is"
sharplight serve ~/models/qwen3-8b-q4_k_m.gguf --workers 10.0.0.2:7070,10.0.0.3:7070
```

The split weights every stage by its free memory (a worker reports the sum over its devices, its CPU fallback included when it offloads);
`--split 1,2,2` gives one share per stage instead. After a run the CLI prints per worker the executions, the mean round trip, the
worker's own time and what the network and the protocol added.

Worker options: `--listen host:port` (port 0 picks one and prints it), `--advertise host:port` (the address neighbours use, when it
differs from the coordinator's), `--gpus N` (place this worker's segments over N local GPUs), `--memory 20G` (memory offered per device;
coordinators weight their split by it), and the runtime options of the other commands (`--provider`, `--device`, `--cpu`,
`--cpu-offload`, `--experts-on-disk`, `--expert-cache`, `--disable-features`).

## Design

### A provider whose devices are remote workers

`RemoteExecutionProvider` is an `IExecutionProvider` like the CUDA or CPU provider. Its devices (`RemoteComputeDevice`, provider name
`remote`, kind `DeviceKind.Other`) are segment workers:

- `GetMemoryInfo()` asks the worker for its live figures (the sum over its devices), so `PlacementPlanner` weights it by free memory and
  refuses a model that fits nowhere before anything is sent. `Info.Features` are the features every pipeline device of the worker has
  enabled (so the coordinator compiles the paged-KV and window-KV forms only when the worker runs them).
- `Supports(node)` asks the worker whether its devices (or its CPU fallback) run the node; the answer is cached per op, attributes and
  operand types, so a 600-node decode graph costs a few dozen questions.
- `Compile(segment, options)` serializes the segment (below) and sends it; the worker places it over its own devices with
  `CompiledModel.Compile`, so a worker can itself split its segment over several GPUs, offload layers or routed experts to its CPU and leave
  experts on disk ([hardware.md](hardware.md)). It returns an `ICompiledSegment` proxy (`RemoteCompiledSegment`).

Placement, compilation and the run loop of `CompiledModel` are unchanged: a pipeline over `[cuda:0, remote:0, remote:1]` is a pipeline over
three devices. Everything above it (sessions, the multi-sequence paged decoder, the serving engine, speculative decoding, structured
output) works over remote stages as over local ones.

### The segment on the wire

`SegmentCodec` (`Sharplight.Package`, next to the `.slm` IR codec) writes a segment as a standalone graph: the values its nodes read from
outside become the graph's inputs, the values later segments (or the caller) read become its outputs, weights are referenced by name and
checked for type and shape against the worker's own copy, persistent states (KV caches, recurrent states) travel whole, and constants are
inlined. The worker replays the nodes through `GraphBuilder`, which runs every op's type inference again: a worker of another Sharplight
version that infers different types refuses the segment. Control-flow nodes (`Loop`, `If`) stay on the coordinator (their bodies are
placed and compiled like any graph, so their segments can be remote too); string values never leave the process that owns them.

The worker never builds the model's graphs itself, so whatever the coordinator compiled (the session's window-KV layout, the paged
multi-sequence form, the sampling head, speculative-decoding rollback rows) runs on the worker without a second graph generation.

### Model copies are verified

Before it compiles a segment, the worker compares a fingerprint of the segment's weights with the coordinator's
(`RemoteSegmentOptions.WeightVerification`): by default the names, types and shapes plus three 4 KiB samples (start, middle, end) of every
stored weight; `Full` hashes every byte on both sides; `Metadata` only names, types and shapes. Weights the importer derives at load
(permuted q/k projections of Llama-family GGUF files) are compared by metadata unless `Full`. A different model, quantization or revision
fails the load with "this worker's model copy is another model, quantization or revision" instead of producing wrong tokens.

### Boundary values go from worker to worker

`RemoteCompiledSegment` implements `IDeviceForwardingSegment`: when two consecutive segments are remote, `CompiledModel` asks the consumer
to take the value from the producer, the consumer's worker registers a tag, and the producer's worker sends the value under that tag
straight to the consumer's worker after every execution (`PeerValue` frames over a persistent link between the two; within one worker
process the value is handed over in memory). The coordinator sends each worker its execution request (inputs set since the last step,
symbol bindings, runtime scalars) and waits only for the outputs it reads: a step over `[local, A, B]` crosses the network three times
(coordinator → A → B → coordinator). `PeerForwarding = false` (or `SHARPLIGHT_REMOTE_FORWARDING=0`) routes every value through the
coordinator instead (four hops for the same pipeline).

Executions are asynchronous (`ExecutesAsynchronously`): `Execute` returns once the request is sent, and the first output read waits for
the reply. A segment whose ops produce data-dependent symbols runs synchronously, because the next segment needs those symbols at once.

### Sampling on the last stage

When the devices compile a sampling head (`SamplingHead`, the default with GPU and remote devices), it is part of the last segment: the
last worker samples and only the token ids and log-probabilities return (8 bytes per row instead of `vocabulary · 4`, 600 KB per row for
a 152K vocabulary; 0.5 ms per row at 10 GbE). The full logits travel only when the caller reads them (logprobs top-k, grammar masks,
host-side sampling). Returning token ids to the first stage would save nothing more: the coordinator needs them anyway for the scheduler
and detokenization, and the embedding lookup of the next step runs on the first stage.

### Micro-batches and chunked prefill overlap the stages

- **Chunked prefill pipelines by itself.** A prefill chunk that wants no logits reads no output, so its `Run` returns as soon as the
  coordinator's own stage ran and the request is sent; the next chunk's first stage runs while the workers run the previous one. Causality
  holds because every stage processes the chunks of a sequence in order (§88: over two workers, 256 tokens in chunks of 64 prefill
  18-24 % faster than in one pass of 256, while in one process the same chunks are 26-38 % slower than one pass).
- **Several micro-batches in flight.** Several decoders (or engines) over the same remote devices step concurrently: each worker serves
  the executions of all of them in arrival order, so one micro-batch runs on stage 2 while the next runs on stage 1. Each has its own
  segments and KV cache on every worker; workers share the weights between them where the device shares weights
  (`IComputeDevice.SharesWeights`). This raises throughput when a stage's time grows with its rows (prefill, large decode batches); a
  memory-bound decode step reads all weights once per micro-batch, so splitting a small batch gains little (§88).

### Transport

`ISegmentTransport` (connect, listen) and `ISegmentConnection` (send and receive frames) are the extension points;
`TcpSegmentTransport` is the implementation for Ethernet:

- Frames: a 12-byte header (payload length, kind, request id, little-endian) and the payload. Request ids pair replies with requests, so
  several requests are in flight on one connection. A frame longer than `MaxFrameBytes` (1 GiB) or with an unknown kind breaks the stream.
- One persistent connection per peer, `TCP_NODELAY`, keep-alive, 4 MiB socket buffers; small frames go out in one write.
- TLS and mutual TLS: `TcpSegmentTransportOptions.ClientTls` returns the client options per worker (client certificate, validation
  callback, target host), `ServerTls` holds the worker's certificate and its client-certificate policy. Workers use `ClientTls` for their
  links to neighbours too.
- An RDMA transport (InfiniBand, RoCE) implements the same two interfaces, e.g. with registered buffers for large payloads.

The protocol: `Hello`/`HelloAck` (magic, version, role, the worker's devices, memory, model and instance id), `Ping`/`Pong`,
`MemoryRequest`, `SupportsRequest`, `Compile`, `Execute`/`ExecuteReply`, `Prepare`, `ResetState`, `CopyStateRows`, `Release`,
`AddForward`/`AddForwardedInput`, `PeerValue`, `Abort`, `Error` (with a kind: general, out of memory, model mismatch).

### Failure semantics: fail fast, rebuild

- The coordinator pings every connection that has been idle for half the `HeartbeatInterval` (1 s); a worker silent for
  `HeartbeatTimeout` (5 s: no reply, no pong) is dead. A closed connection (a process that exits) is noticed at once.
- A request without a reply within `ExecuteTimeout` (10 min; `CompileTimeout` 60 min for compilations) fails the connection: the worker's
  state is unknown afterwards.
- The remote segments of one compiled model form a pipeline that fails together (`RemotePipeline`): the first failure (an error, a lost
  connection, a timeout) makes every segment of the model throw the same `SegmentWorkerException`, now and for every later step, and
  every other worker is told to abort (a worker waiting for a value from a dead neighbour would otherwise wait for its timeout). A worker
  that fails an execution sends its neighbours a failure marker instead of the value, so they fail at once too.
- The request fails cleanly; to continue, dispose what was compiled and load the model again (with the worker restarted or replaced). A
  restarted worker is reached again at the same address: the coordinator's `RemoteDevicePool` reconnects on the next open, and workers drop
  a link to a neighbour as soon as the neighbour closes it.
- Workers release everything a coordinator compiled when its connection closes or stays silent for the worker's `HeartbeatTimeout` (15 s).

Measured on loopback (§88): a killed worker fails the coordinator's next step in 0.2-0.34 s; a stopped (hung) worker after 5.8 s with the
5 s heartbeat timeout; a coordinator against restarted workers runs normally.

## API

```csharp
// Coordinator: the workers as devices after the local GPU.
using var remote = new RemoteExecutionProvider(new RemoteSegmentOptions { Workers = ["10.0.0.2:7070", "10.0.0.3:7070"] });
var workers = remote.OpenAll();                                         // RemoteComputeDevice per worker, in pipeline order
using var model = LlmModel.Load(loader, "qwen3-8b-q4_k_m.gguf", new LlmLoadOptions
{
    Devices = [new DeviceSlot { Device = gpu }, .. workers.Select(w => new DeviceSlot { Device = w })],
});
// ... sessions, batch decoders and engines as with local devices; workers[i].Statistics has round trips, worker times, bytes.

// Worker: serve a model's layers on local devices.
var definition = services.GetRequiredService<ModelLoader>().Build("qwen3-8b-q4_k_m.gguf");
using var worker = new SegmentWorker(definition, new SegmentWorkerOptions
{
    Listen = "0.0.0.0:7070",
    Devices = [new DeviceSlot { Device = gpu }],
    Fallback = new DeviceSlot { Device = cpu },         // optional CPU offload
    ExpertsOnDisk = true,                               // optional, with a CPU fallback
});
worker.Start();
await worker.Completion;
```

| Type | What it is |
|---|---|
| `RemoteSegmentOptions` | workers (`host:port`, pipeline order), transport, heartbeat interval and timeout, execute and compile timeouts, weight verification, peer forwarding |
| `RemoteExecutionProvider` | the provider: `GetDevices()` connects and describes every worker, `OpenDevice`, `OpenAll()` |
| `RemoteComputeDevice` | one worker as a device: `Endpoint`, `Worker` (`SegmentWorkerInfo`: devices, memory, model, version), `Fault`, `Statistics`, `ResetStatistics()` |
| `RemoteDevicePool` | opens each worker once for every load that names it, reconnects after a failure (`services.AddRemoteSegments()`; hosting uses it) |
| `SegmentWorker`, `SegmentWorkerOptions` | the worker: listen and advertised address, devices, fallback, experts on disk, transport, heartbeat timeout, log |
| `ISegmentTransport`, `ISegmentConnection`, `ISegmentListener`, `TcpSegmentTransport`, `TcpSegmentTransportOptions`, `SegmentEndpoint`, `Frame`, `FrameKind` | the transport and its TLS hooks |
| `SegmentWorkerException` | a worker failed, closed, went silent or timed out (`Worker`, `WorkerAlive`) |
| `RemoteSegmentStatistics` | executions, round-trip, worker and wait times, bytes; `OverheadMsPerExecution` |

Hosting: `HostedModelOptions.Workers` (JSON `"workers": ["10.0.0.2:7070", ...]`) appends the workers to the model's devices in pipeline
order, each weighted by its free memory beyond the reserve; `AddSharplightHosting()` registers `AddRemoteSegments()`, whose options
(`services.Configure<RemoteSegmentOptions>(...)`) carry the transport and its certificates. Automatic device selection never picks a remote
worker: the remote provider is not in the execution provider catalog.

## What a cluster server provides

The engine needs three things from the server that runs it on several nodes (the server's scaling row 21: all shards of a replica in one
node group with a round trip of at most 1 ms, persistent streams between neighbours, one replica is one failure domain):

1. **Discovery and placement.** Which nodes serve which stages of a replica, and their addresses: the server starts a segment worker on
   every shard node (the `SegmentWorker` API, or `sharplight segment-worker`) and gives the owner node the ordered list
   (`HostedModelOptions.Workers` or `RemoteSegmentOptions.Workers`), with `--advertise` / `SegmentWorkerOptions.Advertise` where nodes
   reach each other at other addresses than the owner does. The engine splits by the workers' free memory unless the server gives shares
   (`DeviceSlot.Weight`).
2. **Certificates.** `TcpSegmentTransportOptions.ServerTls` on every worker (its node certificate, client certificates required and
   validated against the cluster CA) and `ClientTls` on the owner and on the workers (their certificate, the CA pinned in the validation
   callback): the cluster's existing mTLS identities serve as they are. Without TLS the worker port must only be reachable inside the node
   group.
3. **Model copies.** Every shard node needs the same model (the same `.slm` package or file) on local disk; the worker reads only its
   layers' bytes, and the fingerprint check refuses a different copy. The owner node needs the file as well (it builds the graphs and reads
   its own stage's weights; the other stages' weights are not read on it except for the fingerprint samples).

On a `SegmentWorkerException` the replica is failed: the server reloads it (restarting or replacing the lost worker), as it does for a lost
single-node replica. `HostStatus` of the owner shows the model's devices, the remote ones included; each worker logs connections,
compilations and releases.

## Testing

`tests/Sharplight.Distributed.Tests`: framing and payloads, TCP on loopback (large frames, closed and refused connections, oversized
frames, mutual TLS with self-signed certificates, a plain client refused by a TLS worker), heartbeats (a worker that goes silent after the
handshake fails within the timeout), pipelines over in-process workers on loopback (SmolLM2-135M, a tiny DeepSeek MoE with latent
attention, a tiny hybrid recurrent + attention MoE: tokens and logits equal a single CPU run, with and without peer forwarding; all stages
remote with several sequences; the session path; a killed worker fails the next step within seconds and every step after at once; a worker
with another model or a changed weight is refused; placement weights workers by their memory) and worker processes of the CLI build
(`MultiProcessTests`, skipped when the CLI is not built). `bench/PipelineBench` measures decode and prefill through workers against the same
split in one process, and the transport's round trip (`--echo` / `--pingpong`).

## Known gaps and next steps

- **Decode micro-batches in the engine.** Several decoders over the same workers overlap their steps (§88), but one serving engine runs one
  decoder; the next step is an engine that keeps two or more groups of sequences in flight, each stepping as soon as its previous step
  returned (no bubble, unlike splitting one step), or several engines over one pipeline behind one model name.
- **Lower protocol overhead.** A step adds 0.15-0.35 ms per worker on loopback (thread hand-offs on both ends: reader to executor on the
  worker, reader to the waiting thread on the coordinator); spinning briefly before blocking and writing replies from the executor's
  buffer would take most of it.
- **Boundary values in f16 / bf16** halve the bytes on the wire (an option, since the outputs then differ from the single-machine run).
- **RDMA transport** (InfiniBand / RoCE) behind `ISegmentTransport`; **tensor and expert parallel across nodes** for RDMA networks and
  large MoE models (over TCP the per-layer all-reduces cost several milliseconds per token, see [roadmap.md](roadmap.md)).
- **A coordinator without the model file**: the coordinator needs the graphs and its own stage's weights; a metadata-only package would
  spare it the other stages' bytes on disk.
- Studio does not create remote stages yet.
