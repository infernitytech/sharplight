using Sharplight.Distributed.Transport;

namespace Sharplight.Distributed;

/// <summary>A worker's reply and when the reader thread received it (a <see cref="System.Diagnostics.Stopwatch"/> timestamp).</summary>
internal readonly record struct WorkerReply(Frame Frame, long ArrivedAt);
