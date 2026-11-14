using System.Threading.Channels;
using Sharplight.Llm.Structured;
using Sharplight.Tokenizers;

namespace Sharplight.Llm.Serving;

/// <summary>
/// State of one submitted request inside the <see cref="InferenceEngine"/>: queue bookkeeping, prompt progress, the per-request
/// sampler / detokenizer / stop filter and the optional constraint matcher. Owned by the scheduler thread once admitted;
/// only <see cref="CancelRequested"/> and <see cref="Finished"/> are touched by other threads.
/// </summary>
internal sealed class EngineSequence
{
    public EngineSequence(long id, GenerationRequest request, ChannelWriter<GenerationChunk> writer, int maxTokens, long submitTicks)
    {
        Id = id;
        Request = request;
        Writer = writer;
        Prompt = request.PromptTokens;
        MaxTokens = maxTokens;
        SubmitTicks = submitTicks;
    }

    public long Id { get; }
    public GenerationRequest Request { get; }
    public ChannelWriter<GenerationChunk> Writer { get; }
    public int[] Prompt { get; }
    public int MaxTokens { get; }
    public long SubmitTicks { get; }
    public long DeadlineTicks { get; set; } = long.MaxValue;
    public long QueueDeadlineTicks { get; set; } = long.MaxValue;
    public CancellationTokenRegistration Registration { get; set; }

    /// <summary>Set from any thread; the scheduler finishes the request at its next step boundary.</summary>
    public volatile bool CancelRequested;
    /// <summary>1 once the final chunk was written (set with <see cref="Interlocked.Exchange(ref int, int)"/>).</summary>
    public int Finished;

    public EngineSlot? Slot { get; set; }
    /// <summary>Prompt tokens already in the slot's context.</summary>
    public int PromptPos { get; set; }
    /// <summary>Prompt tokens reused from the prefix cache.</summary>
    public int Cached { get; set; }
    /// <summary>Tokens generated so far.</summary>
    public int Generated { get; set; }
    public long LastTokenTicks { get; set; }
    /// <summary>When the request got a slot (0 while queued).</summary>
    public long AdmitTicks { get; set; }
    /// <summary>When the first token was generated (0 before).</summary>
    public long FirstTokenTicks { get; set; }
    /// <summary>Draft tokens verified for this request.</summary>
    public int DraftTokens { get; set; }
    /// <summary>Draft tokens accepted for this request.</summary>
    public int AcceptedDraftTokens { get; set; }

    /// <summary>The token sampled last and not yet fed to the model, followed by its draft tokens; reused as the step input.</summary>
    public int[] Input { get; private set; } = new int[1];
    /// <summary>Tokens of <see cref="Input"/> in use: 1 (the pending token) plus the draft length.</summary>
    public int InputLength { get; set; } = 1;
    public int Pending
    {
        get => Input[0];
        set => Input[0] = value;
    }

    public RequestSampler Sampler { get; set; } = null!;
    public StreamingDecoder Decoder { get; set; } = null!;
    public StopStringFilter Filter { get; set; } = null!;
    public HashSet<int> StopIds { get; set; } = null!;
    public IConstraintMatcher? Matcher { get; set; }
    /// <summary>Scratch for the constraint's allowed tokens (allocated on first use).</summary>
    public TokenMask? Mask { get; set; }
    /// <summary>Copy of the unmasked logits row for log-probabilities (allocated on first use).</summary>
    public float[]? RawLogits { get; set; }

    public bool Prefilling => PromptPos < Prompt.Length;

    /// <summary>
    /// Preempted (paged KV cache): the context tokens to compute again before the sequence continues, or null. Fed like a prompt
    /// without logits; <see cref="ReplayPos"/> of them are back in the cache.
    /// </summary>
    public int[]? Replay { get; set; }
    public int ReplayPos { get; set; }
    /// <summary>Masked diffusion: the sequence being denoised (created at its first step), or null.</summary>
    public DiffusionSequence? Diffusion { get; set; }
    /// <summary>Logical admission time (the newest running sequence is preempted first).</summary>
    public long AdmitTick { get; set; }

    /// <summary>Makes room for <paramref name="length"/> input tokens (pending plus drafts).</summary>
    public void EnsureInput(int length)
    {
        if (Input.Length < length)
        {
            var grown = new int[Math.Max(length, Input.Length * 2)];
            Input.CopyTo(grown, 0);
            Input = grown;
        }
    }
}
