using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Vision;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Single-slot <see cref="IBatchDecoder"/> over an <see cref="InferenceSession"/>: inputs of a step run one after another, and
/// a context can only be reset, not truncated, so prefix reuse is limited to whole-context matches. The fallback of
/// <see cref="LlmModel.CreateBatchDecoder"/> for models whose graph has no multi-sequence form and for encoder-decoder models (the
/// prompt chunks of a new context are collected and encoded together; such contexts are never reused as prefixes). Not thread-safe.
/// </summary>
/// <param name="session">The session that holds the single context.</param>
/// <param name="ownsSession">Dispose <paramref name="session"/> together with the decoder.</param>
public sealed class SessionBatchDecoder(InferenceSession session, bool ownsSession = true) : IBatchDecoder, IDeviceSamplingDecoder
{
    private readonly float[] _logits = new float[session.VocabSize];

    /// <inheritdoc/>
    public int SlotCount => 1;
    /// <inheritdoc/>
    public int ContextLength => session.ContextLength;
    /// <inheritdoc/>
    public int MaxBatchTokens => (int)session.Compiled.Options.MaxTokens;
    /// <inheritdoc/>
    public int VocabSize => session.VocabSize;
    /// <inheritdoc/>
    public bool BatchesSequences => false;
    /// <inheritdoc/>
    public bool CanTruncate => false;
    /// <inheritdoc/>
    public bool ReusesPrefixes => !session.IsEncoderDecoder;
    /// <inheritdoc/>
    public long DeviceMemoryBytes => session.Compiled.ResidentMemoryBytes;
    /// <inheritdoc/>
    public int EmbeddingWidth => session.EmbeddingWidth;
    /// <inheritdoc/>
    public bool BidirectionalMediaRows => session.BidirectionalImageRows;
    /// <inheritdoc/>
    public int PositionAxes => Math.Max(1, session.PositionAxes);
    /// <inheritdoc/>
    public int TextPositionAxes => Math.Min(PositionAxes, session.TextPositionAxes);

    /// <inheritdoc/>
    public int Length(int slot)
    {
        Check(slot);
        return session.Position;
    }

    /// <inheritdoc/>
    public void Reset(int slot)
    {
        Check(slot);
        session.Reset();
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><paramref name="length"/> is neither 0 nor the current length.</exception>
    public void Truncate(int slot, int length)
    {
        Check(slot);
        if (length == session.Position)
        {
            return;
        }

        if (length != 0)
        {
            throw new NotSupportedException("This decoder cannot truncate a context.");
        }

        session.Reset();
    }

    /// <inheritdoc/>
    public int MaxLogitsRows => 1;

    /// <inheritdoc/>
    bool IDeviceSamplingDecoder.SamplesOnDevice => session.SamplesOnDevice;

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">An input asks for more than one logits row.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs, IReadOnlyList<SamplingRow?> sampling, Span<int> tokens, Span<float> logprobs)
    {
        var res = new Memory<float>[inputs.Count];
        for (int i = 0, row0 = 0; i < inputs.Count; row0 += inputs[i].OutputRows, i++)
        {
            Check(inputs[i].Slot);
            if (inputs[i].OutputRows > 1)
            {
                throw new NotSupportedException("This decoder returns the logits of the last row only.");
            }

            if (!inputs[i].WantLogits && session.IsEncoderDecoder && session.Position == 0)
            {
                session.AppendSource(inputs[i].Tokens.Span);   // a prompt chunk of an encoder-decoder model: encoded with the last one
                continue;
            }

            if (inputs[i].HasMultimodalRows)
            {
                var (prompt, embeddings) = ToPrompt(inputs[i]);
                if (inputs[i].WantLogits && i < sampling.Count && sampling[i] is { } mediaRow)
                {
                    (tokens[row0], logprobs[row0]) = session.ProcessSampled(prompt, embeddings, mediaRow);
                    continue;
                }

                var mediaLogits = session.Process(prompt, embeddings);
                if (inputs[i].WantLogits)
                {
                    mediaLogits.CopyTo(_logits, 0);
                    res[i] = _logits;
                }

                continue;
            }

            if (inputs[i].WantLogits && i < sampling.Count && sampling[i] is { } row)
            {
                (tokens[row0], logprobs[row0]) = session.ProcessSampled(inputs[i].Tokens.Span, row);
                continue;
            }

            var lg = session.Process(inputs[i].Tokens.Span);
            if (inputs[i].WantLogits)
            {
                lg.CopyTo(_logits, 0);
                res[i] = _logits;
            }
        }
        return res;
    }

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException">An input asks for more than one logits row.</exception>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs) => Step(inputs, [], [], []);

    /// <summary>
    /// An input with embedding rows, explicit positions or a position offset as the session's multimodal prompt: every row's position is
    /// given explicitly, and the session's rope position afterwards is its cache position (later inputs carry their offset again).
    /// </summary>
    private (MultimodalPrompt Prompt, IReadOnlyList<float[]> Embeddings) ToPrompt(SequenceInput input)
    {
        int n = input.Tokens.Length;
        int axes = session.PositionAxes;
        int width = session.EmbeddingWidth;
        if (width == 0)
        {
            throw new NotSupportedException("This decoder's session takes token rows only; create it with media input (LlmModel.CreateBatchDecoder) for embedding rows.");
        }

        var imageIndex = new int[n];
        var imageRow = new int[n];
        imageIndex.AsSpan().Fill(-1);
        var embeddings = new List<float[]>();
        foreach (var run in input.Embeddings ?? [])
        {
            int rows = run.Rows(width);
            if (rows * width != run.Values.Length || run.Row < 0 || run.Row + rows > n)
            {
                throw new ArgumentException($"An embedding run ({run.Values.Length} values at row {run.Row}) is not whole rows of width {width} inside the input's {n} rows.");
            }

            int first = 0;
            float[] values;
            if (MemoryMarshal.TryGetArray(run.Values, out var segment) && segment.Array is { } array && segment.Offset % width == 0)
            {
                values = array;
                first = segment.Offset / width;
            }
            else
            {
                values = run.Values.ToArray();
            }

            for (int j = 0; j < rows; j++)
            {
                imageIndex[run.Row + j] = embeddings.Count;
                imageRow[run.Row + j] = first + j;
            }

            embeddings.Add(values);
        }

        int[] positions;
        if (!input.Positions.IsEmpty)
        {
            if (input.Positions.Length != n * axes)
            {
                throw new ArgumentException($"The input has {input.Positions.Length} position ids for {n} rows of {axes} axes.");
            }

            positions = input.Positions.ToArray();
        }
        else
        {
            positions = new int[n * axes];
            for (int t = 0; t < n; t++)
            {
                for (int a = 0; a < session.TextPositionAxes && a < axes; a++)
                {
                    positions[t * axes + a] = session.Position + t + input.PositionOffset;
                }
            }
        }

        var media = input.MediaWeights.Span;
        var prompt = new MultimodalPrompt
        {
            Tokens = input.Tokens.ToArray(),
            ImageIndex = imageIndex,
            ImageRow = imageRow,
            Positions = positions,
            PositionAxes = axes,
            NextPosition = session.Position + n,
            MediaWeights = media.IsEmpty ? null : media.ToArray(),
        };
        return (prompt, embeddings);
    }

    private static void Check(int slot)
    {
        if (slot != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    /// <summary>Disposes the session when owned.</summary>
    public void Dispose()
    {
        if (ownsSession)
        {
            session.Dispose();
        }
    }
}
