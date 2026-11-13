using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>The inputs <see cref="PackedSequences.Declare"/> adds to a packed-sequence graph.</summary>
/// <param name="Tokens">Token ids [T].</param>
/// <param name="Positions">Position of each token within its sequence [T, 1].</param>
/// <param name="RowSpans">Per token: first row and row count it attends to [T, 2].</param>
/// <param name="SequenceSpans">Per sequence: first row and length [B, 2].</param>
/// <param name="TokenTypes">Token type ids [T], when the model has token types.</param>
public sealed record PackedSequenceInputs(Value Tokens, Value Positions, Value RowSpans, Value SequenceSpans, Value? TokenTypes);
