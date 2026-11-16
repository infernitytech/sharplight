using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>The span tables <see cref="SpanScoring.Declare"/> adds to a <see cref="Models.WellKnown.ScoreSpansGraph"/>, with their columns split out.</summary>
/// <param name="QuestionSpans">Per question: first row and row count [Q, 2].</param>
/// <param name="QuestionKinds">Per question: its kind [Q] (0 yes/no, 1 choice, 2 score).</param>
/// <param name="QuestionSequences">Per question: the index of the packed sequence it is in [Q].</param>
/// <param name="QuestionOptions">Per question: its first option and option count [Q, 2] (rows of the option table).</param>
/// <param name="OptionSpans">Per option: first row and row count [O, 2].</param>
/// <param name="OptionQuestions">Per option: the index of its question [O].</param>
internal sealed record SpanScoringInputs(Value QuestionSpans, Value QuestionKinds, Value QuestionSequences, Value QuestionOptions, Value OptionSpans, Value OptionQuestions);
