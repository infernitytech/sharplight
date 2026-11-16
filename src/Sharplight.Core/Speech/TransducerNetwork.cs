using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.Speech;

/// <summary>
/// Runs the prediction and joint networks of a transducer speech model (<see cref="WellKnown.TransducerPredictGraph"/>,
/// <see cref="WellKnown.TransducerJointGraph"/>) and keeps the prediction network's recurrent state on the host. Not thread-safe.
/// </summary>
internal sealed class TransducerNetwork : IDisposable
{
    private readonly CompiledModel _predict;
    private readonly CompiledModel _joint;
    private readonly Value _token;
    private readonly Value _hiddenIn;
    private readonly Value _cellIn;
    private readonly Value _frame;
    private readonly Value _predictionIn;
    private readonly float[] _hidden;
    private readonly float[] _cell;
    private readonly float[] _prediction;
    private readonly float[] _logits;
    private readonly RunParameters _run = new();

    /// <summary>Compiles the decoder component's graphs.</summary>
    /// <param name="model">The loaded model.</param>
    public TransducerNetwork(TaskModel model)
    {
        _predict = model.CompileComponent(ModelComponentRoles.Decoder, WellKnown.TransducerPredictGraph);
        try
        {
            _joint = model.CompileComponent(ModelComponentRoles.Decoder, WellKnown.TransducerJointGraph);
        }
        catch
        {
            _predict.Dispose();
            throw;
        }

        _token = _predict.Graph.Input(InputRole.TokenIds);
        _hiddenIn = _predict.Graph.Inputs.Single(v => v.Name == WellKnown.TransducerHiddenState);
        _cellIn = _predict.Graph.Inputs.Single(v => v.Name == WellKnown.TransducerCellState);
        _frame = _joint.Graph.Input(InputRole.EncoderOutput);
        _predictionIn = _joint.Graph.Input(InputRole.HiddenStates);
        _hidden = new float[_hiddenIn.Shape.Elements()];
        _cell = new float[_cellIn.Shape.Elements()];
        _prediction = new float[_predictionIn.Shape.Elements()];
        _logits = new float[(int)_joint.Graph.Outputs[WellKnown.LogitsOutput].Shape[-1].Fixed];
    }

    /// <summary>Values of a projected encoder row.</summary>
    public int Width => _prediction.Length;

    /// <summary>Logits of the joint network (tokens, then durations).</summary>
    public int Outputs => _logits.Length;

    /// <summary>Clears the recurrent state and predicts from <paramref name="start"/> (the blank token starts a transcript).</summary>
    /// <param name="start">The start token.</param>
    public void Reset(int start)
    {
        Array.Clear(_hidden);
        Array.Clear(_cell);
        Predict(start);
    }

    /// <summary>Advances the prediction network by an emitted token.</summary>
    /// <param name="token">The token.</param>
    public void Predict(int token)
    {
        _predict.SetInput<int>(_token, [token]);
        _predict.SetInput<float>(_hiddenIn, _hidden);
        _predict.SetInput<float>(_cellIn, _cell);
        _predict.Run(_run);
        _predict.GetOutput<float>(WellKnown.TransducerPrediction, _prediction);
        _predict.GetOutput<float>(WellKnown.TransducerHiddenState, _hidden);
        _predict.GetOutput<float>(WellKnown.TransducerCellState, _cell);
    }

    /// <summary>The joint network's logits for one projected encoder row and the current prediction (the array is reused).</summary>
    /// <param name="row">The projected encoder row [<see cref="Width"/>].</param>
    public float[] Joint(ReadOnlySpan<float> row)
    {
        _joint.SetInput(_frame, row);
        _joint.SetInput<float>(_predictionIn, _prediction);
        _joint.Run(_run);
        _joint.GetOutput<float>(WellKnown.LogitsOutput, _logits);
        return _logits;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _joint.Dispose();
        _predict.Dispose();
    }
}
