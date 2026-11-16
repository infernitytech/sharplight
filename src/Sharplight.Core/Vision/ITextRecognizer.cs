using Sharplight.Tasks;

namespace Sharplight.Vision;

/// <summary>
/// Optical character recognition (kind <see cref="Models.ModelKinds.Ocr"/>): an image in, the text it shows out (TrOCR reads a line of text; models that locate
/// text return boxes). Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). Requests are serialized; the compute never runs on the
/// calling thread.
/// </summary>
public interface ITextRecognizer : ITaskClient
{
    /// <summary>Reads the text of an image.</summary>
    /// <param name="image">The image, any size.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels the request, also between the generated tokens.</param>
    public Task<TextRecognition> RecognizeAsync(ImageInput image, TextRecognitionOptions? options = null, CancellationToken cancellationToken = default);
}
