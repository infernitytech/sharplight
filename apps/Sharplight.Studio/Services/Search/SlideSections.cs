using System.Text;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using A = DocumentFormat.OpenXml.Drawing;

namespace Sharplight.Studio.Services.Search;

/// <summary>Reads a PPTX slide by slide (the slide's text and its speaker notes).</summary>
internal static class SlideSections
{
    /// <summary>One section per slide, in presentation order, headed by the slide's first line.</summary>
    /// <param name="path">The PPTX.</param>
    /// <param name="limits">The slide limit.</param>
    /// <param name="cancellationToken">Cancels reading.</param>
    public static IReadOnlyList<DocumentSection> Read(string path, DocumentLimits limits, CancellationToken cancellationToken)
    {
        using var document = PresentationDocument.Open(path, isEditable: false);
        var sections = new List<DocumentSection>();
        if (document.PresentationPart is not { } presentation || presentation.Presentation?.SlideIdList is not { } ids)
        {
            return sections;
        }

        int number = 0;
        foreach (var id in ids.Elements<DocumentFormat.OpenXml.Presentation.SlideId>())
        {
            cancellationToken.ThrowIfCancellationRequested();
            number++;
            if (number > limits.MaxPages)
            {
                throw new InvalidDataException($"The presentation has more than {limits.MaxPages} slides.");
            }

            if (id.RelationshipId?.Value is not { } relationship || presentation.GetPartById(relationship) is not SlidePart slide)
            {
                continue;
            }

            string text = TextOf(slide.Slide);
            if (slide.NotesSlidePart?.NotesSlide is { } notes && TextOf(notes) is { Length: > 0 } speaker)
            {
                text += "\nNotes: " + speaker;
            }

            if (text.Trim().Length > 0)
            {
                string heading = text.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "";
                sections.Add(new DocumentSection(text.Trim(), heading.Length > 120 ? heading[..120] : heading, number));
            }
        }

        return sections;
    }

    private static string TextOf(OpenXmlElement root)
    {
        var text = new StringBuilder();
        foreach (var paragraph in root.Descendants<A.Paragraph>())
        {
            string line = paragraph.InnerText.Trim();
            if (line.Length > 0)
            {
                text.AppendLine(line);
            }
        }

        return text.ToString().Trim();
    }
}
