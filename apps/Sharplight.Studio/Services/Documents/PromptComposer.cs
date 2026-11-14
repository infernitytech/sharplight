using System.Text;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Documents;

/// <summary>Builds the text of a user message with its documents inlined.</summary>
internal static class PromptComposer
{
    /// <summary>The documents (as <c>&lt;document name="..."&gt;</c> blocks) followed by the message text.</summary>
    public static string Compose(string text, IEnumerable<AttachmentRecord> attachments)
    {
        var documents = attachments.Where(attachment => attachment.Kind == AttachmentKind.Document && !string.IsNullOrEmpty(attachment.Text)).ToList();
        if (documents.Count == 0)
        {
            return text;
        }

        var builder = new StringBuilder();
        foreach (var document in documents)
        {
            builder.Append("<document name=\"").Append(document.Name).AppendLine("\">");
            builder.AppendLine(document.Text);
            builder.AppendLine("</document>");
            builder.AppendLine();
        }

        builder.Append(text);
        return builder.ToString();
    }
}
