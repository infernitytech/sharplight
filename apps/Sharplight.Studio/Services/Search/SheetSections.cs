using System.Text;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Spreadsheet;

namespace Sharplight.Studio.Services.Search;

/// <summary>Reads an XLSX sheet by sheet, one line per row (<c>cell | cell</c>), streaming so a large sheet is never held as a DOM.</summary>
internal static class SheetSections
{
    /// <summary>One section per sheet, headed by the sheet's name and numbered in workbook order.</summary>
    /// <param name="path">The XLSX.</param>
    /// <param name="limits">The sheet and cell limits.</param>
    /// <param name="cancellationToken">Cancels reading.</param>
    public static IReadOnlyList<DocumentSection> Read(string path, DocumentLimits limits, CancellationToken cancellationToken)
    {
        using var document = SpreadsheetDocument.Open(path, isEditable: false);
        var sections = new List<DocumentSection>();
        if (document.WorkbookPart is not { } workbook || workbook.Workbook?.Sheets is not { } sheets)
        {
            return sections;
        }

        var strings = SharedStrings(workbook, limits);
        int number = 0;
        foreach (var sheet in sheets.Elements<Sheet>())
        {
            number++;
            if (number > limits.MaxPages)
            {
                break;
            }

            if (sheet.Id?.Value is not { } id || workbook.GetPartById(id) is not WorksheetPart part)
            {
                continue;
            }

            string text = Rows(part, strings, limits, cancellationToken);
            if (text.Length > 0)
            {
                sections.Add(new DocumentSection(text, sheet.Name?.Value ?? $"Sheet {number}", number));
            }
        }

        return sections;
    }

    private static List<string> SharedStrings(WorkbookPart workbook, DocumentLimits limits)
    {
        var strings = new List<string>();
        if (workbook.SharedStringTablePart is not { } part)
        {
            return strings;
        }

        using var reader = OpenXmlReader.Create(part);
        while (reader.Read() && strings.Count < limits.MaxCells)
        {
            if (reader.ElementType == typeof(SharedStringItem) && reader.IsStartElement)
            {
                strings.Add(((SharedStringItem)reader.LoadCurrentElement()!).InnerText);
            }
        }

        return strings;
    }

    private static string Rows(WorksheetPart part, List<string> strings, DocumentLimits limits, CancellationToken cancellationToken)
    {
        var text = new StringBuilder();
        int cells = 0;
        using var reader = OpenXmlReader.Create(part);
        while (reader.Read() && cells < limits.MaxCells)
        {
            if (reader.ElementType != typeof(Row) || !reader.IsStartElement)
            {
                continue;
            }

            cancellationToken.ThrowIfCancellationRequested();
            var row = (Row)reader.LoadCurrentElement()!;
            var values = new List<string>();
            foreach (var cell in row.Elements<Cell>())
            {
                cells++;
                string value = ValueOf(cell, strings);
                if (value.Length > 0)
                {
                    values.Add(value);
                }
            }

            if (values.Count > 0)
            {
                text.AppendLine(string.Join(" | ", values));
            }
        }

        return text.ToString().Trim();
    }

    private static string ValueOf(Cell cell, List<string> strings)
    {
        string raw = cell.CellValue?.Text ?? "";
        if (cell.DataType?.Value == CellValues.SharedString)
        {
            return int.TryParse(raw, out int index) && index >= 0 && index < strings.Count ? strings[index] : "";
        }

        if (cell.DataType?.Value == CellValues.InlineString)
        {
            return cell.InlineString?.InnerText ?? "";
        }

        return raw;
    }
}
