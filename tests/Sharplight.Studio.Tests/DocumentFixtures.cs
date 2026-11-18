using System.IO.Compression;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml;
using S = DocumentFormat.OpenXml.Spreadsheet;
using W = DocumentFormat.OpenXml.Wordprocessing;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;
using A = DocumentFormat.OpenXml.Drawing;
using P = DocumentFormat.OpenXml.Presentation;

namespace Sharplight.Studio.Tests;

/// <summary>Writes tiny documents of every format the Search page reads, with generated text only.</summary>
internal static class DocumentFixtures
{
    /// <summary>A PDF with one page per text.</summary>
    public static void Pdf(string path, params string[] pages)
    {
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        foreach (string text in pages)
        {
            var page = builder.AddPage(UglyToad.PdfPig.Content.PageSize.A4);
            page.AddText(text, 12, new PdfPoint(40, 700), font);
        }

        File.WriteAllBytes(path, builder.Build());
    }

    /// <summary>A Word document; a paragraph written as <c>#Title</c> uses the Heading1 style, and <c>|a|b</c> becomes a table row.</summary>
    public static void Docx(string path, params string[] paragraphs)
    {
        using var document = WordprocessingDocument.Create(path, WordprocessingDocumentType.Document);
        var main = document.AddMainDocumentPart();
        var body = new W.Body();
        main.Document = new W.Document(body);
        W.Table? table = null;
        foreach (string text in paragraphs)
        {
            if (text.StartsWith('|'))
            {
                table ??= new W.Table();
                var row = new W.TableRow();
                foreach (string cell in text.Trim('|').Split('|'))
                {
                    row.Append(new W.TableCell(new W.Paragraph(new W.Run(new W.Text(cell)))));
                }

                table.Append(row);
                continue;
            }

            if (table is not null)
            {
                body.Append(table);
                table = null;
            }

            var paragraph = new W.Paragraph();
            string content = text;
            if (text.StartsWith('#'))
            {
                paragraph.Append(new W.ParagraphProperties(new W.ParagraphStyleId { Val = "Heading1" }));
                content = text[1..];
            }

            paragraph.Append(new W.Run(new W.Text(content)));
            body.Append(paragraph);
        }

        if (table is not null)
        {
            body.Append(table);
        }
    }

    /// <summary>A presentation with one slide per text (lines separated by <c>\n</c>).</summary>
    public static void Pptx(string path, params string[] slides)
    {
        using var document = PresentationDocument.Create(path, PresentationDocumentType.Presentation);
        var presentation = document.AddPresentationPart();
        var ids = new P.SlideIdList();
        presentation.Presentation = new P.Presentation(ids);
        uint id = 256;
        foreach (string text in slides)
        {
            var part = presentation.AddNewPart<SlidePart>();
            var textBody = new P.TextBody(new A.BodyProperties(), new A.ListStyle());
            foreach (string line in text.Split('\n'))
            {
                textBody.Append(new A.Paragraph(new A.Run(new A.Text(line))));
            }

            var shape = new P.Shape(
                new P.NonVisualShapeProperties(new P.NonVisualDrawingProperties { Id = 2, Name = "Text" }, new P.NonVisualShapeDrawingProperties(), new P.ApplicationNonVisualDrawingProperties()),
                new P.ShapeProperties(),
                textBody);
            part.Slide = new P.Slide(new P.CommonSlideData(new P.ShapeTree(
                new P.NonVisualGroupShapeProperties(new P.NonVisualDrawingProperties { Id = 1, Name = "" }, new P.NonVisualGroupShapeDrawingProperties(), new P.ApplicationNonVisualDrawingProperties()),
                new P.GroupShapeProperties(),
                shape)));
            ids.Append(new P.SlideId { Id = id++, RelationshipId = presentation.GetIdOfPart(part) });
        }
    }

    /// <summary>A workbook with the sheets <c>name -> rows</c>; text cells use the shared string table, <c>#</c> prefixed cells are numbers, <c>~</c> prefixed cells inline strings.</summary>
    public static void Xlsx(string path, params (string Name, string[][] Rows)[] sheets)
    {
        using var document = SpreadsheetDocument.Create(path, SpreadsheetDocumentType.Workbook);
        var workbook = document.AddWorkbookPart();
        workbook.Workbook = new S.Workbook();
        var sheetList = workbook.Workbook.AppendChild(new S.Sheets());
        var shared = new S.SharedStringTable();
        uint sheetId = 1;
        foreach (var (name, rows) in sheets)
        {
            var part = workbook.AddNewPart<WorksheetPart>();
            var data = new S.SheetData();
            uint rowNumber = 1;
            foreach (string[] values in rows)
            {
                var row = new S.Row { RowIndex = rowNumber++ };
                foreach (string value in values)
                {
                    if (value.StartsWith('#'))
                    {
                        row.Append(new S.Cell { CellValue = new S.CellValue(value[1..]) });
                    }
                    else if (value.StartsWith('~'))
                    {
                        row.Append(new S.Cell { DataType = S.CellValues.InlineString, InlineString = new S.InlineString(new S.Text(value[1..])) });
                    }
                    else
                    {
                        shared.Append(new S.SharedStringItem(new S.Text(value)));
                        row.Append(new S.Cell { DataType = S.CellValues.SharedString, CellValue = new S.CellValue((shared.ChildElements.Count - 1).ToString(System.Globalization.CultureInfo.InvariantCulture)) });
                    }
                }

                data.Append(row);
            }

            part.Worksheet = new S.Worksheet(data);
            sheetList.Append(new S.Sheet { Id = workbook.GetIdOfPart(part), SheetId = sheetId++, Name = name });
        }

        workbook.AddNewPart<SharedStringTablePart>().SharedStringTable = shared;
    }

    /// <summary>A zip with one highly compressible part of <paramref name="bytes"/> zero bytes, named like a Word part (a stand-in for a zip bomb).</summary>
    public static void ZipBomb(string path, int bytes)
    {
        using var stream = File.Create(path);
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create);
        var entry = zip.CreateEntry("word/document.xml", CompressionLevel.SmallestSize);
        using var data = entry.Open();
        data.Write(new byte[bytes]);
    }
}
