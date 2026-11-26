namespace Sharplight.TimeSeries.Tests;

/// <summary>Reading series from delimited text.</summary>
public class TimeSeriesTableTests
{
    [Fact]
    public void Reads_a_header_a_time_column_and_numeric_columns()
    {
        var table = TimeSeriesTable.Parse("date,sales,\"price, eur\"\n2024-01-01,10,1.5\n2024-01-02,,1.25\n2024-01-03,12.5,x\n");
        Assert.Equal("date", table.TimeColumn);
        Assert.Equal(["2024-01-01", "2024-01-02", "2024-01-03"], table.Timestamps);
        Assert.Equal(["sales", "price, eur"], table.Columns.Select(c => c.Name));
        Assert.Equal(3, table.RowCount);
        Assert.Equal(10f, table.Column("SALES").Values[0]);
        Assert.True(float.IsNaN(table.Column("sales").Values[1]));
        Assert.True(float.IsNaN(table.Column("price, eur").Values[2]));
    }

    [Fact]
    public void Reads_semicolons_tabs_and_headerless_tables()
    {
        var semicolons = TimeSeriesTable.Parse("t;value\n1;3.5\n2;4.5\n3;5\n");
        Assert.Equal("t", semicolons.TimeColumn);
        Assert.Equal([3.5f, 4.5f, 5f], semicolons.Column("value").Values);
        var tabs = TimeSeriesTable.Parse("1.5\t2\n2.5\t3\n");
        Assert.Null(tabs.Timestamps);
        Assert.Equal(["column 1", "column 2"], tabs.Columns.Select(c => c.Name));
        var single = TimeSeriesTable.Parse("1\n2\n3\n4\n");
        Assert.Equal([1f, 2f, 3f, 4f], Assert.Single(single.Columns).Values);
    }

    [Fact]
    public void Rejects_tables_without_numbers()
    {
        Assert.Throws<FormatException>(() => TimeSeriesTable.Parse(""));
        Assert.Throws<FormatException>(() => TimeSeriesTable.Parse("a,b\nc,d\n"));
    }
}
