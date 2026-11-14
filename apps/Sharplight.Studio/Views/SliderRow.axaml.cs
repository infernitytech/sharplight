using System.Globalization;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;

namespace Sharplight.Studio.Views;

/// <summary>A labelled slider showing its value.</summary>
internal sealed partial class SliderRow : UserControl
{
    /// <summary>Label text.</summary>
    public static readonly StyledProperty<string> LabelProperty = AvaloniaProperty.Register<SliderRow, string>(nameof(Label), "");

    /// <summary>The value (two-way).</summary>
    public static readonly StyledProperty<double> ValueProperty = AvaloniaProperty.Register<SliderRow, double>(nameof(Value), defaultBindingMode: BindingMode.TwoWay);

    /// <summary>Lowest value.</summary>
    public static readonly StyledProperty<double> MinimumProperty = AvaloniaProperty.Register<SliderRow, double>(nameof(Minimum));

    /// <summary>Highest value.</summary>
    public static readonly StyledProperty<double> MaximumProperty = AvaloniaProperty.Register<SliderRow, double>(nameof(Maximum), 1);

    /// <summary>Step.</summary>
    public static readonly StyledProperty<double> SmallChangeProperty = AvaloniaProperty.Register<SliderRow, double>(nameof(SmallChange), 0.01);

    /// <summary>Number format of the shown value.</summary>
    public static readonly StyledProperty<string> FormatProperty = AvaloniaProperty.Register<SliderRow, string>(nameof(Format), "0.00");

    public SliderRow()
    {
        InitializeComponent();
        Slider.ValueChanged += (_, args) =>
        {
            double snapped = SmallChange > 0 ? Math.Round(args.NewValue / SmallChange) * SmallChange : args.NewValue;
            if (Math.Abs(snapped - Value) > 1e-9)
            {
                Value = snapped;
            }
        };
        Sync();
    }

    /// <summary>Label text.</summary>
    public string Label { get => GetValue(LabelProperty); set => SetValue(LabelProperty, value); }

    /// <summary>The value.</summary>
    public double Value { get => GetValue(ValueProperty); set => SetValue(ValueProperty, value); }

    /// <summary>Lowest value.</summary>
    public double Minimum { get => GetValue(MinimumProperty); set => SetValue(MinimumProperty, value); }

    /// <summary>Highest value.</summary>
    public double Maximum { get => GetValue(MaximumProperty); set => SetValue(MaximumProperty, value); }

    /// <summary>Step.</summary>
    public double SmallChange { get => GetValue(SmallChangeProperty); set => SetValue(SmallChangeProperty, value); }

    /// <summary>Number format of the shown value.</summary>
    public string Format { get => GetValue(FormatProperty); set => SetValue(FormatProperty, value); }

    /// <inheritdoc/>
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == LabelProperty || change.Property == ValueProperty || change.Property == MinimumProperty || change.Property == MaximumProperty
            || change.Property == SmallChangeProperty || change.Property == FormatProperty)
        {
            Sync();
        }
    }

    private void Sync()
    {
        if (Slider is null)
        {
            return;
        }

        LabelText.Text = Label;
        Slider.Minimum = Minimum;
        Slider.Maximum = Maximum;
        Slider.SmallChange = SmallChange;
        Slider.LargeChange = SmallChange * 10;
        if (Math.Abs(Slider.Value - Value) > 1e-9)
        {
            Slider.Value = Value;
        }

        ValueText.Text = Value.ToString(Format, CultureInfo.InvariantCulture);
    }
}
