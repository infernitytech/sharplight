using Xunit;
using Sharplight.Speech.Normalization;

namespace Sharplight.Core.Tests.Speech;

/// <summary>The text normalizers of the supported languages: numbers, ordinals, dates, times, currency, percentages and abbreviations as spoken words.</summary>
public class TextNormalizationTests
{
    public static TheoryData<string, string, string> Cases => new()
    {
        // English
        { "en", "I have 3 apples and 21 pears.", "I have three apples and twenty-one pears." },
        { "en", "It costs $5.50.", "It costs five dollars and fifty cents." },
        { "en", "The population is 1,234,567.", "The population is one million two hundred thirty-four thousand five hundred sixty-seven." },
        { "en", "Pi is 3.14.", "Pi is three point one four." },
        { "en", "He finished 1st, she 22nd.", "He finished first, she twenty-second." },
        { "en", "Growth was 12.5% and -4 degrees.", "Growth was twelve point five percent and minus four degrees." },
        { "en", "Dr. Smith met Mr. Jones.", "Doctor Smith met Mister Jones." },
        { "en", "See you on May 3, 2024 at 10:30.", "See you on May third, twenty twenty-four at ten thirty." },
        { "en", "The meeting is at 9:05 pm.", "The meeting is at nine oh five p m." },
        { "en", "The date is 2024-12-25.", "The date is December twenty-fifth, twenty twenty-four." },
        { "en", "It was 1999.", "It was one thousand nine hundred ninety-nine." },
        { "en", "Call 007 now & win €20.", "Call zero zero seven now and win twenty euros." },
        { "en", "That is £3 million.", "That is three million pounds." },

        // German
        { "de", "Ich habe 3 Äpfel und 21 Birnen.", "Ich habe drei Äpfel und einundzwanzig Birnen." },
        { "de", "Das kostet 5,50 €.", "Das kostet fünf Euro und fünfzig Cent." },
        { "de", "Es gibt 1.234.567 Einwohner.", "Es gibt eine Million zweihundertvierunddreißigtausendfünfhundertsiebenundsechzig Einwohner." },
        { "de", "Pi ist 3,14.", "Pi ist drei Komma eins vier." },
        { "de", "Am 3.5.2024 um 10:30 Uhr.", "Am dritten Mai zweitausendvierundzwanzig um zehn Uhr dreißig." },
        { "de", "Am 24. Dezember 1999.", "Am vierundzwanzigsten Dezember neunzehnhundertneunundneunzig." },
        { "de", "Das sind 12,5 % z. B. für Dr. Müller.", "Das sind zwölf Komma fünf Prozent zum Beispiel für Doktor Müller." },
        { "de", "Es sind 101 und 1001 und 2100.", "Es sind einhunderteins und eintausendeins und zweitausendeinhundert." },
        { "de", "Um 8 Uhr.", "Um acht Uhr." },

        // French
        { "fr", "Le numéro est 3 puis 21.", "Le numéro est trois puis vingt et un." },
        { "fr", "Cela coûte 5,50 €.", "Cela coûte cinq euros et cinquante centimes." },
        { "fr", "Il y a 71 et 80 et 91 et 99.", "Il y a soixante et onze et quatre-vingts et quatre-vingt-onze et quatre-vingt-dix-neuf." },
        { "fr", "Il y a 200 et 201 et 2000 et 1 000 000.", "Il y a deux cents et deux cent un et deux mille et un million." },
        { "fr", "C'est le 1er janvier 2024 et le 3 mai.", "C'est le premier janvier deux mille vingt-quatre et le trois mai." },
        { "fr", "Le 2e et le 21e.", "Le deuxième et le vingt et unième." },
        { "fr", "À 10:30, soit 12,5 % de M. Dupont.", "À dix heures trente, soit douze virgule cinq pour cent de Monsieur Dupont." },
        { "fr", "Le 03/05/2024.", "Le trois mai deux mille vingt-quatre." },

        // Spanish
        { "es", "El número es 3 y luego 21.", "El número es tres y luego veintiuno." },
        { "es", "Cuesta 5,50 €.", "Cuesta cinco euros con cincuenta céntimos." },
        { "es", "Hay 100 y 101 y 1000 y 2500 y 1.000.000.", "Hay cien y ciento uno y mil y dos mil quinientos y un millón." },
        { "es", "El 3 de mayo de 2024 a las 10:30.", "El tres de mayo de dos mil veinticuatro a las diez y treinta." },
        { "es", "Es el 1.º y luego el 3º.", "Es el primero y luego el tercero." },
        { "es", "Un 12,5 % del Sr. López.", "Un doce coma cinco por ciento del señor López." },
        { "es", "Hay 1.500.000.000 y 31 y 1021.", "Hay mil quinientos millones y treinta y uno y mil veintiuno." },

        // Italian
        { "it", "Ho 3 mele e 21 pere.", "Ho tre mele e ventuno pere." },
        { "it", "Costa 5,50 €.", "Costa cinque euro e cinquanta centesimi." },
        { "it", "Ci sono 28 e 33 e 101 e 108 e 180.", "Ci sono ventotto e trentatré e centouno e centootto e centottanta." },
        { "it", "Ci sono 1000 e 2000 e 1.000.000 e 2023.", "Ci sono mille e duemila e un milione e duemilaventitré." },
        { "it", "Il 1° gennaio 2024 e il 3 maggio.", "Il primo gennaio duemilaventiquattro e il tre maggio." },
        { "it", "Il 12,5 % del Sig. Rossi.", "Il dodici virgola cinque per cento del signor Rossi." },
        { "it", "Il 21° e l'11° e il 23°.", "Il ventunesimo e l'undicesimo e il ventitreesimo." },

        // Portuguese
        { "pt", "O número é 3 e depois 21.", "O número é três e depois vinte e um." },
        { "pt", "Custa R$ 5,50.", "Custa cinco reais e cinquenta centavos." },
        { "pt", "Há 100 e 101 e 1000 e 1100 e 2020.", "Há cem e cento e um e mil e mil e cem e dois mil e vinte." },
        { "pt", "Há 1.000.000 e 2.500.000.", "Há um milhão e dois milhões e quinhentos mil." },
        { "pt", "Em 3 de maio de 2024 às 10:30.", "Em três de maio de dois mil e vinte e quatro às dez e trinta." },
        { "pt", "É o 1º e 12,5 % do Sr. Silva.", "É o primeiro e doze vírgula cinco por cento do senhor Silva." },

        // Chinese
        { "zh", "我有3个苹果和21个梨。", "我有三个苹果和二十一个梨。" },
        { "zh", "我有2个孩子。", "我有两个孩子。" },
        { "zh", "价格是¥5.50。", "价格是五元五角。" },
        { "zh", "2024年5月3日10:30。", "二零二四年五月三日十点三十分。" },
        { "zh", "一共101人，1005元，10000人。", "一共一百零一人，一千零五元，一万人。" },
        { "zh", "增长了12.5%，温度-4度。", "增长了百分之十二点五，温度负四度。" },
        { "zh", "电话是0108899。", "电话是零一零八八九九。" },
        { "zh", "共1234567人。", "共一百二十三万四千五百六十七人。" },

        // Japanese
        { "ja", "りんごが3個と21個。", "りんごが三個と二十一個。" },
        { "ja", "2024年5月3日の10:30に。", "二千二十四年五月三日の十時三十分に。" },
        { "ja", "値段は¥1500です。", "値段は千五百円です。" },
        { "ja", "10000人と100人と1000円。", "一万人と百人と千円。" },
        { "ja", "成長率は12.5%、気温は-4度。", "成長率は十二点五パーセント、気温はマイナス四度。" },
        { "ja", "２０２４年。", "二千二十四年。" },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Spells_the_text_as_spoken(string language, string written, string spoken)
        => Assert.Equal(spoken, TextNormalizers.For(language).Normalize(written));

    [Fact]
    public void Knows_every_supported_language_and_falls_back_to_English()
    {
        foreach (string language in TextNormalizers.Supported)
        {
            Assert.Equal(language, TextNormalizers.For(language).Language);
        }

        Assert.Equal("de", TextNormalizers.For("de-AT").Language);
        Assert.Equal("en", TextNormalizers.For("xx").Language);
        Assert.Equal("en", TextNormalizers.For(null).Language);
        Assert.Same(TextNormalizers.For("fr"), TextNormalizers.For("FR"));
    }

    [Fact]
    public void Detects_chinese_and_japanese_by_script()
    {
        Assert.Equal("zh", TextNormalizers.Detect("今天天气很好"));
        Assert.Equal("ja", TextNormalizers.Detect("今日はいい天気です"));
        Assert.Equal("en", TextNormalizers.Detect("hello"));
        Assert.Equal("de", TextNormalizers.Detect("hallo", "de"));
    }

    [Fact]
    public void Leaves_text_without_numbers_alone()
    {
        Assert.Equal("Hello, world!", TextNormalizers.For("en").Normalize("Hello, world!"));
        Assert.Equal("Guten Tag, wie geht es dir?", TextNormalizers.For("de").Normalize("Guten Tag, wie geht es dir?"));
    }
}
