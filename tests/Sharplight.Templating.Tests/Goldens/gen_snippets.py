# Generates snippets.json: template snippets rendered by real Python jinja2 (HF-style environment).
# Usage: venv/bin/python gen_snippets.py out.json
import json, sys, datetime
import jinja2, jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXED_NOW = datetime.datetime(2026, 1, 15, 10, 30, 45)

def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

def make_env(trim=True, lstrip=True, sandbox=True, keep_nl=False):
    cls = ImmutableSandboxedEnvironment if sandbox else jinja2.Environment
    env = cls(trim_blocks=trim, lstrip_blocks=lstrip, keep_trailing_newline=keep_nl, extensions=[jinja2.ext.loopcontrols, jinja2.ext.do])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: FIXED_NOW.strftime(fmt)
    return env

def render(source, ctx, **opts):
    env = make_env(**opts)
    return env.from_string(source).render(**ctx)

S = []  # (name, template, ctx, opts)
def t(name, template, ctx=None, **opts):
    S.append((name, template, ctx or {}, opts))

# ---- literals & output
t("int", "{{ 42 }}")
t("float", "{{ 1.5 }} {{ 1.0 }} {{ 0.1 + 0.2 }} {{ 1/3 }} {{ 2 ** 0.5 }}")
t("float_exp", "{{ 1e16 }} {{ 1.5e-7 }} {{ 123456789012345678.0 }} {{ 0.0001 }} {{ 0.00001 }} {{ 1e15 }}")
t("bool_none", "{{ true }} {{ false }} {{ none }} {{ True }} {{ False }} {{ None }}")
t("string_escapes", r"""{{ 'a\nb\t"q" \\ \'s\'' }}|{{ "xéy\x41" }}""")
t("string_adjacent", "{{ 'a' 'b' }}")
t("list_repr", "{{ ['a', \"b'c\", 1, none, true, 1.5, {'k': [1, 'x']}] }}")
t("tuple_repr", "{{ (1, 2) }} {{ [] }} {{ {} }}")
t("dict_literal", "{{ {'a': 1, 'b': 2}.b }} {{ {'a': {'b': 3}}.a.b }} {{ {1: 'x'}[1] }}")
t("undefined_prints_empty", "[{{ nope }}][{{ nope|default('d') }}]")
t("long_int", "{{ 9007199254740993 }} {{ 2 ** 62 }}")

# ---- arithmetic & logic
t("arith", "{{ 1 + 2 }} {{ 5 - 7 }} {{ 3 * 4 }} {{ 10 / 4 }} {{ 10 / 5 }} {{ 7 // 2 }} {{ -7 // 2 }} {{ 7 % 3 }} {{ -7 % 3 }} {{ 2 ** 10 }} {{ 2 ** -1 }}")
t("arith_float", "{{ 7.5 // 2 }} {{ 7.5 % 2 }} {{ 1.5 * 2 }} {{ 3 - 0.5 }} {{ -7.5 % 2 }}")
t("precedence", "{{ 1 + 2 * 3 }} {{ (1 + 2) * 3 }} {{ 2 ** 3 ** 2 }} {{ -2 ** 2 }} {{ 10 - 4 - 3 }}")
t("concat", "{{ 'a' ~ 1 ~ none ~ true ~ 1.5 ~ [1] }}")
t("str_ops", "{{ 'ab' * 3 }} {{ 'a' + 'b' }} {{ [1] + [2] }} {{ [0] * 3 }}")
t("compare", "{{ 1 < 2 }} {{ 2 <= 2 }} {{ 'a' < 'b' }} {{ 1 == 1.0 }} {{ 'a' != 'b' }} {{ 1 < 2 < 3 }} {{ 1 < 3 < 2 }} {{ [1,2] == [1,2] }} {{ {'a':1} == {'a':1} }}")
t("compare_bool", "{{ true == 1 }} {{ true + true }} {{ false < true }}")
t("in_ops", "{{ 1 in [1,2] }} {{ 3 not in [1,2] }} {{ 'a' in 'cat' }} {{ 'k' in {'k': 1} }} {{ 'z' in {'k': 1} }} {{ 'x' in none_var }}")
t("bool_ops", "{{ 1 and 2 }} {{ 0 or 'x' }} {{ '' or none }} {{ not 0 }} {{ [] and 1 }} {{ none or [] }}")
t("ternary", "{{ 'a' if true else 'b' }} {{ 'a' if false else 'b' }} [{{ 'a' if false }}] {{ 1 if false else 2 if false else 3 }}")
t("truthiness", "{% for v in [0, 1, '', 'a', [], [0], {}, {'a':1}, none, 0.0, 0.5] %}{{ 'T' if v else 'F' }}{% endfor %}")
t("not_precedence", "{{ not 1 == 2 }} {{ not 'a' in 'abc' }} {{ not x is defined }}")

# ---- subscripts, attrs, slices
t("getitem", "{{ [10,20,30][0] }} {{ [10,20,30][-1] }} {{ 'abc'[1] }} [{{ [1][5] }}] {{ d['a'] }} {{ d.a }} {{ d['b'][0] }} {{ d.b.0 }}", {"d": {"a": 1, "b": [7, 8]}})
t("slices", "{{ l[1:] }} {{ l[:2] }} {{ l[::-1] }} {{ l[-2:] }} {{ l[:-1] }} {{ l[::2] }} {{ l[1:3] }} {{ l[10:] }} {{ l[::-2] }} {{ l[3:0:-1] }} {{ l[-1:-4:-1] }}", {"l": [1, 2, 3, 4, 5]})
t("string_slices", "{{ s[::-1] }} {{ s[1:4] }} {{ s[-3:] }} {{ s[:0] }}", {"s": "abcdef"})
t("dict_attr_vs_key", "{{ d.get('k') }} {{ d['items'] }} {{ d.k }}", {"d": {"k": "v", "items": [1]}})

# ---- statements: if / for / set
t("if_elif_else", "{% if x == 1 %}one{% elif x == 2 %}two{% else %}many{% endif %}", {"x": 2})
t("if_nested", "{% if a %}{% if b %}ab{% else %}a{% endif %}{% else %}none{% endif %}", {"a": True, "b": False})
t("for_basic", "{% for i in [1,2,3] %}{{ i }},{% endfor %}")
t("for_loop_vars", "{% for i in ['a','b','c'] %}{{ loop.index }}/{{ loop.index0 }}/{{ loop.revindex }}/{{ loop.revindex0 }}/{{ loop.first }}/{{ loop.last }}/{{ loop.length }} {% endfor %}")
t("for_prev_next", "{% for i in [1,2,3] %}[{{ loop.previtem }}|{{ i }}|{{ loop.nextitem }}]{% endfor %}")
t("for_prev_next_defined", "{% for i in [1,2,3] %}{% if loop.previtem is defined %}p{{ loop.previtem }}{% endif %}{% if loop.nextitem is defined %}n{{ loop.nextitem }}{% endif %};{% endfor %}")
t("for_else", "{% for i in [] %}x{% else %}empty{% endfor %}|{% for i in [1] %}x{% else %}empty{% endfor %}")
t("for_else_undefined", "{% for i in nothing %}x{% else %}none{% endfor %}")
t("for_filter", "{% for i in range(10) if i is even %}{{ i }}{{ loop.length }} {% endfor %}")
t("for_unpack", "{% for k, v in d.items() %}{{ k }}={{ v }};{% endfor %}", {"d": {"a": 1, "b": 2}})
t("for_unpack_nested", "{% for a, (b, c) in [[1, [2, 3]], [4, [5, 6]]] %}{{ a }}{{ b }}{{ c }} {% endfor %}")
t("for_dict_keys", "{% for k in d %}{{ k }}{% endfor %}", {"d": {"x": 1, "y": 2}})
t("for_string", "{% for c in 'abc' %}{{ c }}-{% endfor %}")
t("for_cycle", "{% for i in range(5) %}{{ loop.cycle('a', 'b', 'c') }}{% endfor %}")
t("for_nested_loop_scope", "{% for i in [1,2] %}{% for j in [3,4] %}{{ i }}{{ j }}{{ loop.index }} {% endfor %}{{ loop.index }}|{% endfor %}")
t("for_depth", "{% for i in [1] %}{{ loop.depth }}{% for j in [1] %}{{ loop.depth }}{{ loop.depth0 }}{% endfor %}{% endfor %}")
t("for_reverse_range", "{% for i in range(3, 0, -1) %}{{ i }}{% endfor %}{% for i in range(2, 5) %}{{ i }}{% endfor %}")
t("for_break_continue", "{% for i in range(10) %}{% if i == 2 %}{% continue %}{% endif %}{% if i == 5 %}{% break %}{% endif %}{{ i }}{% endfor %}")
t("for_break_nested", "{% for i in range(3) %}{% for j in range(3) %}{% if j == 1 %}{% break %}{% endif %}{{ i }}{{ j }} {% endfor %}{% endfor %}")
t("for_slice_reverse", "{% for m in ms[::-1] %}{{ m }}{{ loop.index0 }}{% endfor %}", {"ms": ["a", "b", "c"]})
t("set_basic", "{% set x = 1 %}{% set y = x + 1 %}{{ x }}{{ y }}")
t("set_tuple", "{% set a, b = 1, 2 %}{{ a }}{{ b }}{% set c, d = [3, 4] %}{{ c }}{{ d }}")
t("set_list_dict", "{% set l = [1, 2] %}{% set d = {'a': l} %}{{ d.a[1] }}")
t("set_scope_loop", "{% set x = 1 %}{% for i in range(3) %}{% set x = x + 1 %}{% endfor %}{{ x }}")
t("set_scope_if", "{% if true %}{% set x = 5 %}{% endif %}{{ x }}")
t("namespace", "{% set ns = namespace(n=0, s='') %}{% for i in range(4) %}{% set ns.n = ns.n + i %}{% set ns.s = ns.s ~ i %}{% endfor %}{{ ns.n }} {{ ns.s }}")
t("namespace_flag", "{% set ns = namespace(found=false) %}{% for i in [1,2,3] %}{% if i == 2 %}{% set ns.found = true %}{% endif %}{% endfor %}{{ ns.found }}")
t("set_block", "{% set x %}hello {{ 1 + 1 }}{% endset %}[{{ x }}]")
t("set_block_filter", "{% set x | upper %}hello{% endset %}[{{ x }}]")
t("filter_block", "{% filter upper %}hello {{ 'w' }}{% endfilter %}")
t("filter_block_chain", "{% filter trim | upper %}  hi  {% endfilter %}")
t("do_ext", "{% set l = [] %}{% do l.append(1) %}{% do l.append(2) %}{{ l }}", sandbox=False)
t("list_mutation", "{% set l = [1] %}{% set _ = l.append(2) %}{{ l }}{{ l.pop() }}{{ l }}", sandbox=False)
t("dict_mutation", "{% set d = {'a': 1} %}{% set _ = d.update({'b': 2}) %}{{ d }}{{ d.pop('a') }}{{ d }}{{ d.setdefault('c', 3) }}{{ d }}", sandbox=False)

# ---- macros
t("macro_basic", "{% macro m(a, b=2) %}{{ a }}-{{ b }}{% endmacro %}{{ m(1) }} {{ m(1, 3) }} {{ m(b=5, a=0) }}")
t("macro_default_expr", "{% macro m(a, b=a * 2) %}{{ a }}{{ b }}{% endmacro %}{{ m(2) }}")
t("macro_global_access", "{% set g = 'G' %}{% macro m() %}{{ g }}{{ x }}{% endmacro %}{{ m() }}", {"x": "X"})
t("macro_recursive", "{% macro fact(n) %}{% if n <= 1 %}1{% else %}{{ n * fact(n - 1)|int }}{% endif %}{% endmacro %}{{ fact(5) }}")
t("macro_varargs", "{% macro m() %}{{ varargs }}{{ kwargs }}{% endmacro %}{{ m(1, 2, a=3) }}", sandbox=False)
t("macro_caller", "{% macro wrap(x) %}<{{ caller(x) }}>{% endmacro %}{% call(v) wrap(1) %}[{{ v }}]{% endcall %}")
t("macro_no_caller_args", "{% macro wrap() %}<{{ caller() }}>{% endmacro %}{% call wrap() %}body{% endcall %}")
t("macro_in_loop", "{% for i in [1,2] %}{% macro m(x) %}{{ x }}{{ i }}{% endmacro %}{{ m('a') }}{% endfor %}")
t("macro_whitespace", "{%- macro m(x) -%}\n  [{{ x }}]\n{%- endmacro -%}\n{{ m(1) }}|{{ m(2) }}")
t("macro_returns_string", "{% macro m() %}x{% endmacro %}{{ m() ~ m() }}{{ m() is string }}")
t("macro_calls_macro", "{% macro a(x) %}a{{ x }}{% endmacro %}{% macro b(x) %}{{ a(x) }}b{% endmacro %}{{ b(1) }}")
t("macro_args_scope", "{% set x = 'outer' %}{% macro m(x) %}{{ x }}{% endmacro %}{{ m('inner') }}{{ x }}")

# ---- tests
t("tests_basic", "{{ x is defined }} {{ y is defined }} {{ y is undefined }} {{ n is none }} {{ s is string }} {{ d is mapping }} {{ l is iterable }} {{ s is iterable }} {{ l is sequence }} {{ i is number }} {{ i is integer }} {{ f is float }} {{ f is integer }} {{ b is boolean }} {{ b is true }} {{ b is false }} {{ i is not none }}", {"x": 1, "n": None, "s": "s", "d": {}, "l": [], "i": 3, "f": 1.5, "b": True})
t("tests_misc", "{{ 3 is odd }} {{ 4 is even }} {{ 9 is divisibleby 3 }} {{ 9 is divisibleby(2) }} {{ 1 is eq 1 }} {{ 1 is equalto 2 }} {{ 1 is ne 2 }} {{ 1 is lt 2 }} {{ 2 is le 2 }} {{ 3 is gt 2 }} {{ 3 is ge 4 }} {{ 'a' is in ['a'] }} {{ none is sameas none }} {{ true is sameas true }}")
t("tests_mapping_not", "{{ 1 is not mapping }} {{ [] is not iterable }} {{ x is not defined }} {{ d is not string }}", {"d": {}})
t("tests_in_if", "{% if x is defined and x is not none %}ok{% endif %}{% if x is not defined or x is none %}no{% endif %}", {"x": 5})
t("tests_lower_upper", "{{ 'abc' is lower }} {{ 'ABC' is upper }} {{ 'aBc' is lower }}")
t("tests_callable", "{% macro m() %}{% endmacro %}{{ m is callable }} {{ 1 is callable }}")
t("tests_with_filter", "{{ l|length is odd }} {{ s|trim is string }}", {"l": [1, 2, 3], "s": " a "})
t("tests_ternary", "{{ 'y' if x is defined else 'n' }} {{ 'y' if z is defined else 'n' }}", {"x": 1})
t("tests_bool_number", "{{ true is number }} {{ true is integer }} {{ none is number }}")

# ---- filters
t("f_len", "{{ [1,2]|length }} {{ 'abc'|length }} {{ {'a':1}|length }} {{ [1]|count }}")
t("f_case", "{{ 'hello World'|upper }} {{ 'HeLLo'|lower }} {{ 'hELLO wORLD'|capitalize }} {{ 'hello big-world (x) [y]'|title }} {{ \"it's a test\"|title }}")
t("f_trim", "[{{ '  a b  '|trim }}] [{{ 'xxaxx'|trim('x') }}] [{{ '\\n a\\n'|trim }}]")
t("f_first_last", "{{ [1,2,3]|first }} {{ [1,2,3]|last }} {{ 'abc'|first }} {{ 'abc'|last }} [{{ []|first }}]")
t("f_join", "{{ [1,2,3]|join(', ') }} {{ ['a','b']|join }} {{ ms|join('-', attribute='n') }}", {"ms": [{"n": "x"}, {"n": "y"}]})
t("f_default", "{{ u|default('d') }} {{ u|d('e') }} {{ 0|default('z') }} {{ 0|default('z', true) }} {{ ''|default('z', true) }} {{ none|default('z') }} {{ none|default('z', boolean=true) }} {{ u|default }}|")
t("f_replace", "{{ 'aaa'|replace('a', 'b') }} {{ 'aaa'|replace('a', 'b', 2) }} {{ 'abc'|replace('', '-') }} {{ 'x.y'|replace('.', '') }}")
t("f_string", "{{ 1|string }} {{ [1,'a']|string }} {{ none|string }} {{ true|string }} {{ 1.0|string }} {{ {'a': 1}|string }}")
t("f_int_float", "{{ '5'|int }} {{ '5.7'|int }} {{ 'x'|int }} {{ 'x'|int(9) }} {{ 3.9|int }} {{ -3.9|int }} {{ '2.5'|float }} {{ 'z'|float }} {{ 3|float }} {{ true|int }} {{ '  7 '|int }} {{ none|int }}")
t("f_list", "{{ 'abc'|list }} {{ {'a':1}|list }} {{ range(3)|list }}")
t("f_items_keys_values", "{{ d|items|list }} {{ d|dictsort }} {{ d.keys()|list }} {{ d.values()|list }} {{ d.items()|list }}", {"d": {"b": 2, "a": 1}})
t("f_sort", "{{ [3,1,2]|sort }} {{ [3,1,2]|sort(reverse=true) }} {{ ['b','A','c']|sort }} {{ ['b','A','c']|sort(case_sensitive=true) }} {{ ms|sort(attribute='n')|map(attribute='n')|list }} {{ ms|sort(attribute='n', reverse=true)|map(attribute='n')|join }}", {"ms": [{"n": 3}, {"n": 1}, {"n": 2}]})
t("f_unique", "{{ [1,2,1,3,2]|unique|list }} {{ ['a','A','b']|unique|list }} {{ ['a','A','b']|unique(case_sensitive=true)|list }}")
t("f_reverse", "{{ [1,2,3]|reverse|list }} {{ 'abc'|reverse }}")
t("f_sum_min_max", "{{ [1,2,3]|sum }} {{ [1,2,3]|sum(start=10) }} {{ [1.5,2]|sum }} {{ [3,1,2]|min }} {{ [3,1,2]|max }} {{ ['b','a']|min }} {{ ms|sum(attribute='n') }} {{ ms|max(attribute='n') }}", {"ms": [{"n": 3}, {"n": 1}]})
t("f_abs_round", "{{ -3|abs }} {{ -3.5|abs }} {{ 3.14159|round(2) }} {{ 2.5|round }} {{ 3.5|round }} {{ 3.7|round|int }} {{ 3.14159|round(1, 'ceil') }} {{ 3.14159|round(1, 'floor') }} {{ 1234|round(-2) }}")
t("f_indent", "{{ 'a\\nb\\n\\nc'|indent(2) }}|{{ 'a\\nb'|indent(2, true) }}|{{ 'a\\n\\nb'|indent(2, false, true) }}|{{ 'a\\nb'|indent('--') }}")
t("f_safe_escape", "{{ '<a href=\"x\">&\\'</a>'|e }} {{ '<b>'|safe }} {{ '<b>'|escape }}")
t("f_center_wordcount", "[{{ 'ab'|center(7) }}] [{{ 'abc'|center(8) }}] [{{ 'a'|center(2) }}] {{ 'hello big  world_x 1'|wordcount }}")
t("f_batch_slice", "{{ [1,2,3,4,5]|batch(2)|list }} {{ [1,2,3]|batch(2, 0)|list }} {{ [1,2,3,4,5]|slice(2)|list }}")
t("f_map", "{{ [1,2]|map('string')|list }} {{ ['a','b']|map('upper')|join }} {{ ms|map(attribute='n')|list }} {{ ms|map(attribute='z', default='-')|list }} {{ ['a b']|map('replace', ' ', '_')|list }} {{ ms|map(attribute='n')|map('int')|sum }}", {"ms": [{"n": "1"}, {"n": "2"}]})
t("f_select_reject", "{{ range(6)|select('even')|list }} {{ range(6)|reject('even')|list }} {{ [0,1,'',None,'a']|select|list }} {{ [0,1,'']|reject|list }} {{ [1,5,10]|select('gt', 3)|list }} {{ ['a','b']|select('in', ['b'])|list }}")
t("f_selectattr", "{{ ms|selectattr('a')|map(attribute='n')|list }} {{ ms|rejectattr('a')|map(attribute='n')|list }} {{ ms|selectattr('n', 'gt', 1)|map(attribute='n')|list }} {{ ms|selectattr('n', 'equalto', 2)|list }} {{ ms|rejectattr('n', 'in', [1, 2])|map(attribute='n')|list }}", {"ms": [{"n": 1, "a": True}, {"n": 2, "a": False}, {"n": 3, "a": True, "o": {"p": 1}}]})
t("f_selectattr_string", "{{ ms|selectattr('role', 'equalto', 'user')|list|length }}{{ ms|rejectattr('role', 'equalto', 'user')|list|length }}{{ ms|selectattr('role', 'in', ['system', 'user'])|list|length }}", {"ms": [{"role": "system"}, {"role": "user"}, {"role": "assistant"}, {"role": "user"}]})
t("f_attr", "{{ d|attr('a') }} {{ ms|map(attribute='n.m')|list }}", {"d": {"a": 1}, "ms": [{"n": {"m": 5}}]})
t("f_dictsort", "{% for k, v in d|dictsort %}{{ k }}{{ v }}{% endfor %} {% for k, v in d|dictsort(by='value') %}{{ k }}{% endfor %} {% for k, v in d|dictsort(reverse=true) %}{{ k }}{% endfor %}", {"d": {"b": 1, "A": 3, "c": 2}})
t("f_truncate", "{{ 'hello world foo'|truncate(10) }} {{ 'hello world foo bar baz qux'|truncate(12, true) }} {{ 'short'|truncate(10) }} {{ 'hello world foo bar baz qux'|truncate(15, false, '!') }}")
t("f_urlencode", "{{ 'a b&c/d~é'|urlencode }}")
t("f_format", "{{ '%s-%d'|format('a', 3) }} {{ '%05.1f|%x|%5s|%-5s|' | format(3.14159, 255, 'a', 'b') }} {{ '%s' % 'x' }} {{ '%s=%s' % ('a', 1) }} {{ '%r' % 'q' }} {{ '100%%' % () }}")
t("f_groupby", "{% for g, items in ms|groupby('k') %}{{ g }}:{{ items|map(attribute='v')|join }};{% endfor %}", {"ms": [{"k": "a", "v": 1}, {"k": "b", "v": 2}, {"k": "a", "v": 3}]})
t("f_chain", "{{ ' Hello World '|trim|lower|replace(' ', '_')|upper }}")
t("f_filter_on_expr", "{{ (1 + 2)|string ~ 'x' }} {{ -1|abs }} {{ 'a' ~ 'b'|upper }}")
t("f_xmlattr", "<a{{ ' ' ~ {'x': 1, 'y': 'a\"b'}|xmlattr }}>")
t("f_list_of_dict_len", "{{ ms|length }}{{ ms|first|length }}", {"ms": [{"a": 1, "b": 2}]})

# ---- tojson
J = {"s": "héllo \"q\" \\ \n\t  \x1f \x7f é 😀", "n": 3, "f": 1.5, "t": True, "z": None, "l": [1, [2, []], {}], "d": {"a": {"b": [1, 2]}, "c": {}}, "u": "ü", "big": 1e20, "neg": -0.0, "small": 1e-7}
t("tojson_basic", "{{ j|tojson }}", {"j": J})
t("tojson_indent", "{{ j|tojson(indent=2) }}|{{ j|tojson(indent=4) }}", {"j": J})
t("tojson_ascii", "{{ j|tojson(ensure_ascii=true) }}", {"j": J})
t("tojson_sep", "{{ j|tojson(separators=(',', ':')) }}|{{ j|tojson(indent=1, separators=(',', ': ')) }}", {"j": J})
t("tojson_sort", "{{ {'b': 1, 'a': {'z': 1, 'y': 2}}|tojson(sort_keys=true) }}")
t("tojson_scalars", "{{ 'x'|tojson }} {{ 1|tojson }} {{ none|tojson }} {{ true|tojson }} {{ 1.0|tojson }} {{ []|tojson }} {{ {}|tojson }} {{ [1,2]|tojson(indent=2) }} {{ {}|tojson(indent=2) }}")
t("tojson_nonstring_keys", "{{ {1: 'a', 'k': 'b', 2.5: 'c'}|tojson }}")
t("tojson_ensure_positional", "{{ 'é'|tojson(true) }}|{{ [1]|tojson(false, 2) }}")
t("tojson_in_string", "{{ 'x=' ~ ({'a': 1}|tojson) }} {{ {'a': 1}|tojson|safe }} {{ ('<' ~ (['é']|tojson))|trim }}")

t("markup", "{{ 'a'|safe + '<' }}|{{ '<'|e }}|{{ '<'|safe }}|{{ 'x' ~ ('<'|safe) }}|{{ '<' + 'x'|safe }}|{{ ('<'|e)|length }}|{{ '&'|e|e }}|{{ 'q\"'|safe|tojson }}|{{ 'a'|safe is string }}")
t("unicode_codepoints", "{{ s|length }}{{ s[0] }}{{ s[::-1] }}{{ s|reverse }}{{ s[1:] }}{{ s|first }}{{ s|last }}{{ s|list|length }}", {"s": "😀a😀"})
t("filters_on_none", "{{ none|select|list }}{{ none|map(attribute='a')|list }}{{ none|selectattr('a')|list }}{{ []|rejectattr('a')|list }}")

# ---- string methods
t("m_strip", "[{{ '  a '.strip() }}] [{{ '  a '.lstrip() }}] [{{ '  a '.rstrip() }}] [{{ 'xxaxx'.strip('x') }}] [{{ 'xxaxx'.lstrip('x') }}] [{{ 'xxaxx'.rstrip('x') }}] [{{ '\\n\\t a \\n'.strip() }}]")
t("m_split", "{{ 'a b  c'.split() }} {{ 'a,b,,c'.split(',') }} {{ 'a,b,c'.split(',', 1) }} {{ ' a b '.split() }} {{ ''.split(',') }} {{ ''.split() }} {{ 'a--b'.split('--') }} {{ 'a b c d'.split(None, 2) }}")
t("m_rsplit", "{{ 'a,b,c'.rsplit(',', 1) }} {{ 'a b c'.rsplit() }} {{ 'a.b.c.d'.rsplit('.', 2) }}")
t("m_startswith", "{{ 'abc'.startswith('ab') }} {{ 'abc'.startswith('bc') }} {{ 'abc'.endswith('bc') }} {{ 'abc'.startswith(('x', 'a')) }} {{ 'abc'.endswith(('x', 'y')) }} {{ 'abc'.startswith('') }} {{ 'abc'.startswith('b', 1) }}")
t("m_case", "{{ 'ab c'.upper() }} {{ 'AB C'.lower() }} {{ 'ab cD-ef'.title() }} {{ 'hELLO'.capitalize() }} {{ 'aBc'.swapcase() }}")
t("m_replace_find", "{{ 'aXa'.replace('a', 'b') }} {{ 'aaa'.replace('a', 'b', 1) }} {{ 'abcabc'.find('c') }} {{ 'abc'.find('z') }} {{ 'abcabc'.rfind('b') }} {{ 'abcabc'.find('b', 2) }} {{ 'abcabc'.index('c') }} {{ 'abab'.count('ab') }}")
t("m_join_format", "{{ ', '.join(['a', 'b']) }} {{ '-'.join('abc') }} {{ '{}-{}'.format(1, 'a') }} {{ '{0}{1}{0}'.format('x', 'y') }} {{ '{a}:{b}'.format(a=1, b=2) }} {{ '{{x}}'.format() }} {{ '{:>5}|{:<5}|{:^5}|{:05d}|{:.2f}'.format('a', 'b', 'c', 42, 3.14159) }} {{ '{!r}'.format('s') }}")
t("m_is", "{{ '12'.isdigit() }} {{ 'a1'.isalnum() }} {{ 'ab'.isalpha() }} {{ ' '.isspace() }} {{ 'ab'.islower() }} {{ 'AB'.isupper() }} {{ ''.isdigit() }}")
t("m_misc", "{{ '5'.zfill(3) }} {{ '-5'.zfill(4) }} {{ 'a'.ljust(3) }}| {{ 'a'.rjust(3) }}| {{ 'ab'.center(6) }}| {{ 'prefix_x'.removeprefix('prefix_') }} {{ 'x.txt'.removesuffix('.txt') }} {{ 'a=b=c'.partition('=') }} {{ 'a=b=c'.rpartition('=') }} {{ 'a\\nb\\r\\nc'.splitlines() }}")
t("m_chain", "{{ ' a,b '.strip().split(',')|join('+') }} {{ 'abc'.upper().lower().title() }}")
t("m_on_expr", "{{ (a ~ b).upper() }} {{ x.name.lower() }} {{ x['name'].split('-')[1] }}", {"a": "q", "b": "r", "x": {"name": "AB-CD"}})
t("m_dict", "{{ d.get('a') }} [{{ d.get('z') }}] {{ d.get('z', 'dflt') }} {{ d.items()|list }} {{ d.keys()|list }} {{ d.values()|list }} {{ d.get('n').get('m') }}", {"d": {"a": 1, "n": {"m": 2}}})
t("m_list", "{{ l.index(2) }} {{ l.count(1) }} {{ l.copy() }}", {"l": [1, 2, 1]})
t("m_list_mut", "{% set l = [3, 1] %}{% set _ = l.append(2) %}{% set _ = l.extend([7, 8]) %}{% set _ = l.insert(0, 9) %}{% set _ = l.reverse() %}{{ l }}{{ l.pop(0) }}{{ l.pop() }}{{ l }}", sandbox=False)

# ---- globals
t("g_range", "{{ range(3)|list }} {{ range(1, 4)|list }} {{ range(10, 0, -3)|list }} {{ range(0)|list }} {{ range(5)|length }}")
t("g_dict", "{{ dict(a=1, b=2) }} {{ dict([['x', 1]]) }} {{ dict({'q': 1}, r=2) }} {{ dict() }}")
t("g_strftime", "{{ strftime_now('%Y-%m-%d') }} {{ strftime_now('%d %b %Y') }} {{ strftime_now('%A, %B %d, %Y') }} {{ strftime_now('%H:%M:%S %p %I %y %j') }}")
t("g_namespace_print", "{{ namespace(a=1).a }}")
t("g_raise_not_taken", "{% if false %}{{ raise_exception('bad') }}{% endif %}ok")

# ---- whitespace
t("ws_default_trim", "a\n{% if true %}\nb\n{% endif %}\nc\n")
t("ws_default_notrim", "a\n{% if true %}\nb\n{% endif %}\nc\n", trim=False, lstrip=False)
t("ws_lstrip_only", "a\n  {% if true %}\nb\n  {% endif %}\nc", trim=False, lstrip=True)
t("ws_trim_only", "a\n  {% if true %}\nb\n  {% endif %}\nc", trim=True, lstrip=False)
t("ws_minus_both", "a  \n  {%- if true -%}  \n  b  \n  {%- endif -%}  \n  c")
t("ws_minus_var", "a  {{- 'b' -}}  c  {{ 'd' -}}  \n e  {{- 'f' }}  g")
t("ws_plus", "a\n  {%+ if true %}\nb{% endif %}", trim=False, lstrip=True)
t("ws_comment", "a\n  {# comment #}\nb\n  {#- c -#}  \nd{# x #}e")
t("ws_comment_multiline", "a{# line1\nline2 #}b\n{# c #}\nd")
t("ws_var_not_stripped", "a\n  {{ 'x' }}\n  {{ 'y' }}\nb")
t("ws_text_before_tag_same_line", "x {% if true %}y{% endif %}\n  z {% if true %}\nw{% endif %}")
t("ws_trailing_newline", "abc\n")
t("ws_trailing_newline_keep", "abc\n", keep_nl=True)
t("ws_trailing_two_newlines", "abc\n\n")
t("ws_tabs", "a\n\t{% if true %}\n\tb\n\t{% endif %}\nc")
t("ws_crlf", "a\r\n{% if true %}\r\nb\r\n{% endif %}\r\nc")
t("ws_for_lines", "{% for i in [1,2,3] %}\n{{ i }}\n{% endfor %}\nend")
t("ws_for_indented", "<\n    {% for i in [1,2] %}\n    item {{ i }}\n    {% endfor %}\n>")
t("ws_nested_indented", "{% for i in [1,2] %}\n  {% if i == 1 %}\n  one\n  {% else %}\n  other\n  {% endif %}\n{% endfor %}")
t("ws_set_lines", "{% set x = 1 %}\n{% set y = 2 %}\n{{ x }}{{ y }}\n")
t("ws_first_line_block", "  {% if true %}x{% endif %}")
t("ws_empty_block_lines", "a\n\n{% if true %}\n\nb\n{% endif %}\n\nc")
t("ws_minus_newline_before_endfor", "{% for i in [1,2] -%}\n  {{ i }}\n{%- endfor %}\nx")
t("ws_inline_if", "{{ 'a' }}{% if true %} b{% endif %} c")

# ---- raw
t("raw_basic", "{% raw %}{{ x }} {% if %}{% endraw %}after")
t("raw_ws", "a\n  {%- raw -%}\n  {{ x }}\n  {%- endraw -%}\n  b")
t("raw_multi", "{% raw %}\n{{ a }}\n{% endraw %}\n{% raw %}{# c #}{% endraw %}")

# ---- comments / misc syntax
t("comment_in_block", "{# {% if %} #}a{# #}b")
t("dict_braces_close", "{{ {'a': {'b': 1}} }}|{{ {'a': 1}.a }}{{ {} }}")
t("string_with_delims", "{{ '}}' }}{{ '%}' }}{{ '{#' }}{% set x = '%}' %}{{ x }}")
t("block_in_expr_strings", "{% if 'a}}b' == 'a}}b' %}y{% endif %}")
t("multiline_expr", "{{ [\n 1,\n 2\n] }}{% set x = {\n 'a': 1,\n} %}{{ x }}")
t("trailing_commas", "{{ [1, 2,] }} {{ (1,) }} {{ {'a': 1,} }} {{ range(3,)|list }}")
t("kwargs_call", "{{ dict(a=1)|tojson }}")
t("unicode_text", "héllo {{ 'wörld' }} 😀 {{ '日本語'|length }} {{ '😀'|length }}")
t("name_with_digits_underscore", "{% set _a1 = 1 %}{% set b_2 = _a1 + 1 %}{{ b_2 }}")
t("loop_var_after", "{% for i in [1] %}{% endfor %}[{{ loop }}][{{ i }}]")
t("if_in_for_set", "{% set ns = namespace(out=[]) %}{% for m in ms %}{% if m.r == 'u' %}{% set ns.out = ns.out + [m.c] %}{% endif %}{% endfor %}{{ ns.out }}", {"ms": [{"r": "u", "c": "a"}, {"r": "s", "c": "b"}, {"r": "u", "c": "c"}]})
t("complex_template", "{%- for m in ms -%}{{ '<' ~ m.r ~ '>' }}{{ m.c | trim }}{{ '</' ~ m.r ~ '>' }}{% if not loop.last %}\n{% endif %}{%- endfor -%}", {"ms": [{"r": "u", "c": " a "}, {"r": "s", "c": "b"}]})
t("messages_first", "{{ messages[0].role }}{{ messages[0]['content'] }}{{ messages|length }}{{ messages[1:]|length }}{{ messages[-1].role }}", {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]})
t("not_in_dict_list", "{% if 'a' not in ['b'] and 'k' not in {'z': 1} %}ok{% endif %}")
t("in_string_undefined", "{{ 'a' in s }}", {"s": "bab"})
t("cond_nested_parens", "{{ ('a' if x else 'b') ~ ('c' if not x else 'd') }}", {"x": 0})
t("is_in_filter_args", "{{ [1, 2, 3]|select('odd')|list }} {{ [1,2,3]|rejectattr('x', 'defined')|list|length if false else 'k' }}")
t("loop_in_macro_call", "{% macro m(i) %}{{ i }}{% endmacro %}{% for i in range(3) %}{{ m(i) }}{{ loop.index }}{% endfor %}")
t("int_float_division_formats", "{{ 6 / 3 }} {{ 7 / 2 }} {{ (6 / 3)|int }} {{ 2.0 * 3 }} {{ 10 // 3.0 }} {{ 1e3 }} {{ 5 % 3.0 }}")
t("string_mul_zero", "[{{ 'a' * 0 }}] [{{ 'a' * -1 }}]")
t("nested_data", "{% for u in users %}{{ u.name }}:{% for r in u.roles %}{{ r }}{% if not loop.last %},{% endif %}{% endfor %};{% endfor %}", {"users": [{"name": "a", "roles": ["x", "y"]}, {"name": "b", "roles": []}]})
t("loop_changed", "{% for i in [1,1,2,2,3] %}{{ loop.changed(i) }} {% endfor %}")

# ---- errors (expected to raise)
E = []
def e(name, template, ctx=None, **opts):
    E.append((name, template, ctx or {}, opts))
e("err_undefined_call", "{{ nope() }}")
e("err_add_str_int", "{{ 'a' + 1 }}")
e("err_unknown_filter", "{{ 1|nofilter }}")
e("err_unknown_tag", "{% frobnicate %}")
e("err_unclosed_if", "{% if true %}x")
e("err_unclosed_var", "{{ 1 ")
e("err_unclosed_string", "{{ 'a }}")
e("err_unexpected_endfor", "{% endfor %}")
e("err_raise", "{{ raise_exception('custom message') }}")
e("err_compare_types", "{{ 1 < 'a' }}")
e("err_for_none", "{% for i in none %}{% endfor %}")
e("err_division_zero", "{{ 1 / 0 }}")
e("err_unpack", "{% for a, b in [[1]] %}{% endfor %}")
e("err_set_attr_non_ns", "{% set d = {} %}{% set d.x = 1 %}")
e("err_bad_syntax", "{{ 1 + }}")
e("err_missing_endblock_in_macro", "{% macro m() %}")
e("err_call_nonmacro", "{{ 'abc'() }}")
e("err_int_sub_str", "{{ 1 - 'a' }}")
e("err_unknown_test", "{{ 1 is nosuchtest }}")
e("err_elif_after_else", "{% if 1 %}{% else %}{% elif 2 %}{% endif %}")

def main(out):
    rows = []
    for name, tpl, ctx, opts in S:
        if name is None: continue
        try:
            res = render(tpl, ctx, **opts)
            rows.append({"name": name, "template": tpl, "context": ctx, "options": opts, "output": res})
        except Exception as ex:
            print("PYTHON ERROR in positive snippet", name, type(ex).__name__, ex, file=sys.stderr)
    for name, tpl, ctx, opts in E:
        try:
            res = render(tpl, ctx, **opts)
            print("NO ERROR in error snippet", name, repr(res), file=sys.stderr)
        except Exception as ex:
            rows.append({"name": name, "template": tpl, "context": ctx, "options": opts, "error": str(ex) if name == "err_raise" else type(ex).__name__})
    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(rows), "snippets")

main(sys.argv[1])
