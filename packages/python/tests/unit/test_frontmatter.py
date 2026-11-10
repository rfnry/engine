import yaml

from rfnry_engine import frontmatter

SAMPLE = """
yes_: on
no_: no
off_: Off
y: y
t: true
f: False
upper: TRUE
float_: 1.0
int_: 12
octal: 0o14
hex_: 0x1f
sci: 1e3
nothing: null
tilde: ~
date_: 2024-01-02
stamp: 2024-01-02 03:04:05
text: "on"
quoted: 'no'
version: 1.2.3
list_: [on, off, yes, 1.5, a]
nested: {k: no, n: [true, false]}
multi: |
  line one
  line two
folded: >
  a
  b
"""


def python_loader() -> type[yaml.SafeLoader]:
    class Loader(yaml.SafeLoader):
        pass

    Loader.yaml_implicit_resolvers = dict(frontmatter.StrictLoader.yaml_implicit_resolvers)
    return Loader


def test_the_loader_parses_the_same_with_libyaml_and_without() -> None:
    expected = yaml.load(SAMPLE, Loader=python_loader())
    assert yaml.load(SAMPLE, Loader=frontmatter.StrictLoader) == expected
    assert expected["yes_"] == "on" and expected["no_"] == "no"
    assert expected["t"] is True and expected["f"] is False and expected["upper"] is True
    assert expected["float_"] == 1.0 and expected["int_"] == 12


def test_the_loader_is_libyaml_when_pyyaml_has_it() -> None:
    assert issubclass(frontmatter.StrictLoader, getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    assert not issubclass(python_loader(), getattr(yaml, "CSafeLoader", ()))


def test_a_crlf_body_reads_the_same_with_or_without_frontmatter() -> None:
    assert frontmatter.body_of("a\r\nb\r\n") == "a\nb"
    assert frontmatter.body_of("\ufeffa\r\nb\r\n") == "a\nb"
    assert frontmatter.body_of("---\r\nx: 1\r\n---\r\na\r\nb\r\n") == "a\nb"
