from enum import Enum

###############################################################
############################ types ############################
###############################################################


class DType(Enum):
    F32 = 0
    F64 = 1

    def __str__(self) -> str:
        return "f32" if self == DType.F32 else "f64"


###############################################################
########################## functions ##########################
###############################################################


def funcname(name, mi, ni):
    return f"{name}_{mi}_{ni}"


def watermarked(code):
    return f"{HEAD}\n{code.replace(HEAD, '')}"


def func_template(
    name,
    param_list,
    value_type: DType,
    code,
    inline: bool = True,
    const: bool = False,
):
    code = f"""

{"#[inline]" if inline else ""}
pub {"const" if const else ""} fn {name}({param_list}) -> {str(value_type)} {{

    {code}

}}
"""
    return watermarked(code)


def is_valid_rust_ident(name: str) -> bool:
    # 检查是否是 Rust 关键字
    if name in RUST_KEYWORDS:
        return False

    # 检查是否以字母或下划线开头
    if not (name[0].isalpha() or name[0] == "_"):
        return False

    # 检查是否包含非法字符
    for char in name[1:]:
        if not (char.isalnum() or char == "_"):
            return False

    return True


def assert_name(name: str):
    assert is_valid_rust_ident(
        name
    ), f"Invalid function name! Expects a valid Rust identifier, found `{name}`"


###############################################################
########################## constants ##########################
###############################################################

HEAD = """
/*

* Code generated by Symars. Thank you for using Symars!
  Symars is licensed under MIT licnese.
  Repository: https://github.com/Da1sypetals/Symars

* Computation code is not intended for manual editing.

* If you find an error,
  or if you believe Symars generates incorrect result, 
  please raise an issue under our repo with minimal reproducible example.

*/
"""


RUST_KEYWORDS = {
    "abstract",
    "alignof",
    "as",
    "become",
    "box",
    "break",
    "const",
    "continue",
    "crate",
    "do",
    "else",
    "enum",
    "extern",
    "false",
    "final",
    "fn",
    "for",
    "if",
    "impl",
    "in",
    "let",
    "loop",
    "macro",
    "match",
    "mod",
    "move",
    "mut",
    "offsetof",
    "override",
    "priv",
    "proc",
    "pub",
    "pure",
    "ref",
    "return",
    "Self",
    "self",
    "sizeof",
    "static",
    "struct",
    "super",
    "trait",
    "true",
    "type",
    "typeof",
    "unsafe",
    "unsized",
    "use",
    "virtual",
    "where",
    "while",
    "yield",
}