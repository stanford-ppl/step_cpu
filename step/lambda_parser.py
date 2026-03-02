"""Lambda AST -> C++ expression string translation."""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any

# torch function name -> C++ function name
_TORCH_FUNC_MAP: dict[str, str] = {
    "pow": "torch::pow",
    "tanh": "torch::tanh",
    "exp": "torch::exp",
    "sqrt": "torch::sqrt",
    "abs": "torch::abs",
    "log": "torch::log",
    "sigmoid": "torch::sigmoid",
    "relu": "torch::relu",
    "clamp": "torch::clamp",
    "mm": "torch::mm",
}

# Python operator AST node -> C++ operator string
_BINOP_MAP: dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

_UNARYOP_MAP: dict[type, str] = {
    ast.USub: "-",
    ast.UAdd: "+",
}


class LambdaTranslator(ast.NodeVisitor):
    """Translates a Python lambda AST body into a C++ expression string.

    Parameters:
        func_params: names of the lambda parameters (e.g., ('a',) or ('a', 'b'))
        closures: mapping of closure variable names to their captured values
        param_to_cpp: optional mapping from lambda param names to C++ variable names
    """

    def __init__(
        self,
        func_params: tuple[str, ...],
        closures: dict[str, Any],
        param_to_cpp: dict[str, str] | None = None,
    ):
        self.func_params = func_params
        self.closures = closures
        self.param_to_cpp = param_to_cpp or {p: p for p in func_params}

    def translate(self, node: ast.expr) -> str:
        """Translate an AST expression node to a C++ expression string."""
        return self.visit(node)

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        # Lambda parameter -> C++ variable name
        if name in self.func_params:
            return self.param_to_cpp[name]
        # Closure variable -> inline the value
        if name in self.closures:
            return _format_constant(self.closures[name])
        raise ValueError(f"Unknown variable in lambda: {name!r}")

    def visit_Constant(self, node: ast.Constant) -> str:
        return _format_constant(node.value)

    def visit_Num(self, node: ast.Num) -> str:
        return _format_constant(node.n)

    def visit_BinOp(self, node: ast.BinOp) -> str:
        op_str = _BINOP_MAP.get(type(node.op))
        if op_str is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({left} {op_str} {right})"

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        op_str = _UNARYOP_MAP.get(type(node.op))
        if op_str is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = self.visit(node.operand)
        return f"({op_str}{operand})"

    def visit_Attribute(self, node: ast.Attribute) -> str:
        obj = self.visit(node.value)
        if node.attr == "T":
            return f"{obj}.t()"
        raise ValueError(f"Unsupported attribute access: .{node.attr}")

    def visit_Call(self, node: ast.Call) -> str:
        func_name = _extract_func_name(node.func)
        if func_name is None:
            raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")

        # Handle torch.func_name(...)
        cpp_func = _TORCH_FUNC_MAP.get(func_name)
        if cpp_func is None:
            raise ValueError(f"Unsupported torch function: {func_name!r}")

        args = [self.visit(arg) for arg in node.args]
        return f"{cpp_func}({', '.join(args)})"

    def generic_visit(self, node: ast.AST) -> str:
        raise ValueError(f"Unsupported AST node in lambda: {type(node).__name__}: {ast.dump(node)}")


def _extract_func_name(node: ast.expr) -> str | None:
    """Extract the function name from a call target like `torch.tanh`."""
    if isinstance(node, ast.Attribute):
        # torch.tanh -> "tanh"
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def _format_constant(value: Any) -> str:
    """Format a Python constant as a C++ literal."""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise ValueError(f"Unsupported constant type: {type(value).__name__}: {value!r}")


def parse_lambda(func, param_to_cpp: dict[str, str] | None = None) -> tuple[ast.expr, tuple[str, ...], dict[str, Any]]:
    """Parse a Python lambda and extract its AST body, parameter names, and closures.

    Returns:
        (body_ast, param_names, closures)
    """
    source = inspect.getsource(func)
    source = textwrap.dedent(source).strip()

    # The lambda may be embedded in a larger expression (e.g., UnaryMap(x, lambda a: ...))
    # or in an assignment (func = lambda a: ...).  Always try exec mode first since
    # it handles both statements and expressions.
    # When a lambda spans continuation lines, inspect.getsource may return only
    # the lambda line with trailing unmatched parentheses — progressively strip them.
    tree = None
    for _ in range(10):
        try:
            tree = ast.parse(source, mode="exec")
            break
        except SyntaxError:
            pass
        try:
            tree = ast.parse(source, mode="eval")
            break
        except SyntaxError:
            pass
        # Strip one trailing paren/comma and retry
        stripped = source.rstrip()
        if stripped and stripped[-1] in ")],":
            source = stripped[:-1]
        else:
            break
    if tree is None:
        tree = ast.parse(source, mode="exec")
    lambda_node = _find_lambda(tree)
    if lambda_node is None:
        raise ValueError(f"Could not find lambda in source: {source!r}")

    param_names = tuple(arg.arg for arg in lambda_node.args.args)

    # Extract closure values
    closures: dict[str, Any] = {}
    if func.__closure__ is not None:
        freevars = func.__code__.co_freevars
        for name, cell in zip(freevars, func.__closure__):
            closures[name] = cell.cell_contents

    return lambda_node.body, param_names, closures


def _find_lambda(node: ast.AST) -> ast.Lambda | None:
    """Find the first Lambda node in an AST tree."""
    if isinstance(node, ast.Lambda):
        return node
    for child in ast.iter_child_nodes(node):
        result = _find_lambda(child)
        if result is not None:
            return result
    return None


def translate_lambda(
    func,
    param_to_cpp: dict[str, str] | None = None,
) -> tuple[str, tuple[str, ...], dict[str, Any], ast.expr]:
    """Parse a lambda and translate it to a C++ expression.

    Returns:
        (cpp_expr, param_names, closures, body_ast)
    """
    body_ast, param_names, closures = parse_lambda(func, param_to_cpp)
    translator = LambdaTranslator(param_names, closures, param_to_cpp)
    cpp_expr = translator.translate(body_ast)
    return cpp_expr, param_names, closures, body_ast
