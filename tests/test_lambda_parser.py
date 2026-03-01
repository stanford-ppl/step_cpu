"""Unit tests for lambda -> C++ translation."""

import ast
import math

import torch

from step.lambda_parser import LambdaTranslator, parse_lambda, translate_lambda


class TestLambdaTranslator:
    """Tests for individual AST node translations."""

    def test_arithmetic_add(self):
        func = lambda a, b: a + b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert params == ("a", "b")
        assert cpp == "(a + b)"

    def test_arithmetic_mul(self):
        func = lambda a, b: a * b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(a * b)"

    def test_arithmetic_sub(self):
        func = lambda a, b: a - b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(a - b)"

    def test_arithmetic_div(self):
        func = lambda a, b: a / b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(a / b)"

    def test_constant_mul(self):
        func = lambda a: 0.5 * a  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(0.5 * a)"

    def test_constant_add(self):
        func = lambda a: 1.0 + a  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(1.0 + a)"

    def test_torch_pow(self):
        func = lambda a: torch.pow(a, 3.0)  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "torch::pow(a, 3.0)"

    def test_torch_tanh(self):
        func = lambda a: torch.tanh(a)  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "torch::tanh(a)"

    def test_torch_exp(self):
        func = lambda a: torch.exp(a)  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "torch::exp(a)"

    def test_torch_sqrt(self):
        func = lambda a: torch.sqrt(a)  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "torch::sqrt(a)"

    def test_closure_variable(self):
        c_sqrt = math.sqrt(2.0 / math.pi)
        func = lambda a: c_sqrt * a  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert "0.7978845608028654" in cpp
        assert "* a" in cpp
        assert closures == {"c_sqrt": c_sqrt}

    def test_complex_expression(self):
        c_pow = 0.044715
        func = lambda a, b: a + c_pow * b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert "0.044715" in cpp
        assert params == ("a", "b")

    def test_param_to_cpp_mapping(self):
        func = lambda a: torch.tanh(a)  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func, param_to_cpp={"a": "t_in"})
        assert cpp == "torch::tanh(t_in)"

    def test_unary_neg(self):
        func = lambda a: -a  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert cpp == "(-a)"

    def test_nested_expression(self):
        func = lambda a, b: 0.5 * a * b  # noqa: E731
        cpp, params, closures, _ = translate_lambda(func)
        assert "0.5" in cpp
        assert "a" in cpp
        assert "b" in cpp


class TestParseLambda:
    """Tests for lambda parsing and closure extraction."""

    def test_simple_lambda(self):
        func = lambda x: x  # noqa: E731
        body, params, closures = parse_lambda(func)
        assert params == ("x",)
        assert closures == {}
        assert isinstance(body, ast.Name)

    def test_closure_extraction(self):
        val = 42.0
        func = lambda x: val * x  # noqa: E731
        body, params, closures = parse_lambda(func)
        assert closures == {"val": 42.0}

    def test_multi_closure(self):
        a = 1.0
        b = 2.0
        func = lambda x: a * x + b  # noqa: E731
        body, params, closures = parse_lambda(func)
        assert closures == {"a": 1.0, "b": 2.0}
