"""Tests for --replace module substitution in causal_language_modeling.py.

Run from darpa/modified/:
    pytest test_clm_replace.py -v
"""
import math
import sys
import pathlib

# Ensure mocha/ root is on sys.path so 'step' package is importable
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

import causal_language_modeling as clm
from causal_language_modeling import (
    GPT2MLPStepWrapper,
    _REPLACEMENT_REGISTRY,
    _build_gpt2mlp_replacement,
    _build_gpt2mlp_fused,
    apply_replacements,
)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (compile / load once per test session)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compiled_kernel():
    """Compile the STeP GPT2MLP kernel once for the whole session."""
    return _build_gpt2mlp_replacement()


@pytest.fixture(scope="module")
def distilgpt2_model():
    """Load distilgpt2 once for the whole session (CPU, eval mode)."""
    model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# TestGPT2MLPStepWrapper — unit tests (fast)
# ---------------------------------------------------------------------------

class TestGPT2MLPStepWrapper:
    @pytest.fixture(autouse=True)
    def _setup(self, distilgpt2_model, compiled_kernel):
        self.model = distilgpt2_model
        self.kernel = compiled_kernel
        # Use block 0's MLP as a representative module
        self.original_mlp = self.model.transformer.h[0].mlp
        self.wrapper = GPT2MLPStepWrapper(self.original_mlp, self.kernel)
        self.wrapper.eval()

    def _make_input(self, batch=2, seq=10, hidden=768):
        torch.manual_seed(0)
        return torch.randn(batch, seq, hidden)

    def test_output_shape_preserved(self):
        x = self._make_input()
        with torch.no_grad():
            out = self.wrapper(x)
        assert out.shape == x.shape

    def test_output_numerically_close_to_original(self):
        x = self._make_input()
        with torch.no_grad():
            out_wrapper = self.wrapper(x)
            out_original = self.original_mlp(x)
        torch.testing.assert_close(out_wrapper, out_original, atol=1e-4, rtol=1e-3)

    def test_training_mode_uses_original_path(self):
        self.wrapper.train()
        assert self.wrapper.training is True
        # Verify the output has the right shape in training mode (dropout makes
        # exact numeric comparison unreliable, so we only check shape here)
        x = self._make_input()
        with torch.no_grad():
            out_train = self.wrapper(x)
        assert out_train.shape == x.shape
        # Reset to eval
        self.wrapper.eval()

    def test_submodules_registered(self):
        child_names = dict(self.wrapper.named_children())
        assert "c_fc" in child_names
        assert "c_proj" in child_names

    def test_various_sequence_lengths(self):
        for seq_len in (1, 5, 128):
            x = torch.randn(1, seq_len, 768)
            with torch.no_grad():
                out = self.wrapper(x)
            assert out.shape == (1, seq_len, 768), f"Shape mismatch for seq_len={seq_len}"

    def test_batch_size_gt_1(self):
        x = torch.randn(3, 10, 768)
        with torch.no_grad():
            out = self.wrapper(x)
        assert out.shape == (3, 10, 768)


# ---------------------------------------------------------------------------
# TestReplacementRegistry — unit tests (fast)
# ---------------------------------------------------------------------------

class TestReplacementRegistry:
    def test_unknown_replacement_raises(self, distilgpt2_model):
        with pytest.raises(ValueError, match="Unknown replacement"):
            apply_replacements(distilgpt2_model, ["not_a_real_module"])

    def test_empty_replace_list_is_noop(self, distilgpt2_model):
        # Collect original MLP types before
        before = [type(b.mlp).__name__ for b in distilgpt2_model.transformer.h]
        apply_replacements(distilgpt2_model, [])
        after = [type(b.mlp).__name__ for b in distilgpt2_model.transformer.h]
        assert before == after

    def test_gpt2mlp_replaces_all_blocks(self, distilgpt2_model, compiled_kernel):
        # Load a fresh model so we don't mutate the shared fixture permanently
        model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
        model.eval()
        apply_replacements(model, ["gpt2mlp"])
        for i, block in enumerate(model.transformer.h):
            assert isinstance(block.mlp, GPT2MLPStepWrapper), \
                f"Block {i} MLP was not replaced"

    def test_registry_contains_gpt2mlp(self):
        assert "gpt2mlp" in _REPLACEMENT_REGISTRY


# ---------------------------------------------------------------------------
# TestInferenceIntegration — end-to-end (slower)
# ---------------------------------------------------------------------------

class TestInferenceIntegration:
    PROMPT = "Why is the sky blue?"

    def _run_infer(self, extra_args=None):
        args = ["--mode", "infer", "--prompt", self.PROMPT, "--cpu-only", "--no-instrument"]
        if extra_args:
            args += extra_args
        return clm.main(args)

    def test_infer_without_replace_returns_result(self):
        result = self._run_infer()
        assert result is not None

    def test_infer_with_gpt2mlp_replace_runs(self):
        result = self._run_infer(["--replace", "gpt2mlp"])
        assert result is not None

    def test_timing_comparison(self, capsys):
        """Informational: compare timing with and without replacement.

        This test always passes — it just prints the speedup.
        """
        import time

        t0 = time.perf_counter()
        self._run_infer()
        baseline = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._run_infer(["--replace", "gpt2mlp"])
        replaced = time.perf_counter() - t0

        speedup = baseline / replaced if replaced > 0 else float("inf")
        print(f"\n[timing] baseline={baseline:.2f}s  replaced={replaced:.2f}s  speedup={speedup:.2f}x")
        # Always passes — informational only
        assert True


# ---------------------------------------------------------------------------
# TestFusedCodegen — verify the codegen produces optimized (fused) C++
# ---------------------------------------------------------------------------

class TestFusedCodegen:
    """Test that the STeP code generator emits fused C++ for the GPT2MLP pattern."""

    @pytest.fixture(scope="class")
    def generated_cpp(self):
        """Generate C++ source from the GPT2MLP STeP definition."""
        import inspect
        import step
        from step import TensorProxy, StepGraph, reset_id_counter, _trace_ctx
        from step.codegen import CppCodegen

        def gpt2_mlp_step(hidden_states, W_fc, b_fc, W_proj, b_proj):
            M = hidden_states.shape[0]
            K_tile = 256
            N_tile = 256

            act_fc = step.tensor_to_stream(hidden_states, vec=[M, K_tile])
            act_fc = step.Flatten(act_fc, min_rank=0, max_rank=1)
            act_fc_buf = step.Bufferize(act_fc, rank=1)
            act_fc_rep = step.Streamify(act_fc_buf, repeat_factor=[12], rank=1)

            W_fc_T = W_fc.T.contiguous()
            wfc_stream = step.tensor_to_stream(W_fc_T, vec=[N_tile, K_tile])

            fc_partial = step.BinaryMap(
                act_fc_rep, wfc_stream, lambda a, w: torch.mm(a, w.T)
            )
            fc_accum = step.Accum(fc_partial, rank=1)

            bfc_stream = step.tensor_to_stream(b_fc.unsqueeze(0), vec=[1, N_tile])
            bfc_stream = step.Flatten(bfc_stream, min_rank=0, max_rank=1)

            fc_biased = step.BinaryMap(fc_accum, bfc_stream, lambda a, b: a + b)

            _math = math
            c_sqrt = _math.sqrt(2.0 / _math.pi)
            c_pow = 0.044715

            x3 = step.UnaryMap(fc_biased, lambda a: torch.pow(a, 3.0))
            x_in = step.BinaryMap(fc_biased, x3, lambda a, b: a + c_pow * b)
            t_in = step.UnaryMap(x_in, lambda a: c_sqrt * a)
            t = step.UnaryMap(t_in, lambda a: torch.tanh(a))
            onep = step.UnaryMap(t, lambda a: 1.0 + a)
            gelu_out = step.BinaryMap(fc_biased, onep, lambda a, b: 0.5 * a * b)

            proj_buf = step.Bufferize(gelu_out, rank=1)
            proj_act_rep = step.Streamify(proj_buf, repeat_factor=[3], rank=1)

            W_proj_T = W_proj.T.contiguous()
            wproj_stream = step.tensor_to_stream(W_proj_T, vec=[N_tile, K_tile])

            proj_partial = step.BinaryMap(
                proj_act_rep, wproj_stream, lambda a, w: torch.mm(a, w.T)
            )
            proj_accum = step.Accum(proj_partial, rank=1)

            bproj_stream = step.tensor_to_stream(b_proj.unsqueeze(0), vec=[1, N_tile])
            bproj_stream = step.Flatten(bproj_stream, min_rank=0, max_rank=1)

            proj_biased = step.BinaryMap(proj_accum, bproj_stream, lambda a, b: a + b)

            output = step.stream_to_tensor(proj_biased, like_tensor=hidden_states)
            return output

        gpt2_mlp_step._param_ndims = [2, 2, 1, 2, 1]

        # Trace the function to get the IR graph
        sig = inspect.signature(gpt2_mlp_step)
        tensor_params = list(sig.parameters.keys())
        ndim_hints = gpt2_mlp_step._param_ndims

        reset_id_counter()
        graph = StepGraph(func_name=gpt2_mlp_step.__name__, tensor_params=tensor_params)
        _trace_ctx.graph = graph
        _trace_ctx.name_counter = {}

        try:
            proxies = [TensorProxy(name=p, ndim=ndim_hints[i])
                       for i, p in enumerate(tensor_params)]
            gpt2_mlp_step(*proxies)
        finally:
            _trace_ctx.graph = None
            _trace_ctx.name_counter = None

        codegen = CppCodegen(graph)
        return codegen.generate()

    def test_generated_cpp_has_no_transpose(self, generated_cpp):
        """Fused C++ should NOT contain .t().contiguous() transforms."""
        assert ".t().contiguous()" not in generated_cpp

    def test_generated_cpp_has_fused_gelu(self, generated_cpp):
        """Fused C++ should contain at::gelu instead of manual GELU chain."""
        assert "at::gelu" in generated_cpp

    def test_generated_cpp_has_no_loop(self, generated_cpp):
        """Fused C++ should NOT contain tiled for-loops."""
        assert "for (int64_t" not in generated_cpp

    def test_generated_cpp_has_torch_mm(self, generated_cpp):
        """Fused C++ should use torch::mm for matmul."""
        assert "torch::mm" in generated_cpp

    def test_fused_kernel_correctness(self):
        """Verify the fused kernel output matches PyTorch native across M values."""
        from transformers import AutoModelForCausalLM as AMLM

        model = AMLM.from_pretrained("distilbert/distilgpt2")
        model.eval()
        mlp = model.transformer.h[0].mlp

        W_fc = mlp.c_fc.weight.detach()
        b_fc = mlp.c_fc.bias.detach()
        W_proj = mlp.c_proj.weight.detach()
        b_proj = mlp.c_proj.bias.detach()

        compiled = _build_gpt2mlp_replacement()

        for M in [1, 5, 20, 128]:
            torch.manual_seed(42)
            x = torch.randn(M, 768)

            # PyTorch reference (Conv1D forward: x @ W + b)
            ref = torch.mm(x, W_fc) + b_fc
            ref = F.gelu(ref, approximate="tanh")
            ref = torch.mm(ref, W_proj) + b_proj

            # STeP compiled kernel
            out = compiled(x, W_fc, b_fc, W_proj, b_proj)

            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3,
                                       msg=f"Mismatch at M={M}")


# ---------------------------------------------------------------------------
# TestFusedReplacement — test the hand-written fused kernel via --replace
# ---------------------------------------------------------------------------

class TestFusedReplacement:
    def test_registry_contains_gpt2mlp_fused(self):
        assert "gpt2mlp_fused" in _REPLACEMENT_REGISTRY

    def test_gpt2mlp_fused_replacement(self):
        """Test the hand-written fused kernel via --replace gpt2mlp_fused."""
        result = clm.main([
            "--mode", "infer",
            "--prompt", "Why is the sky blue?",
            "--cpu-only",
            "--no-instrument",
            "--replace", "gpt2mlp_fused",
        ])
        assert result is not None

    def test_fused_kernel_matches_pytorch(self):
        """Verify hand-written fused kernel matches PyTorch native MLP."""
        from transformers import AutoModelForCausalLM as AMLM

        model = AMLM.from_pretrained("distilbert/distilgpt2")
        model.eval()
        mlp = model.transformer.h[0].mlp

        W_fc = mlp.c_fc.weight.detach()
        b_fc = mlp.c_fc.bias.detach()
        W_proj = mlp.c_proj.weight.detach()
        b_proj = mlp.c_proj.bias.detach()

        fused = _build_gpt2mlp_fused()

        for M in [1, 5, 20]:
            torch.manual_seed(42)
            x = torch.randn(M, 768)

            ref = torch.mm(x, W_fc) + b_fc
            ref = F.gelu(ref, approximate="tanh")
            ref = torch.mm(ref, W_proj) + b_proj

            out = fused(x, W_fc, b_fc, W_proj, b_proj)
            torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-3,
                                       msg=f"Fused mismatch at M={M}")
