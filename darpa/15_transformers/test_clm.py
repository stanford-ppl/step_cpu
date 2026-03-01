import pytest
import causal_language_modeling

def test_small_run():
    benchmark_result = causal_language_modeling.main(['--samples', '256'])
    assert benchmark_result.perplexity < 50.0
    assert benchmark_result.elapsed_time > 5.0
