from scipy.integrate import quad
import pytest
from pytest import approx



def test_should_return_quad_linear_function():
    def linear(x):
        return 2 * x + 3
    
    answer = quad_linear(linear, 0, 10)
    test, _ = quad(linear, 0, 10)

    assert answer == pytest.approx(test, rel = 1e-5)