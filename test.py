from scipy.integrate import quad
import pytest
from pytest import approx

def quad_linear(x, beg, end):
    """
    Integrates a given linear function over the specified range.
    
    Args:
        x: The linear function to integrate.
        beg: The start of the integration range.
        end: The end of the integration range.
    
    Returns:
        The result of the integration.
    """
    answer, _ = quad(x, beg, end)
    return answer

def test_should_return_quad_linear_function():
    def linear(x):
        return 2 * x + 3
    
    answer = quad_linear(linear, 0, 10)
    test, _ = quad(linear, 0, 10)

    assert answer == pytest.approx(test, rel = 1e-5)

def 