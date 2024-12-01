from scipy.integrate import quad
from scipy.linalg import solve
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

def system_equation(coeffs, conts):
    answer = solve(coeffs, conts)
    return {"X": answer[0], "Y": answer[1]}

def test_should_return_quad_linear_function():
    def linear(x):
        return 2 * x + 3
    
    answer = quad_linear(linear, 0, 10)
    test, _ = quad(linear, 0, 10)

    assert answer == pytest.approx(test, rel = 1e-5)

def test_should_return_answer_of_system_equation():
    coeffs = [[2, 1], [1, -1]]
    conts = [7,1]

    answer = system_equation(coeffs, conts)
    test = solve(coeffs, conts)

    assert answer["X"] == pytest.approx(test[0], rel= 1e-5)
    assert answer["Y"] == pytest.approx(test[1], rel = 1e-5)
