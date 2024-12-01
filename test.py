from scipy.integrate import quad
from scipy.linalg import solve
import pytest
from pytest import approx
import numpy as np

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
    """
    Solves a system of two linear equations.
    
    Args:
        coeffs: The coefficients of the equations.
        conts: The constants of the equations.
    
    Returns:
        Fields "X" and "Y" containing the answers.
    """
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

def test_should_return_samples_from_normal_distrubutions():
    mean = 5.0
    stan_dev = 1.0
    sample = 1000

    samples = normal_distrubutions_samples(mean, stan_dev, sample)
    assert len(samples) == sample
    assert pytest.approx(np.mean(samples), rel = 0.1) == mean
    assert pytest.approx(np.stan_dev(samples), rel = 0.1) == stan_dev
