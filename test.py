from scipy.integrate import quad
from pytest import approx

def test_should_return_quad_linear_function():
    def linear(x):
        x = 2 * x + 3
        return x
    
    answer = quad_linear(x, 0, 10)
    test, _ = quad(x, 0, 10)

    assert answer == pytest.approx(test, rel = 1e - 5)