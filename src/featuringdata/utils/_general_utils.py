
import math


def round_to_n_sigfig(x, n=3):
    """
    Round a number to 'n' significant digits.

    Parameters
    ----------
    x : int or float
        Any number to round.

    n : int
        Number of desired significant digits.

    Returns
    -------
    x_round : float or int
        The rounded number.

    Examples
    --------
    >>> round_to_n_sigfig(234.5, n=3)
    235
    >>> round_to_n_sigfig(0.2345, n=3)
    0.235
    """

    # First check if zero is passed to the function to avoid an error:
    if x == 0:
        return int(x)
    # Since n should be at least 1:
    if n < 1:
        n = 1

    # This one line does the actual rounding:
    x_round = round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))
    # If rounding creates a number with no digits beyond the decimal point,
    #  then make it an integer:
    if x_round > 10 ** (n - 1):
        x_round = int(x_round)
    return x_round

