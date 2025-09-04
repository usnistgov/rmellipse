from itertools import cycle

class colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def iter_colors(cls):
        attrs = dir(cls)
        for a in attrs:
            if "_" not in a:
                yield a, getattr(cls, a)


class symbols:
    CHECK = "\u2714"
    XBOX = "\u2612"
    BIGX = "\u2A09"

    @classmethod
    def iter_symbols(cls):
        attrs = dir(cls)
        for a in attrs:
            if "_" not in a:
                yield a, getattr(cls, a)


def cstr(*values, color: colors = None):
    """
    Color values

    Parameters
    ----------
    color : colors, optional
        _description_, by default None

    Returns
    -------
    values:
        tuple of strings, so that when passed
        through print they are colored
    """
    cvalues = [v for v in values]
    cvalues[0] = color + str(cvalues[0])
    cvalues[-1] = (cvalues[-1]) + colors.ENDC
    return cvalues


def cprint(*values, color: colors = None, **kwargs):
    """
    Print *values with a color.

    Parameters
    ----------
    color : colors, optional
        Color to print in, None does no color, default prinnt
        statement.
    """
    if color:
        cvalues = cstr(*values, color=color)
        print(*cvalues, **kwargs)
    else:
        print(*values, **kwargs)

braile_load = ['⣾','⣽','⣻','⢿','⡿','⣟','⣯','⣷']

if __name__ == "__main__":
    for name, color in colors.iter_colors():
        cprint("testing :", name, color=color)

    for name, symbol in symbols.iter_symbols():
        cprint("testing :", name, symbol)
