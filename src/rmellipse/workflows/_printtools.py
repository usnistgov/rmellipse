import sys


class Colors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKCYAN = '\033[96m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	RED = '\033[31m'

	@classmethod
	def iter_colors(cls):
		attrs = dir(cls)
		for a in attrs:
			if '_' not in a:
				yield a, getattr(cls, a)


class symbols:
	CHECK = '\u2714'
	XBOX = '\u2612'
	BIGX = '\u2a09'

	@classmethod
	def iter_symbols(cls):
		attrs = dir(cls)
		for a in attrs:
			if '_' not in a:
				yield a, getattr(cls, a)


class Printer:
	"""
	Class for managing printing.
	"""

	def __init__(self, header: str, header_color: Colors):
		"""
		Initialize a printer.

		Parameters
		----------
		header : str
			Printed only on the first call to the printer.
		header_color : Color
			Color of the header
		"""
		self.header = header
		self.header_color = header_color
		self.header_printed = False
		self.print_count = 0

	def cprint(self, *args, **kwargs):
		if not self.header_printed:
			cprint(self.header, color=self.header_color)
		cprint(*args, **kwargs)


def cstr(*values, color: Colors = None):
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
	cvalues[-1] = (cvalues[-1]) + Colors.ENDC
	return cvalues


def cprint(*values, color: Colors = None, **kwargs):
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


class PrintManager:
	def __init__(self):
		self.new_lines = 0

	def cprint(self, *args, end='\n', **kwargs):
		if end == '\n':
			self.new_lines += 1
		self.new_lines += ''.join([str(a) for a in args]).count('\n')
		cprint(*args, end=end, **kwargs)
		sys.stdout.flush()

	def clear(self):
		# sys.stdout.write('\n')
		sys.stdout.write('\033[A\033[2K\r' * (self.new_lines))
		self.new_lines = 0


braile_load = ['⣾', '⣽', '⣻', '⢿', '⡿', '⣟', '⣯', '⣷']


if __name__ == '__main__':
	for name, color in Colors.iter_colors():
		cprint('testing :', name, color=color)

	for name, symbol in symbols.iter_symbols():
		cprint('testing :', name, symbol)

	pm = PrintManager()
	pm.cprint('shouldnt', 'be', '\n', 'visible')
	pm.cprint(222222, 'can you see me?')
	pm.clear()
	pm.cprint('hello invis \n')
	pm.clear()
