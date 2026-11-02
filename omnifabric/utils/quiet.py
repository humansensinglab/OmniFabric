"""Hide what third-party code prints about its internal state, keeping progress bars and errors."""

import contextlib
import io
import sys


class _FilteredStdout(io.TextIOBase):
    def __init__(self, out, progress):
        self.out, self.progress = out, progress
        self.buffer_, self.in_bar = '', False

    def write(self, s):
        if self.progress and s.startswith('\r'):  # carriage-return progress bar, e.g. '\rProgress: [###---] 40.0%'
            self.out.write(s)
            self.out.flush()
            self.in_bar = True
            return len(s)
        self.buffer_ += s
        *lines, self.buffer_ = self.buffer_.split('\n')
        for line in lines:
            if 'error' in line.lower() or 'fail' in line.lower():
                self.end_bar()
                self.out.write(line + '\n')
        return len(s)

    def end_bar(self):
        if self.in_bar:
            self.out.write('\n')
            self.in_bar = False

    def flush(self):
        self.out.flush()


@contextlib.contextmanager
def quiet(progress=False):
    """Drop stdout inside the block except lines mentioning an error or failure (and, with progress=True,
    carriage-return progress bars). Warnings and tracebacks on stderr are not affected."""
    filtered = _FilteredStdout(sys.stdout, progress)
    try:
        with contextlib.redirect_stdout(filtered):
            yield
    finally:
        filtered.end_bar()
