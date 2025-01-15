import ast
import contextlib
import sys
from io import StringIO


def extract_code(string):
    code = string.split('```python')[-1]
    code = code.split('```')[0]
    return code


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def run_code(res):
    code = extract_code(res)
    with stdoutIO() as s:
        exec(code)
    res = s.getvalue()
    res = ast.literal_eval(res)
    return res,code


def is_right(generate_code, res, answer, code):
    if type(res) != list:
        if answer == res:
            generate_code.append(code)
            return True
        else:
            generate_code.append(code)
            return False
    else:
        if answer in res:
            generate_code.append(code)
            return True
        else:
            generate_code.append(code)
            return False