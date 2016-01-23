with open("csplda.py") as f:
    code = compile(f.read(), "csplda.py", 'exec')
    exec(code, None, None)