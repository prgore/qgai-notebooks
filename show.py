import sys, nbformat
nb = nbformat.read(sys.argv[1], 4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    for o in c.get("outputs", []):
        if o.output_type == "stream": print(f"[{i}] {o.text.rstrip()}")
        elif o.output_type == "error": print(f"[{i}] ERROR {o.ename}: {o.evalue}")
        elif o.output_type in ("display_data", "execute_result"):
            print(f"[{i}] <{', '.join(k for k in o.data if k != 'text/plain') or o.data.get('text/plain','')[:80]}>")
