with open('table1.tex','r') as f:
    data = f.read()
# Restore 0.00e+00 for s in {0.1, 0.3} forward error and mismatch
data = data.replace(
    'bfloat16 & 0.1 & — & — & — \\\\',
    'bfloat16 & 0.1 & — & 0.00e+00 & 0.00e+00 \\\\'
)
data = data.replace(
    'bfloat16 & 0.3 & — & — & — \\\\',
    'bfloat16 & 0.3 & — & 0.00e+00 & 0.00e+00 \\\\'
)
data = data.replace(
    'float32 & 0.1 & — & — & — \\\\',
    'float32 & 0.1 & — & 0.00e+00 & 0.00e+00 \\\\'
)
data = data.replace(
    'float32 & 0.3 & — & — & — \\\\',
    'float32 & 0.3 & — & 0.00e+00 & 0.00e+00 \\\\'
)
# Update note to say "—" instead of "no"
data = data.replace(
    'Note: probe pass $=$ ``no'' only for $s\\in\\{0.1,0.3\\}$, which were evaluated via the Bertoin algebraic-emulation audit rather than the six-point unit probe; blank cells are em-dashed.',
    'Note: probe pass $=$ ``---'' for $s\\in\\{0.1,0.3\\}$, which were evaluated via the Bertoin algebraic-emulation audit rather than the six-point unit probe; forward error and mismatch are $0$ for these slopes.'
)
with open('table1.tex','w') as f:
    f.write(data)
print('Table 1 fixed')