import re
with open('app_gui.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = re.sub(r'c_start, c_stop, c_desktop = st\.columns\(\[1, 1, 1\.5\]\)', 'c_start, c_stop = st.columns([1, 1])', code)
code = re.sub(r'        with c_desktop:.*?st\.error\(f"Error: \{ex\}"\)\n', '', code, flags=re.DOTALL)

with open('app_gui.py', 'w', encoding='utf-8') as f:
    f.write(code)
