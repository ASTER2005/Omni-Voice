from pathlib import Path

root = Path("C:/Omni_Voice")
fixes = [
    ('with open(cfg_path, "r", encoding="utf-8") as f:', 'with open(cfg_path, "r", encoding="utf-8") as f:'),
    ('with open(cfg_path, encoding="utf-8") as f:',      'with open(cfg_path, encoding="utf-8") as f:'),
]
changed = []
for py in root.rglob("*.py"):
    try:
        text = py.read_text(encoding="utf-8")
    except Exception:
        continue
    new = text
    for old, replacement in fixes:
        new = new.replace(old, replacement)
    if new != text:
        py.write_text(new, encoding="utf-8")
        changed.append(py.name)
print("Fixed files:", changed)
print(f"Total: {len(changed)} files patched")
