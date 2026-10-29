"""Finds trained-model files and class names under one or more folders, and prints a report to paste back. READS only: it never loads or runs a model file.
    python3 tools/find_model_files.py <folder> [<folder> ...]
Lists *.onnx / *.pth / *.pt / *.safetensors / *.ckpt (size + sha256), and the `names:` / class lists from any data.yaml, classes.txt, args.yaml or *.json that look like detector configs."""
import hashlib, json, os, re, sys

EXT = (".onnx", ".pth", ".pt", ".safetensors", ".ckpt")
SKIP = {"node_modules", ".git", "site-packages", "__pycache__", ".venv", "venv"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def main(roots):
    models, configs = [], []
    for root in roots:
        for d, dirs, files in os.walk(os.path.expanduser(root)):
            dirs[:] = [x for x in dirs if x not in SKIP]
            for f in files:
                p = os.path.join(d, f); lo = f.lower()
                if lo.endswith(EXT): models.append(p)
                elif lo in ("data.yaml", "classes.txt", "args.yaml", "class_names.txt", "labels.txt") or (lo.endswith(".json") and ("class" in lo or "categories" in lo or "_annotations.coco" in lo)): configs.append(p)
    print(f"== {len(models)} model file(s)")
    for p in sorted(models, key=os.path.getmtime, reverse=True)[:40]: print(f"{os.path.getsize(p) / 1e6:9.1f} MB  {sha(p)[:16]}  {p}")
    print(f"\n== class lists found in {len(configs)} config file(s)")
    for p in configs[:40]:
        try:
            t = open(p, errors="ignore").read(200_000)
            names = re.findall(r"names:\s*(\[.*?\]|(?:\n\s+\d+:\s*.+)+)", t, re.S)
            if p.lower().endswith(".json"):
                j = json.loads(t); cats = j.get("categories") if isinstance(j, dict) else None; names = [json.dumps([c.get("name") for c in cats])] if cats else []
            elif p.lower().endswith(".txt"): names = [json.dumps(t.split())]
            if names: print(f"{p}\n    {names[0].strip()[:300]}")
        except Exception as e: print(f"{p}: could not read ({type(e).__name__})")


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    main(sys.argv[1:])
