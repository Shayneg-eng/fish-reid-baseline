import sys
from pathlib import Path

# Run this from C:\Coding\Research\Fishal Identification\baseline_salmon
p = Path(sys.argv[1] if len(sys.argv) > 1 else "data/salmon_reid/reid_dataset/reid/analysis1")

print(f"=== Deep listing of {p} ===")
count = 0
for sub in sorted(p.rglob("*")):
    print(sub.relative_to(p), "(dir)" if sub.is_dir() else f"({sub.stat().st_size} bytes)")
    count += 1
    if count >= 60:
        print("... (truncated)")
        break

txt_files = list(p.glob("*.txt"))
if txt_files:
    print(f"\n=== First 20 lines of {txt_files[0].name} ===")
    with open(txt_files[0], "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            print(line.rstrip())
