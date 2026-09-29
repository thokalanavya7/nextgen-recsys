import gzip, json, subprocess, sys, io, os
RAW = "/home/ubuntu/repos/nextgen-recsys/data/raw"
need = set()
for l in gzip.open(os.path.join(RAW, "reviews_Beauty_5.json.gz"), "rt"):
    need.add(json.loads(l)["asin"])
print("need", len(need), "asins", flush=True)
p1 = subprocess.Popen(
    ["curl", "-s", "--retry", "3",
     "http://snap.stanford.edu/data/amazon/productGraph/metadata.json.gz"],
    stdout=subprocess.PIPE)
p2 = subprocess.Popen(["zcat"], stdin=p1.stdout, stdout=subprocess.PIPE)
p1.stdout.close()
found = 0
with open(os.path.join(RAW, "meta_beauty_2014.jsonl"), "w") as out:
    buf = io.TextIOWrapper(p2.stdout, encoding="utf-8", errors="replace")
    for line in buf:
        if not line.strip():
            continue
        # cheap prefilter before json parse
        if '"asin"' not in line and "'asin'" not in line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            import ast
            try:
                r = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
        if r.get("asin") in need:
            out.write(line)
            found += 1
            if found % 2000 == 0:
                print("found", found, flush=True)
            if found >= len(need):
                break
p1.kill(); p2.kill()
print("DONE found", found, flush=True)
