"""Latency of raw-line retrieval (inverted index) vs a full scan as history grows; ingest time with provenance folding on."""
import time
from random import Random
from nsa import NSA, RuntimeConfig
from nsa.memory.extract import normalize_key
class Null:
    model="n"
    def generate(self,p,**k): return ""
rng=Random(0)
words=[f"w{i}" for i in range(400)]+["staging","billing","payments","auth","search","reports","gateway","mailer","scheduler","uploads","database","port","email","region","cache","size","queue","name","log","level","owner","team","admin","timeout","seconds"]
def line(): return " ".join(rng.choice(words) for _ in range(9))
print(f"{'lines':>8}{'index build s':>15}{'indexed lines_for ms':>22}{'full scan ms':>14}", flush=True)
for n in (1000, 10000, 100000):
    a=NSA(Null(), config=RuntimeConfig(retrieve_lines=3, provenance_window=256))
    t=time.perf_counter()
    for _ in range(n): a.observe(line(), source="chat")
    build=time.perf_counter()-t
    q="What is the cache size for billing?"
    t=time.perf_counter()
    for _ in range(50): a.lines_for(q,3)
    idx=(time.perf_counter()-t)/50*1e3
    reps=3 if n>=100000 else 20
    t=time.perf_counter()
    qs=set(normalize_key(q).split())
    for _ in range(reps):
        sorted(((len(qs & set(normalize_key(l).split())),i) for i,l in enumerate(a._lines)),reverse=True)[:3]
    scan=(time.perf_counter()-t)/reps*1e3
    print(f"{n:>8}{build:>15.2f}{idx:>22.2f}{scan:>14.1f}", flush=True)
