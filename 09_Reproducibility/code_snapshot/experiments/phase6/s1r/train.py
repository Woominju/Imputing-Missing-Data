"""S1R training: unchanged train6 loop on S1R data. usage: train.py <train6 args without --out>"""
import sys
from experiments.phase6.s1r.common import patch, ROOT
v = "s1r"; patch()
import experiments.phase6.train6 as T
T.build = __import__("experiments.phase6.core6", fromlist=["build"]).build
sys.argv = [sys.argv[0]] + sys.argv[1:] + ["--out", f"{ROOT}/runs"]
T.main()
