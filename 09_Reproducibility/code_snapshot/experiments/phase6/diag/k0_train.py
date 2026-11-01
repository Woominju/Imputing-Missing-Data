"""POST-HOC DIAGNOSTIC training: unchanged experiments/phase6/train6.py loop on the diagnostic data.
usage: k0_train.py VERSION(J|S) <train6 args without --out>"""
import sys
from experiments.phase6.diag.k0_common import patch, ROOT
v = sys.argv[1]; patch(v)
import experiments.phase6.train6 as T
T.build = __import__("experiments.phase6.core6", fromlist=["build"]).build
sys.argv = [sys.argv[0]] + sys.argv[2:] + ["--out", f"{ROOT}/runs_{v}"]
T.main()
