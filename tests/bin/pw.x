#!/usr/bin/env python3
"""Test stand-in for pw.x. Writes a pw.x-like output to stdout.

FAKE_PW_MODE: ok (default) | noconv | crash | nod3 | exit1
FAKE_PW_LOG:  file to which each call appends the input file name
Energy: -(number of atoms) Ry - 0.001 × (sum of z coordinates) Ry; D3 term: -0.01 × atoms Ry.
"""
import os, sys
args = sys.argv[1:]
fin = args[args.index("-in") + 1]
if os.environ.get("FAKE_PW_LOG"):
    open(os.environ["FAKE_PW_LOG"], "a").write(os.path.abspath(fin) + "\n")
lines = open(fin).read().splitlines()
i = next(k for k, l in enumerate(lines) if l.startswith("ATOMIC_POSITIONS"))
zs = []
for l in lines[i + 1:]:
    p = l.split()
    if len(p) < 4: break
    try: zs.append(float(p[3]))
    except ValueError: break
mode = os.environ.get("FAKE_PW_MODE", "ok")
print("     Program PWSCF v.7.6 starts on  1Jan2026 at  0: 0: 0")
print("     Estimated max dynamical RAM per process >       0.10 GB")
if mode == "crash":
    sys.exit(0)
if mode != "nod3":
    print(f"     DFT-D3 Dispersion         = {-0.01 * len(zs):16.8f} Ry")
print(f"!    total energy              = {-len(zs) - 0.001 * sum(zs):16.8f} Ry")
print("     convergence NOT achieved after 300 iterations: stopping" if mode == "noconv"
      else "     convergence has been achieved in  10 iterations")
print("   JOB DONE.")
sys.exit(1 if mode == "exit1" else 0)
