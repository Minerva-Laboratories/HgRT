# test_mercurius.py is a script (needs the prepared Mercurius weights, HGRT_MERCURIUS_DIR): run it directly,
#   python tests/test_mercurius.py [weights.pt] [snapshot.pt]
# pytest skips it so `python -m pytest` works on any machine.
collect_ignore = ["test_mercurius.py"]
