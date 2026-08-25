#!/bin/zsh
# zsh does NOT word-split unquoted parameter expansions, so a "for spec in 'arm k'" loop with
# `set -- $spec` passes "graph-regex 3" as ONE argument. Explicit lines instead.
cd /Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag
V=.venv/bin/python
B=_experiments/regex_router/run_bench.py
L=_experiments/regex_router/results/run.log
: > $L
run() { echo "======== arm=$1 chunks=$2 ========" >> $L; $V $B --arm $1 --chunks $2 --n 50 >> $L 2>&1; }
run graph-regex 3
run graph-regex 10
run graph-planner 3
run graph-planner 10
run basic 3
run basic 10
echo "ALL RUNS COMPLETE" >> $L
