#!/bin/zsh
cd /Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag
V=.venv/bin/python
B=_experiments/regex_router/run_bench.py
J=_experiments/regex_router/judge_arms.py
L=_experiments/regex_router/results/missing.log
: > $L
export TG_SCHEMA=datasets/enterprise/schema_v2.json
# 5-chunk arms: the page's "Equal budget - 5 chunks each" section needs them
echo "=== regex k5 ===" >> $L;   $V $B --arm graph-regex   --chunks 5 --n 50 >> $L 2>&1
echo "=== planner k5 ===" >> $L; $V $B --arm graph-planner --chunks 5 --n 50 >> $L 2>&1
# judge every arm that has no score yet, same judge + rubric as the published columns
echo "=== judging ===" >> $L
$V $J graph-regex_k5 graph-planner_k5 graph-regex_k10 graph-planner_k10 >> $L 2>&1
echo "MISSING RUNS COMPLETE" >> $L
