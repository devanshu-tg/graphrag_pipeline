#!/bin/zsh
# The scorecard AND all three budget sections read budget50_judge.json, whose question set is
# eval_final50_v3.jsonl - NOT the 50 in the per-question table. Matching config from its _meta:
# gemini-3.7-flash, rr_pool 500, judge gemini-3.1-pro-preview, rubric v2.
cd /Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag
V=.venv/bin/python; B=_experiments/regex_router/run_bench.py; J=_experiments/regex_router/judge_arms.py
L=_experiments/regex_router/results/v3.log
: > $L
export TG_SCHEMA=datasets/enterprise/schema_v2.json
Q=datasets/enterprise/eval_final50_v3.jsonl
run() { echo "=== $1 k$2 ===" >> $L
  $V $B --arm $1 --chunks $2 --n 50 --rr-pool 500 --questions $Q --tag _v3 >> $L 2>&1; }
run graph-regex 3
run graph-regex 10
run graph-planner 3
run graph-planner 10
echo "=== judging, rubric v2 ===" >> $L
RUBRIC=v2 $V $J graph-regex_k3_v3 graph-regex_k10_v3 graph-planner_k3_v3 graph-planner_k10_v3 >> $L 2>&1
echo "V3 RUNS COMPLETE" >> $L
