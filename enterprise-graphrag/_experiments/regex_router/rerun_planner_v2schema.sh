#!/bin/zsh
# The published run reads schema_v2.json (11 relations); my first pass let TG_SCHEMA default to
# schema.json (14), which made the planner prompt bigger and inflated the reported saving.
cd /Users/devanshu/Desktop/GraphRAG_Demo/enterprise-graphrag
L=_experiments/regex_router/results/rerun_v2schema.log
: > $L
export TG_SCHEMA=datasets/enterprise/schema_v2.json
mv _experiments/regex_router/results/graph-planner_k3.jsonl \
   _experiments/regex_router/results/graph-planner_k3_schemaV1.jsonl 2>/dev/null
.venv/bin/python _experiments/regex_router/run_bench.py --arm graph-planner --chunks 3 --n 50 >> $L 2>&1
echo "RERUN COMPLETE" >> $L
