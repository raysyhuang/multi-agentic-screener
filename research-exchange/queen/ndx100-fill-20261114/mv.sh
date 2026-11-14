#!/bin/bash
d=$1; shift
A=/cursor/stores/user/grok-bot/agents/00ff6f5b-7e1d-485b-9a6b-50d48c71dc92/assets
while [ $# -gt 0 ]; do id=$2; if [ -f /home/box/.cursor/projects/workspace/agent-tools/$id.txt ]; then p=/home/box/.cursor/projects/workspace/agent-tools/$id.txt; else p=$A/$id.txt; fi
cp $p raw/$d/$1.csv && echo "$d $1 $id $(wc -l < raw/$d/$1.csv)" >> fetch_log.txt; shift 2; done
tail -n 25 fetch_log.txt | awk '{print $2,$4}' | tr '\n' ';'
