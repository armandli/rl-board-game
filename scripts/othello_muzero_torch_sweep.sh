#!/usr/bin/env bash
# Hyperparameter search for othello_muzero_torch_train at an equal wall clock
# budget: every trial trains for --minutes from scratch (in-run evaluation off,
# so all trials get the same GPU time), then its final model plays fixed
# opponents in othello_muzero_torch_arena. Results go to <out>/results.csv,
# one row per trial and opponent, and the summary is printed at the end.
#
#   scripts/othello_muzero_torch_sweep.sh TRIALS_FILE [--minutes 20] [--out data/sweep]
#
# TRIALS_FILE holds one trial per line, "name  extra training options";
# blank lines and lines starting with # are skipped. A trial whose final model
# exists is not trained again, so an interrupted sweep resumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN="$ROOT/build/bin/othello_muzero_torch_train"
ARENA="$ROOT/build/bin/othello_muzero_torch_arena"
MINUTES=20
OUT="$ROOT/data/sweep"
GAMES=100
OPPONENTS="rollout:1000,alphabeta:4"

[[ $# -ge 1 ]] || { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
TRIALS="$1"; shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --minutes)   MINUTES="$2"; shift 2 ;;
    --out)       OUT="$2"; shift 2 ;;
    --games)     GAMES="$2"; shift 2 ;;
    --opponents) OPPONENTS="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
mkdir -p "$OUT"

while read -r name args; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  dir="$OUT/$name"
  if [[ ! -f "$dir/final_model.pt" ]]; then
    echo "=== $(date +%T) training $name: $args"
    # shellcheck disable=SC2086
    "$TRAIN" --data-dir "$dir" --no-resume --no-evaluate --max-minutes "$MINUTES" --seed 1 $args \
      > "$dir.log" 2>&1 || { echo "trial $name failed, see $dir.log"; continue; }
    tail -n 3 "$dir.log" | grep '\[train\] ' | tail -n 1 | cut -c1-200 || true
  fi
  if ! grep -q "^$name," "$OUT/results.csv" 2>/dev/null; then
    "$ARENA" --model "$dir/final_model.pt" --opponent "$OPPONENTS" --games "$GAMES" --csv "$dir/arena.csv" \
      | sed "s/^/  /"
    [[ -f "$OUT/results.csv" ]] || echo "trial,steps,games,opponent,score,elo_diff" > "$OUT/results.csv"
    steps=$(cat "$dir/checkpoint/learner.txt")
    played=$(tail -n 1 "$dir/train_log.csv" | cut -d, -f3)
    tail -n +2 "$dir/arena.csv" | while IFS=, read -r model opp g sims w d l score elo; do
      echo "$name,$steps,$played,$opp,$score,$elo" >> "$OUT/results.csv"
    done
  fi
done < "$TRIALS"

echo "=== summary (score against each opponent; mean over opponents last)"
awk -F, 'NR > 1 { s[$1] += $5; n[$1]++; line[$1] = line[$1] sprintf(" %s %.3f", $4, $5); st[$1] = $2; g[$1] = $3 }
         END { for (t in s) printf "%-28s steps %6d games %6d |%s | mean %.3f\n", t, st[t], g[t], line[t], s[t] / n[t] }' \
  "$OUT/results.csv" | sort -t'|' -k3 -r
