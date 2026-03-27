#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  nostrain GPT demo — Shakespeare over Nostr
#  4 workers train a char-level GPT through a local relay
# ─────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEMO="$ROOT/demo/gpt"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python3"
SESSION="nostrain-gpt"

RELAY_PORT=7778
RELAY_URL="ws://127.0.0.1:$RELAY_PORT"
RUN_NAME="gpt-$(date +%s)"

ROUNDS=5
INNER_STEPS=100
LR=0.0003
OUTER_LR=0.7
MOMENTUM=0.9
BATCH_SIZE=32
TOPK=0.3

C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[1;36m"
C_GREEN="\033[1;32m"
C_RED="\033[1;31m"

# ── Load keys ────────────────────────────────────────────────
if [[ ! -f "$ROOT/.env" ]]; then
    echo -e "${C_RED}Error: .env not found${C_RESET}"
    exit 1
fi
source "$ROOT/.env"

KEYS=("$WORKER_1_KEY" "$WORKER_2_KEY" "$WORKER_3_KEY" "$WORKER_4_KEY")
WORKER_NAMES=("alice" "bob" "carol" "dave")
WORKER_COLORS=("31" "32" "33" "34")

# ── Preflight ────────────────────────────────────────────────
echo -e "${C_CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║     NOSTRAIN GPT DEMO — Shakespeare over Nostr       ║"
echo "  ║                                                      ║"
echo "  ║   4 workers · char-level GPT · DiLoCo outer loop     ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${C_RESET}"

if ! command -v tmux &>/dev/null; then
    echo -e "${C_RED}  tmux required. brew install tmux${C_RESET}"
    exit 1
fi

tmux kill-session -t "$SESSION" 2>/dev/null || true
rm -rf "$ROOT/demo/artifacts"
mkdir -p "$ROOT/demo/artifacts"

# Download data ahead of time
echo -e "${C_GREEN}  Downloading Shakespeare dataset...${C_RESET}"
$PYTHON -c "from demo.gpt.data import download_shakespeare; download_shakespeare()" 2>/dev/null || \
  PYTHONPATH="$ROOT" $PYTHON -c "from demo.gpt.data import download_shakespeare; download_shakespeare()"
echo -e "${C_GREEN}  Model: ~834K params (4L/4H/128E char-level GPT)${C_RESET}"
echo -e "${C_GREEN}  Training: $ROUNDS rounds x $INNER_STEPS inner steps, topk=$TOPK${C_RESET}"
echo ""

# ── tmux layout ──────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -x 200 -y 55

# Pane 0: relay
tmux send-keys -t "$SESSION" "PYTHONPATH=$ROOT $PYTHON $ROOT/demo/relay.py $RELAY_PORT" Enter
sleep 2

# 2x2 grid for workers
tmux split-window -t "$SESSION" -v -p 65
tmux split-window -t "$SESSION:0.1" -h -p 50
tmux split-window -t "$SESSION:0.1" -v -p 50
tmux split-window -t "$SESSION:0.3" -v -p 50

# Pane map: worker 0->1, 1->3, 2->2, 3->4
PANE_MAP=(1 3 2 4)

for i in 0 1 2 3; do
    PANE="${PANE_MAP[$i]}"
    NAME="${WORKER_NAMES[$i]}"
    COLOR="${WORKER_COLORS[$i]}"

    CMD="PYTHONPATH=$ROOT $PYTHON $ROOT/demo/gpt/worker.py"
    CMD+=" --name $NAME"
    CMD+=" --color $COLOR"
    CMD+=" --sec-key ${KEYS[$i]}"
    CMD+=" --shard-id $i"
    CMD+=" --relay $RELAY_URL"
    CMD+=" --run $RUN_NAME"
    CMD+=" --rounds $ROUNDS"
    CMD+=" --inner-steps $INNER_STEPS"
    CMD+=" --lr $LR"
    CMD+=" --outer-lr $OUTER_LR"
    CMD+=" --momentum $MOMENTUM"
    CMD+=" --batch-size $BATCH_SIZE"
    CMD+=" --topk $TOPK"
    CMD+=" --result-out $ROOT/demo/artifacts/${NAME}_gpt_result.json"

    tmux send-keys -t "$SESSION:0.$PANE" "$CMD" Enter
done

# Pane titles
tmux select-pane -t "$SESSION:0.0" -T "relay"
tmux select-pane -t "$SESSION:0.1" -T "alice"
tmux select-pane -t "$SESSION:0.2" -T "carol"
tmux select-pane -t "$SESSION:0.3" -T "bob"
tmux select-pane -t "$SESSION:0.4" -T "dave"
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format " #{pane_title} "
tmux select-pane -t "$SESSION:0.0"

# ── Background: wait then show summary ───────────────────────
(
    while true; do
        COUNT=$(find "$ROOT/demo/artifacts" -name '*_gpt_result.json' 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$COUNT" -ge 4 ]]; then break; fi
        sleep 3
    done
    sleep 3
    for PANE in 4 3 2 1; do
        tmux kill-pane -t "$SESSION:0.$PANE" 2>/dev/null || true
    done
    tmux send-keys -t "$SESSION:0.0" C-c
    sleep 1
    tmux send-keys -t "$SESSION:0.0" "PYTHONPATH=$ROOT $PYTHON $ROOT/demo/gpt/summary.py $ROOT/demo/artifacts 4" Enter
    tmux select-pane -t "$SESSION:0.0" -T "summary"
    tmux set-option -t "$SESSION" pane-border-status top
) &

echo -e "${C_CYAN}  Attaching to tmux session: ${C_BOLD}${SESSION}${C_RESET}"
echo -e "${C_CYAN}  Ctrl-B then D to detach${C_RESET}"
echo ""

tmux attach -t "$SESSION"
