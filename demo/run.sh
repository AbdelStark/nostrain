#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
#  nostrain distributed training demo
#  4 workers exchanging gradients through a local Nostr relay
# ─────────────────────────────────────────────────────────────

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEMO="$ROOT/demo"
VENV="$ROOT/.venv"
PYTHON="$VENV/bin/python3"
NOSTRAIN="$VENV/bin/nostrain"
SESSION="nostrain-demo"

RELAY_PORT=7777
RELAY_URL="ws://127.0.0.1:$RELAY_PORT"
RUN_NAME="demo-$(date +%s)"

ROUNDS=5
INNER_STEPS=80
LOCAL_LR=0.03
OUTER_LR=0.7
MOMENTUM=0.9
BATCH_SIZE=4
TOPK=1.0
ROUND_TIMEOUT=8.0
FEATURES=3

# Colors for output
C_RESET="\033[0m"
C_BOLD="\033[1m"
C_CYAN="\033[1;36m"
C_GREEN="\033[1;32m"
C_YELLOW="\033[1;33m"
C_RED="\033[1;31m"

# ── Load keys from .env ──────────────────────────────────────
if [[ ! -f "$ROOT/.env" ]]; then
    echo -e "${C_RED}Error: .env file not found at $ROOT/.env${C_RESET}"
    exit 1
fi
source "$ROOT/.env"

KEYS=("$WORKER_1_KEY" "$WORKER_2_KEY" "$WORKER_3_KEY" "$WORKER_4_KEY")
WORKER_NAMES=("alice" "bob" "carol" "dave")
WORKER_COLORS=("31" "32" "33" "34")  # red, green, yellow, blue

# ── Preflight ────────────────────────────────────────────────
echo -e "${C_CYAN}"
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║           NOSTRAIN DISTRIBUTED TRAINING DEMO         ║"
echo "  ║                                                      ║"
echo "  ║   4 workers · local relay · DiLoCo outer loop        ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo -e "${C_RESET}"

# Check prerequisites
if ! command -v tmux &>/dev/null; then
    echo -e "${C_RED}  tmux is required. Install with: brew install tmux${C_RESET}"
    exit 1
fi

if [[ ! -f "$NOSTRAIN" ]]; then
    echo -e "${C_RED}  nostrain not found. Run: python3 -m venv .venv && .venv/bin/pip install -e .${C_RESET}"
    exit 1
fi

# Kill any previous demo session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Clean artifacts
rm -rf "$DEMO/artifacts"
mkdir -p "$DEMO/artifacts"

# ── Generate data shards ─────────────────────────────────────
echo -e "${C_GREEN}  Generating data shards...${C_RESET}"
$PYTHON "$DEMO/generate_data.py" "$DEMO"
echo ""

# ── Initialize shared model state ────────────────────────────
echo -e "${C_GREEN}  Initializing model state (${FEATURES} features)...${C_RESET}"
$NOSTRAIN init-state \
    --runtime linear-regression \
    --features "$FEATURES" \
    -o "$DEMO/initial_state.json"
MODEL_HASH=$($NOSTRAIN hash-state "$DEMO/initial_state.json")
echo -e "  Model hash: ${C_BOLD}${MODEL_HASH:0:16}...${C_RESET}"
echo ""

# ── Derive public keys for display ───────────────────────────
echo -e "${C_GREEN}  Workers:${C_RESET}"
for i in 0 1 2 3; do
    PUBKEY=$($NOSTRAIN derive-pubkey "${KEYS[$i]}")
    echo -e "  \033[1;${WORKER_COLORS[$i]}m  ${WORKER_NAMES[$i]}\033[0m  ${PUBKEY:0:16}..  shard_$((i+1)).json"
done
echo ""

# ── Build tmux layout ────────────────────────────────────────
#
#  ┌──────────────────────────────────────────────┐
#  │                RELAY (top)                   │
#  ├──────────────────────┬───────────────────────┤
#  │       alice          │         bob           │
#  ├──────────────────────┼───────────────────────┤
#  │       carol          │         dave          │
#  └──────────────────────┴───────────────────────┘

echo -e "${C_GREEN}  Launching tmux session: ${C_BOLD}${SESSION}${C_RESET}"
echo ""

# Create session with relay pane
tmux new-session -d -s "$SESSION" -x 200 -y 55

# ── Pane 0: Relay ────────────────────────────────────────────
tmux send-keys -t "$SESSION" "$PYTHON $DEMO/relay.py $RELAY_PORT" Enter

# Wait for relay to be ready
sleep 2

# ── Split: relay on top, workers in 2x2 grid below ──────────
# Split top/bottom (relay top 35%, workers bottom 65%)
tmux split-window -t "$SESSION" -v -p 65

# Split bottom into left/right
tmux split-window -t "$SESSION:0.1" -h -p 50

# Split bottom-left into top/bottom (alice top, carol bottom)
tmux split-window -t "$SESSION:0.1" -v -p 50

# Split bottom-right into top/bottom (bob top, dave bottom)
tmux split-window -t "$SESSION:0.3" -v -p 50

# Pane layout after splits:
#   0 = relay
#   1 = alice (top-left)
#   2 = carol (bottom-left)
#   3 = bob   (top-right)
#   4 = dave  (bottom-right)
PANE_MAP=(1 3 2 4)  # maps worker index -> tmux pane index

# ── Launch all workers simultaneously ─────────────────────────
for i in 0 1 2 3; do
    PANE="${PANE_MAP[$i]}"
    NAME="${WORKER_NAMES[$i]}"
    COLOR="${WORKER_COLORS[$i]}"
    SHARD="$DEMO/shard_$((i+1)).json"
    KEY="${KEYS[$i]}"

    CMD="$PYTHON $DEMO/worker.py"
    CMD+=" --name $NAME"
    CMD+=" --color $COLOR"
    CMD+=" --state $DEMO/initial_state.json"
    CMD+=" --dataset $SHARD"
    CMD+=" --relay $RELAY_URL"
    CMD+=" --run $RUN_NAME"
    CMD+=" --sec-key $KEY"
    CMD+=" --rounds $ROUNDS"
    CMD+=" --inner-steps $INNER_STEPS"
    CMD+=" --local-lr $LOCAL_LR"
    CMD+=" --outer-lr $OUTER_LR"
    CMD+=" --momentum $MOMENTUM"
    CMD+=" --batch-size $BATCH_SIZE"
    CMD+=" --topk $TOPK"
    CMD+=" --round-timeout $ROUND_TIMEOUT"
    CMD+=" --heartbeat-interval 10"
    CMD+=" --result-out $DEMO/artifacts/${NAME}_result.json"

    tmux send-keys -t "$SESSION:0.$PANE" "$CMD" Enter
done

# ── Pane titles ──────────────────────────────────────────
tmux select-pane -t "$SESSION:0.0" -T "relay"
tmux select-pane -t "$SESSION:0.1" -T "alice"
tmux select-pane -t "$SESSION:0.2" -T "carol"
tmux select-pane -t "$SESSION:0.3" -T "bob"
tmux select-pane -t "$SESSION:0.4" -T "dave"

# Enable pane borders with titles
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format " #{pane_title} "

# Focus on relay pane
tmux select-pane -t "$SESSION:0.0"

# ── Background: wait for completion, then show summary ───────
(
    # Wait for all 4 result files
    while true; do
        COUNT=$(find "$DEMO/artifacts" -name '*_result.json' 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$COUNT" -ge 4 ]]; then
            break
        fi
        sleep 2
    done

    # Let workers' final output be visible for a moment
    sleep 3

    # Kill worker panes (reverse order to keep indices stable)
    for PANE in 4 3 2 1; do
        tmux kill-pane -t "$SESSION:0.$PANE" 2>/dev/null || true
    done

    # Relay pane (0) is now the only one — stop relay and show summary
    tmux send-keys -t "$SESSION:0.0" C-c
    sleep 1
    tmux send-keys -t "$SESSION:0.0" "$PYTHON $DEMO/summary.py $DEMO/artifacts 4" Enter

    # Update pane title
    tmux select-pane -t "$SESSION:0.0" -T "summary"
    tmux set-option -t "$SESSION" pane-border-status top
) &

# ── Attach ───────────────────────────────────────────────────
echo -e "${C_CYAN}  ┌─────────────────────────────────────────────┐${C_RESET}"
echo -e "${C_CYAN}  │  Attaching to tmux session...               │${C_RESET}"
echo -e "${C_CYAN}  │                                             │${C_RESET}"
echo -e "${C_CYAN}  │  Ctrl-B then D to detach                    │${C_RESET}"
echo -e "${C_CYAN}  │  tmux kill-session -t $SESSION to stop   │${C_RESET}"
echo -e "${C_CYAN}  └─────────────────────────────────────────────┘${C_RESET}"
echo ""

tmux attach -t "$SESSION"
