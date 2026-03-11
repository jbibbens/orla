# CAIS 2026 Demo: Orla Customer Support Workflow
#
# Pre-recording requirements:
#   1. SGLang backends must be running:
#      sudo docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
#   2. Wait for health checks to pass (both SGLang models + Orla server)
#   3. Run `sudo -v` to cache sudo credentials before starting
#   4. chmod +x vhs/demo_run.sh
#   5. Run from the repo root: vhs vhs/demo.tape
#
# Timing: The main Sleep (180s) must be longer than the total runtime of both
# tickets. Do a dry run of vhs/demo_run.sh first and adjust accordingly.

Output share/demo.mp4

Require tmux
Require curl

Set FontSize 14
Set FontFamily "JetBrains Mono"
Set Height 800
Set Width 1800
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 30ms
Set Padding 15

# =========================================================================
# Segment 1: Setup — start backends and verify health
# =========================================================================

Type "sudo docker compose -f deploy/docker-compose.workflow-demo.yaml up -d"
Enter
Sleep 8s

Type "curl -s http://localhost:8081/api/v1/health | jq"
Enter
Sleep 3s

# =========================================================================
# Segment 2: Start tmux with Orla server logs in right pane
# =========================================================================

Type "tmux new-session -d -s demo"
Enter
Sleep 1s

Type "tmux split-window -h -t demo"
Enter
Sleep 1s

# Right pane: tail only new Orla server logs (skip historical output)
Type 'tmux send-keys -t demo:0.1 "sudo docker compose -f deploy/docker-compose.workflow-demo.yaml logs -f orla 2>&1 | tail -n 0 -f" Enter'
Enter
Sleep 3s

# Attach to the tmux session (left pane is active)
Type "tmux attach -t demo"
Enter
Sleep 2s

# =========================================================================
# Segment 3: Run both tickets via the demo script
# =========================================================================

Type "./vhs/demo_run.sh"
Enter
Sleep 40s

# =========================================================================
# Segment 4: Teardown
# =========================================================================

# Exit tmux
Ctrl+b
Type "d"
Sleep 1s

Type "sudo docker compose -f deploy/docker-compose.workflow-demo.yaml down"
Enter
Sleep 8s
