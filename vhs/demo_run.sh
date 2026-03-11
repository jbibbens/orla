#!/bin/bash
# Demo runner for CAIS 2026 video recording.
# Runs both customer support tickets through Orla sequentially.
# Usage: ./vhs/demo_run.sh (from repo root)

set -e

bold="\033[1m"
reset="\033[0m"

echo ""
echo -e "${bold}Orla Customer Support Workflow${reset}"
echo ""
echo "  Light backend: Qwen3-4B (classify)"
echo "  Heavy backend: Qwen3-8B (policy_check, reply, route_ticket)"
echo ""
sleep 3

# --- Ticket 1 ---
echo -e "${bold}Ticket 1: Billing Dispute${reset}"
echo ""
batcat --style=plain --paging=never examples/workflow_demo/tickets/billing.txt
sleep 4
echo ""
TICKET_PATH=examples/workflow_demo/tickets/billing.txt go run ./examples/workflow_demo/cmd/workflow_demo
sleep 3

# --- Ticket 2 ---
echo ""
echo -e "${bold}Ticket 2: Account Compromise${reset}"
echo ""
batcat --style=plain --paging=never examples/workflow_demo/tickets/account_compromise.txt
sleep 4
echo ""
TICKET_PATH=examples/workflow_demo/tickets/account_compromise.txt go run ./examples/workflow_demo/cmd/workflow_demo
sleep 3

echo ""
echo -e "${bold}Demo complete.${reset}"
