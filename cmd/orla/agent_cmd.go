package main

import (
	"github.com/dorcha-inc/orla/internal/agent"
	"github.com/spf13/cobra"
)

// newAgentCmd creates the agent command for one-shot execution
func newAgentCmd() *cobra.Command {
	var modelFlag string

	cmd := &cobra.Command{
		Use:   "agent <prompt>",
		Short: "Execute a one-shot agent prompt",
		Long: `Execute a one-shot agent prompt. Orla processes the prompt and returns the result.

The prompt is provided as a single argument. If the prompt contains spaces, quote it:
  orla agent "list files in the current directory"
  orla agent "summarize this" < file.txt`,
		Args: cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return agent.ExecuteAgentPrompt(args[0], modelFlag)
		},
	}

	cmd.Flags().StringVarP(&modelFlag, "model", "m", "", "Model to use (e.g., ollama:llama3)")
	return cmd
}
