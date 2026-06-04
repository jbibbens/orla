// Package commands defines the cobra command tree rooted at "orla".
package commands

import "github.com/spf13/cobra"

var rootCmd = &cobra.Command{
	Use:           "orla",
	Short:         "Adaptive execution layer for agentic systems",
	SilenceUsage:  true,
	SilenceErrors: true,
}

func init() {
	rootCmd.AddCommand(serveCmd)
}

// Execute runs the root command. Returns the underlying cobra error so
// main can decide on the exit code.
func Execute() error {
	return rootCmd.Execute()
}
