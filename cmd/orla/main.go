package main

import (
	"fmt"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

var (
	// version is set via -ldflags at build time
	version string
	// buildDate is set via -ldflags at build time
	buildDate string
)

func validateVersionAndBuildDate() {
	if version == "" {
		zap.L().Fatal("version is not set, please set the version via -ldflags at build time")
	}

	if buildDate == "" {
		zap.L().Fatal("buildDate is not set, please set the buildDate via -ldflags at build time")
	}
}

func main() {
	validateVersionAndBuildDate()

	rootCmd := &cobra.Command{
		Use:     "orla",
		Short:   "Orla agent engine and CLI",
		Long:    `Orla is an execution engine for building high performance agentic systems. Use "orla serve" to run the agent engine as a service or "orla agent" for one-shot agent runs.`,
		Version: fmt.Sprintf("%s (built: %s)", version, buildDate),
	}

	rootCmd.AddCommand(newServeCmd())
	rootCmd.AddCommand(newAgentCmd())

	rootErr := rootCmd.Execute()
	if rootErr != nil {
		zap.L().Fatal("Error executing root command", zap.Error(rootErr))
	}
}
