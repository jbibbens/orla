package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/dorcha-inc/orla/internal/config"
	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/serving"
	servingapi "github.com/dorcha-inc/orla/internal/serving/api"
	"github.com/spf13/cobra"
	"go.uber.org/zap"
)

// newServeCmd creates the serve command that runs the agent engine as a service
func newServeCmd() *cobra.Command {
	var configPath string
	var listenAddr string
	var prettyLog bool

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Start orla's agent engine as a service",
		Long:  `Start orla's agent engine as a service. The server runs as a long-lived process. Use "orla agent" for one-shot agent runs.`,
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, loadConfigErr := config.LoadConfig(configPath)
			if loadConfigErr != nil {
				zap.L().Fatal("Failed to load config", zap.Error(loadConfigErr))
			}

			// Resolve logging format: CLI flag wins; otherwise config
			resolvedPrettyLog := resolveLogFormat(cfg, prettyLog)

			coreInitErr := core.InitLogger(resolvedPrettyLog)
			if coreInitErr != nil {
				zap.L().Fatal("Failed to initialize logger", zap.Error(coreInitErr))
			}

			listenAddress := cfg.ListenAddress
			if listenAddress == "" {
				listenAddress = "localhost:8081"
			}
			if listenAddr != "" {
				listenAddress = listenAddr
			}

			layer := serving.NewAgenticLayer()

			apiServer := servingapi.NewAgenticServer(layer, listenAddress)

			// Set up signal handling for graceful shutdown
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			sigChan := make(chan os.Signal, 1)
			signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

			// Start server in a goroutine
			errChan := make(chan error, 1)
			go func() {
				if err := apiServer.Start(); err != nil {
					errChan <- err
				}
			}()

			// Wait for signal or error
			select {
			case sig := <-sigChan:
				zap.L().Info("Received signal, shutting down",
					zap.String("signal", sig.String()))
				if err := apiServer.Shutdown(ctx); err != nil {
					return fmt.Errorf("failed to shutdown server: %w", err)
				}
			case err := <-errChan:
				if err != nil {
					return fmt.Errorf("server error: %w", err)
				}
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&configPath, "config", "c", "", "Path to config file (default: uses precedence)")
	cmd.Flags().StringVarP(&listenAddr, "listen-address", "l", "", "Address to listen on (default: from config or localhost:8081)")
	cmd.Flags().BoolVar(&prettyLog, "pretty", false, "Use pretty-printed logs instead of JSON")

	return cmd
}

// resolveLogFormat determines the log format based on CLI flag and config
func resolveLogFormat(cfg *config.OrlaConfig, prettyLog bool) bool {
	if !prettyLog && cfg.LogFormat == "pretty" {
		return true
	}
	return prettyLog
}
