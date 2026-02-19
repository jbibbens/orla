// Package config provides configuration management for Orla: load from a
// single config file path, or use defaults when path is empty.
package config

import (
	"fmt"
	"path/filepath"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/tools"
	"github.com/spf13/viper"
)

const (
	DefaultModel        = "ollama:qwen3:0.6b"
	DefaultMaxToolCalls = 10
)

type OrlaLogLevel string

const (
	OrlaLogLevelDebug OrlaLogLevel = "debug"
	OrlaLogLevelInfo  OrlaLogLevel = "info"
	OrlaLogLevelWarn  OrlaLogLevel = "warn"
	OrlaLogLevelError OrlaLogLevel = "error"
	OrlaLogLevelFatal OrlaLogLevel = "fatal"
)

func ValidLogLevels() map[OrlaLogLevel]struct{} {
	return map[OrlaLogLevel]struct{}{
		OrlaLogLevelDebug: {},
		OrlaLogLevelInfo:  {},
		OrlaLogLevelWarn:  {},
		OrlaLogLevelError: {},
		OrlaLogLevelFatal: {},
	}
}

func IsValidLogLevel(level OrlaLogLevel) bool {
	validLogLevels := ValidLogLevels()
	_, ok := validLogLevels[level]
	return ok
}

// OrlaOutputFormat represents the output format for agent mode
type OrlaOutputFormat string

const (
	OrlaOutputFormatAuto  OrlaOutputFormat = "auto"
	OrlaOutputFormatRich  OrlaOutputFormat = "rich"
	OrlaOutputFormatPlain OrlaOutputFormat = "plain"
)

func ValidOutputFormats() map[OrlaOutputFormat]struct{} {
	return map[OrlaOutputFormat]struct{}{
		OrlaOutputFormatAuto:  {},
		OrlaOutputFormatRich:  {},
		OrlaOutputFormatPlain: {},
	}
}

func IsValidOutputFormat(format OrlaOutputFormat) bool {
	_, ok := ValidOutputFormats()[format]
	return ok
}

type OrlaLogFormat string

const (
	OrlaLogFormatPretty OrlaLogFormat = "pretty"
	OrlaLogFormatJSON   OrlaLogFormat = "json"
)

func ValidLogFormats() map[OrlaLogFormat]struct{} {
	return map[OrlaLogFormat]struct{}{
		OrlaLogFormatPretty: {},
		OrlaLogFormatJSON:   {},
	}
}

func IsValidLogFormat(format OrlaLogFormat) bool {
	_, ok := ValidLogFormats()[format]
	return ok
}

// OrlaConfig represents the orla configuration, including
// the port to listen on, the timeout for tool executions,
// the log format, and the log level. It also includes Agent Mode configuration (RFC 4).
// Tools are supplied in code or via the API, not from config.
type OrlaConfig struct {
	// Server mode configuration (RFC 1)
	ToolsRegistry *tools.ToolsRegistry `yaml:"-"`                                              // always an empty registry after load; tools come from code/API
	Port          int                  `yaml:"port,omitempty" mapstructure:"port"`             // the port to listen on
	Timeout       int                  `yaml:"timeout,omitempty" mapstructure:"timeout"`       // the timeout for tool executions in seconds
	LogFormat     OrlaLogFormat        `yaml:"log_format,omitempty" mapstructure:"log_format"` // the log format, "pretty" or "json"
	LogLevel      string               `yaml:"log_level,omitempty" mapstructure:"log_level"`   // the log level, "debug", "info", "warn", "error", "fatal"

	// Agent mode configuration (RFC 4)
	LLMBackend         *core.LLMBackend `yaml:"llm_backend,omitempty" mapstructure:"llm_backend"`                 // LLM backend configuration (endpoint, type, api_key)
	Model              string           `yaml:"model,omitempty" mapstructure:"model"`                             // model identifier (e.g., "ollama:ministral-3:8b", "openai:gpt-4")
	MaxToolCalls       int              `yaml:"max_tool_calls,omitempty" mapstructure:"max_tool_calls"`           // maximum tool calls per prompt
	Streaming          bool             `yaml:"streaming,omitempty" mapstructure:"streaming"`                     // enable streaming responses
	OutputFormat       OrlaOutputFormat `yaml:"output_format,omitempty" mapstructure:"output_format"`             // output format: "auto", "rich", or "plain"
	ConfirmDestructive bool             `yaml:"confirm_destructive,omitempty" mapstructure:"confirm_destructive"` // prompt for destructive actions
	DryRun             bool             `yaml:"dry_run,omitempty" mapstructure:"dry_run"`                         // default to non-dry-run mode
	ShowThinking       bool             `yaml:"show_thinking,omitempty" mapstructure:"show_thinking"`             // show thinking trace output (for thinking-capable models)
	ShowToolCalls      bool             `yaml:"show_tool_calls,omitempty" mapstructure:"show_tool_calls"`         // show detailed tool call information
	ShowProgress       bool             `yaml:"show_progress,omitempty" mapstructure:"show_progress"`             // show progress messages even when UI is disabled (e.g., when stdin is piped)
}

// setupViper sets defaults and, if configPath is non-empty, reads that file only.
func setupViper(configPath string) error {
	viper.Reset()
	setViperDefaults()
	if configPath == "" {
		return nil
	}
	viper.SetConfigFile(configPath)
	if err := viper.ReadInConfig(); err != nil {
		return fmt.Errorf("failed to read config file: %w", err)
	}
	return nil
}

// setViperDefaults sets default values in Viper
func setViperDefaults() {
	// Server mode defaults
	viper.SetDefault("port", 8080)
	viper.SetDefault("timeout", 30)
	viper.SetDefault("log_format", "json")
	viper.SetDefault("log_level", "info")

	// Agent mode defaults
	viper.SetDefault("model", DefaultModel)
	viper.SetDefault("auto_start_ollama", true)
	viper.SetDefault("auto_configure_ollama_service", false)
	viper.SetDefault("max_tool_calls", DefaultMaxToolCalls)
	viper.SetDefault("streaming", true)
	viper.SetDefault("output_format", "auto")
	viper.SetDefault("confirm_destructive", true)
	viper.SetDefault("dry_run", false)
	viper.SetDefault("show_thinking", false)
	viper.SetDefault("show_tool_calls", false)
	viper.SetDefault("show_progress", false)
}

// LoadConfig loads configuration from a single file path. If configPath is empty,
// returns defaults only (no file read).
func LoadConfig(configPath string) (*OrlaConfig, error) {
	if err := setupViper(configPath); err != nil {
		return nil, err
	}
	cfg := &OrlaConfig{}
	if err := viper.Unmarshal(cfg); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}
	var configFileDir string
	if configPath != "" {
		configFileDir = filepath.Dir(configPath)
	}
	if err := postProcessConfig(cfg, configFileDir); err != nil {
		return nil, err
	}

	// Validate
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// postProcessConfig sets the tools registry to empty (tools are supplied in code or via the API).
func postProcessConfig(cfg *OrlaConfig, _ string) error {
	cfg.ToolsRegistry = &tools.ToolsRegistry{Tools: make(map[string]*core.ToolManifest)}
	return nil
}

// validateConfig validates the configuration
// Note: This function can be called both:
// 1. After LoadConfig() (viper is configured) - can use viper.IsSet() to detect explicit values
// 2. Directly on structs (viper not configured) - viper.IsSet() may not work, so we apply defaults
func validateConfig(cfg *OrlaConfig) error {
	if cfg.Port < 0 || cfg.Port > 65535 {
		return fmt.Errorf("port must be between 0 and 65535, got %d", cfg.Port)
	}
	if cfg.Timeout < 1 {
		return fmt.Errorf("timeout must be at least 1 second, got %d", cfg.Timeout)
	}

	if cfg.LogFormat != "" && !IsValidLogFormat(cfg.LogFormat) {
		return fmt.Errorf("log_format must be one of: %s, got '%s'", core.JoinMapKeys(ValidLogFormats()), cfg.LogFormat)
	}
	if cfg.LogLevel != "" && !IsValidLogLevel(OrlaLogLevel(cfg.LogLevel)) {
		return fmt.Errorf("log_level must be one of: %s, got '%s'", core.JoinMapKeys(ValidLogLevels()), cfg.LogLevel)
	}

	// Since viper handles defaults, these are values that were explicitly set to empty or zero
	// and need to be validated.
	if cfg.Model == "" {
		return fmt.Errorf("model cannot be empty (was explicitly set to empty string)")
	}

	if cfg.MaxToolCalls < 1 {
		return fmt.Errorf("max_tool_calls must be at least 1, got %d", cfg.MaxToolCalls)
	}

	if !IsValidOutputFormat(cfg.OutputFormat) {
		return fmt.Errorf("output_format must be one of: %s, got '%s'", core.JoinMapKeys(ValidOutputFormats()), cfg.OutputFormat)
	}

	return nil
}
