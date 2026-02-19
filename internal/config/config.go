// Package config provides configuration management for Orla: load from a
// single config file path, or use defaults when path is empty.
package config

import (
	"fmt"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/spf13/viper"
	"go.uber.org/zap"
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

var validLogLevels = map[OrlaLogLevel]struct{}{
	OrlaLogLevelDebug: {},
	OrlaLogLevelInfo:  {},
	OrlaLogLevelWarn:  {},
	OrlaLogLevelError: {},
	OrlaLogLevelFatal: {},
}

func IsValidMapKey[K comparable, V any](m map[K]V, key K) bool {
	_, ok := m[key]
	return ok
}

// OrlaOutputFormat represents the output format for agent mode
type OrlaOutputFormat string

const (
	OrlaOutputFormatAuto  OrlaOutputFormat = "auto"
	OrlaOutputFormatRich  OrlaOutputFormat = "rich"
	OrlaOutputFormatPlain OrlaOutputFormat = "plain"
)

var validOutputFormats = map[OrlaOutputFormat]struct{}{
	OrlaOutputFormatAuto:  {},
	OrlaOutputFormatRich:  {},
	OrlaOutputFormatPlain: {},
}

type OrlaLogFormat string

const (
	OrlaLogFormatPretty OrlaLogFormat = "pretty"
	OrlaLogFormatJSON   OrlaLogFormat = "json"
)

var validLogFormats = map[OrlaLogFormat]struct{}{
	OrlaLogFormatPretty: {},
	OrlaLogFormatJSON:   {},
}

// OrlaConfig represents the orla configuration used in both server and agent mode.
type OrlaConfig struct {
	// Server specific configuration
	Port          int  `yaml:"port,omitempty" mapstructure:"port"`                       // the port to listen on
	ToolTimeout   int  `yaml:"tool_timeout,omitempty" mapstructure:"tool_timeout"`       // the timeout for tool executions in seconds
	MaxToolCalls  int  `yaml:"max_tool_calls,omitempty" mapstructure:"max_tool_calls"`   // maximum tool calls per prompt
	ShowToolCalls bool `yaml:"show_tool_calls,omitempty" mapstructure:"show_tool_calls"` // show detailed tool call information

	// Common configuration used by both server and agent mode
	LogFormat    OrlaLogFormat    `yaml:"log_format,omitempty" mapstructure:"log_format"`       // the log format, "pretty" or "json"
	LogLevel     string           `yaml:"log_level,omitempty" mapstructure:"log_level"`         // the log level, "debug", "info", "warn", "error", "fatal"
	LLMBackend   *core.LLMBackend `yaml:"llm_backend,omitempty" mapstructure:"llm_backend"`     // LLM backend configuration (endpoint, type, api_key)
	Model        string           `yaml:"model,omitempty" mapstructure:"model"`                 // model identifier (e.g., "ollama:ministral-3:8b", "openai:gpt-4")
	Streaming    bool             `yaml:"streaming,omitempty" mapstructure:"streaming"`         // enable streaming responses
	OutputFormat OrlaOutputFormat `yaml:"output_format,omitempty" mapstructure:"output_format"` // output format: "auto", "rich", or "plain"
	ShowThinking bool             `yaml:"show_thinking,omitempty" mapstructure:"show_thinking"` // show thinking trace output (for thinking-capable models)
	ShowProgress bool             `yaml:"show_progress,omitempty" mapstructure:"show_progress"` // show progress messages even when UI is disabled (e.g., when stdin is piped)
}

// setupViper sets defaults and, if configPath is non-empty, reads that file only.
func setupViper(configPath string) error {
	viper.Reset()
	setViperDefaults()

	if configPath == "" {
		zap.L().Debug("no config path provided, using defaults")
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
	viper.SetDefault("port", 8080)
	viper.SetDefault("tool_timeout", 30)
	viper.SetDefault("log_format", "json")
	viper.SetDefault("log_level", "info")
	viper.SetDefault("model", DefaultModel)
	viper.SetDefault("auto_start_ollama", true)
	viper.SetDefault("auto_configure_ollama_service", false)
	viper.SetDefault("max_tool_calls", DefaultMaxToolCalls)
	viper.SetDefault("streaming", true)
	viper.SetDefault("output_format", "auto")
	viper.SetDefault("show_thinking", false)
	viper.SetDefault("show_tool_calls", false)
	viper.SetDefault("show_progress", false)
}

// LoadConfig loads configuration from a single file path. If configPath is empty,
// returns defaults only (no file read).
func LoadConfig(configPath string) (*OrlaConfig, error) {
	setupViperErr := setupViper(configPath)
	if setupViperErr != nil {
		return nil, setupViperErr
	}

	cfg := &OrlaConfig{}
	unmarshalErr := viper.Unmarshal(cfg)
	if unmarshalErr != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", unmarshalErr)
	}

	validateConfigErr := validateConfig(cfg)
	if validateConfigErr != nil {
		return nil, validateConfigErr
	}

	return cfg, nil
}

func validateConfig(cfg *OrlaConfig) error {
	if cfg.Port < 0 || cfg.Port > 65535 {
		return fmt.Errorf("port must be between 0 and 65535, got %d", cfg.Port)
	}

	if cfg.ToolTimeout < 1 {
		return fmt.Errorf("timeout must be at least 1 second, got %d", cfg.ToolTimeout)
	}

	if !IsValidMapKey(validLogFormats, cfg.LogFormat) {
		return fmt.Errorf("log_format must be one of: %s, got '%s'", core.JoinMapKeys(validLogFormats), cfg.LogFormat)
	}

	if !IsValidMapKey(validLogLevels, OrlaLogLevel(cfg.LogLevel)) {
		return fmt.Errorf("log_level must be one of: %s, got '%s'", core.JoinMapKeys(validLogLevels), cfg.LogLevel)
	}

	if cfg.MaxToolCalls < 1 {
		return fmt.Errorf("max_tool_calls must be at least 1, got %d", cfg.MaxToolCalls)
	}

	if !IsValidMapKey(validOutputFormats, cfg.OutputFormat) {
		return fmt.Errorf("output_format must be one of: %s, got '%s'", core.JoinMapKeys(validOutputFormats), cfg.OutputFormat)
	}

	return nil
}
