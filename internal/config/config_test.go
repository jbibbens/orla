package config

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

const invalidValue = "invalid"

// setupTestConfig creates a temporary directory with an orla.yaml config file
// and changes to that directory. Returns the temp directory and a cleanup function.
func setupTestConfig(t *testing.T) (tmpDir string, cleanup func()) {
	tmpDir = t.TempDir()

	// Create orla.yaml config
	configPath := filepath.Join(tmpDir, "orla.yaml")
	configContent := "port: 8080\n"
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	// Change to temp directory
	originalDir, err := os.Getwd()
	require.NoError(t, err)
	cleanup = func() {
		if chdirErr := os.Chdir(originalDir); chdirErr != nil {
			// Can't use t.Logf in cleanup, so we ignore the error
			_ = chdirErr
		}
	}
	require.NoError(t, os.Chdir(tmpDir))

	return tmpDir, cleanup
}

func TestLoadConfig_Defaults(t *testing.T) {
	// Empty path: defaults only, no file read
	cfg, err := LoadConfig("")
	require.NoError(t, err)
	assert.Equal(t, 8080, cfg.Port)
	assert.Equal(t, 30, cfg.Timeout)
	assert.Equal(t, DefaultModel, cfg.Model)
	assert.Equal(t, 10, cfg.MaxToolCalls)
	assert.Equal(t, OrlaOutputFormatAuto, cfg.OutputFormat)
	assert.Equal(t, false, cfg.DryRun)
}

func TestLoadConfig_WithSpecificPath(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "custom.yaml")
	configContent := "port: 9000\ntimeout: 60\n"
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	cfg, err := LoadConfig(configPath)
	require.NoError(t, err)

	assert.Equal(t, 9000, cfg.Port)
	assert.Equal(t, 60, cfg.Timeout)
}

func TestLoadConfig_InvalidConfigFile(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid.yaml")
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte("invalid: yaml: content: [unclosed"), 0644))

	_, err := LoadConfig(configPath)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to read config file")
}

func TestValidateConfig(t *testing.T) {
	cfg := &OrlaConfig{
		Port:         8080,
		Timeout:      30,
		Model:        DefaultModel,
		MaxToolCalls: DefaultMaxToolCalls,
		OutputFormat: OrlaOutputFormatAuto,
	}

	err := validateConfig(cfg)
	require.NoError(t, err)

	// Test invalid port
	cfg.Port = 70000
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "port must be between 0 and 65535")

	// Test invalid timeout
	cfg.Port = 8080
	cfg.Timeout = 0
	// validateConfig sets default timeout to 30 if 0, so we need to set it to -1 to trigger error
	cfg.Timeout = -1
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout must be at least 1 second")

	// Test invalid log format
	cfg.Timeout = 30
	cfg.LogFormat = invalidValue
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "log_format must be one of")

	// Test invalid log level
	cfg.LogFormat = "json"
	cfg.LogLevel = invalidValue
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "log_level must be one of")

	// Test invalid max_tool_calls
	cfg.LogLevel = "info"
	cfg.MaxToolCalls = -1
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_tool_calls must be at least 1")

	// Test invalid output_format
	cfg.MaxToolCalls = 10
	cfg.OutputFormat = invalidValue
	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "output_format must be one of")
}

func TestPostProcessConfig_EmptyDir(t *testing.T) {
	cfg := &OrlaConfig{}
	err := postProcessConfig(cfg, "")
	require.NoError(t, err)
	assert.NotNil(t, cfg.ToolsRegistry)
	assert.Empty(t, cfg.ToolsRegistry.Tools)
}

func TestPostProcessConfig_WithDir(t *testing.T) {
	cfg := &OrlaConfig{}
	err := postProcessConfig(cfg, t.TempDir())
	require.NoError(t, err)
	assert.NotNil(t, cfg.ToolsRegistry)
	assert.Empty(t, cfg.ToolsRegistry.Tools)
}

func TestLoadConfig_WithYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "test.yaml")
	configContent := `
port: 9000
timeout: 60
log_format: pretty
log_level: debug
model: openai:gpt-4
max_tool_calls: 20
streaming: false
output_format: rich
confirm_destructive: false
dry_run: true
`
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	cfg, err := LoadConfig(configPath)
	require.NoError(t, err)

	assert.Equal(t, 9000, cfg.Port)
	assert.Equal(t, 60, cfg.Timeout)
	assert.Equal(t, OrlaLogFormatPretty, cfg.LogFormat)
	assert.Equal(t, "debug", cfg.LogLevel)
	assert.Equal(t, "openai:gpt-4", cfg.Model)
	assert.Equal(t, 20, cfg.MaxToolCalls)
	assert.Equal(t, false, cfg.Streaming)
	assert.Equal(t, OrlaOutputFormatRich, cfg.OutputFormat)
	assert.Equal(t, false, cfg.ConfirmDestructive)
	assert.Equal(t, true, cfg.DryRun)
}

func TestLoadConfig_InvalidYAML(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "invalid.yaml")
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte("invalid: yaml: [unclosed"), 0644))

	_, err := LoadConfig(configPath)
	require.Error(t, err)
}

func TestValidateConfig_Defaults(t *testing.T) {
	// Test that validateConfig errors on empty/zero values when called directly
	// (since Viper isn't configured, it can't distinguish between unset and explicitly empty)
	cfg := &OrlaConfig{}
	err := validateConfig(cfg)
	require.Error(t, err)

	// Load a config file with only port set; other fields get defaults
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "orla.yaml")
	configContent := "port: 9000\n"
	// #nosec G306 -- test file permissions are acceptable for temporary test files
	require.NoError(t, os.WriteFile(configPath, []byte(configContent), 0644))

	cfg2, err := LoadConfig(configPath)
	require.NoError(t, err)
	assert.Equal(t, 9000, cfg2.Port)
	assert.Equal(t, 30, cfg2.Timeout)
	assert.Equal(t, DefaultModel, cfg2.Model)
	assert.Equal(t, DefaultMaxToolCalls, cfg2.MaxToolCalls)
	assert.Equal(t, OrlaOutputFormatAuto, cfg2.OutputFormat)
}

func TestValidateConfig_BadValues(t *testing.T) {
	cfg := &OrlaConfig{
		Port: 70000,
	}

	err := validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "port must be between 0 and 65535")

	cfg = &OrlaConfig{
		Model:        "test",
		MaxToolCalls: 10,
		Timeout:      0,
		OutputFormat: "auto",
	}

	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "timeout must be at least 1 second")

	cfg.Timeout = 30
	cfg.Model = ""

	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model cannot be empty")

	cfg.Model = "test"
	cfg.MaxToolCalls = 0

	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_tool_calls must be at least 1")

	cfg.MaxToolCalls = 10
	cfg.OutputFormat = "invalid"

	err = validateConfig(cfg)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "output_format must be one of")
}

func TestConfig_MarshalUnmarshal(t *testing.T) {
	cfg := &OrlaConfig{
		Port:    9000,
		Timeout: 60,
		Model:   "openai:gpt-4",
	}

	data, err := yaml.Marshal(cfg)
	require.NoError(t, err)

	var cfg2 OrlaConfig
	err = yaml.Unmarshal(data, &cfg2)
	require.NoError(t, err)

	assert.Equal(t, cfg.Port, cfg2.Port)
	assert.Equal(t, cfg.Timeout, cfg2.Timeout)
	assert.Equal(t, cfg.Model, cfg2.Model)
}

// TestLoadConfig_ExplicitEmptyValues tests that explicit empty/zero values in config files
// correctly raise validation errors.
func TestLoadConfig_ExplicitEmptyValues(t *testing.T) {
	tmpDir := t.TempDir()

	// Explicit empty model should error
	configPath1 := filepath.Join(tmpDir, "empty_model.yaml")
	require.NoError(t, os.WriteFile(configPath1, []byte(`model: ""`), 0644))
	_, err := LoadConfig(configPath1)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model cannot be empty")

	// Explicit zero max_tool_calls should error
	configPath2 := filepath.Join(tmpDir, "zero_tools.yaml")
	require.NoError(t, os.WriteFile(configPath2, []byte("max_tool_calls: 0"), 0644))
	_, err = LoadConfig(configPath2)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_tool_calls must be at least 1")

	// Only port set; other fields get defaults
	configPath3 := filepath.Join(tmpDir, "partial.yaml")
	require.NoError(t, os.WriteFile(configPath3, []byte("port: 9000"), 0644))
	cfg, err := LoadConfig(configPath3)
	require.NoError(t, err)
	assert.Equal(t, DefaultModel, cfg.Model)
	assert.Equal(t, DefaultMaxToolCalls, cfg.MaxToolCalls)
	assert.Equal(t, 9000, cfg.Port)
}
