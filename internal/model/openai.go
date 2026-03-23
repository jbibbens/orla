package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/sashabaranov/go-openai"
	"go.uber.org/zap"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const defaultStreamBufferSize = 255

// NOTE(jadidbourbaki): I found the official OpenAI SDK client to be not very user friendly and kind of too bloated for our use case.
// So we're going to use the go-openai library to make requests to the OpenAI-compatible API.

// OpenAIProvider implements the Provider interface for OpenAI-compatible APIs.
// This provider is intended to work with any server that implements the OpenAI Chat Completions API format
// such as LM Studio, vLLM, and Ollama. For Ollama, use endpoint http://host:11434/v1 [1].
// [1] https://docs.ollama.com/api/openai-compatibility
type OpenAIProvider struct {
	modelName  string
	client     *openai.Client
	llmBackend *core.LLMBackend
}

// NewOpenAIProvider creates a new OpenAI-compatible provider.
// This works with any server that implements the OpenAI Chat Completions API format.
func NewOpenAIProvider(modelName string, llmBackend *core.LLMBackend) (*OpenAIProvider, error) {
	baseURL, apiKey, err := getOpenAICompatibleEndpoint(llmBackend)
	if err != nil {
		return nil, fmt.Errorf("failed to get OpenAI-compatible endpoint: %w", err)
	}

	// Configure the OpenAI client with custom base URL
	config := openai.DefaultConfig(apiKey)
	config.BaseURL = baseURL

	// Create OpenAI client
	client := openai.NewClientWithConfig(config)

	return &OpenAIProvider{
		modelName:  modelName,
		client:     client,
		llmBackend: llmBackend,
	}, nil
}

// getOpenAICompatibleEndpoint determines the OpenAI-compatible endpoint URL and API key.
func getOpenAICompatibleEndpoint(llmBackend *core.LLMBackend) (string, string, error) {
	if llmBackend == nil {
		return "", "", fmt.Errorf("llm_backend configuration is required for OpenAI-compatible API")
	}

	if llmBackend.Endpoint == "" {
		return "", "", fmt.Errorf("llm_backend.endpoint is required")
	}

	if llmBackend.Type == "" {
		return "", "", fmt.Errorf("llm_backend.type is required if llm_backend is set")
	}

	switch llmBackend.Type {
	case core.LLMInferenceAPITypeOpenAI, core.LLMInferenceAPITypeSGLang:
		// Both expose an OpenAI-compatible /v1/chat/completions endpoint.
	default:
		return "", "", fmt.Errorf("[BUG] llm_backend.type must be %s or %s, got '%s': we should not be using this function for non-openai-compatible inference servers", core.LLMInferenceAPITypeOpenAI, core.LLMInferenceAPITypeSGLang, llmBackend.Type)
	}

	if llmBackend.APIKeyEnvVar == "" {
		zap.L().Debug("llm_backend.api_key_env_var is not set, skipping authentication for OpenAI-compatible API",
			zap.String("llm_backend.endpoint", llmBackend.Endpoint))
		return llmBackend.Endpoint, "", nil
	}

	apiKey := core.GetEnv(llmBackend.APIKeyEnvVar)
	if apiKey == "" {
		return "", "", fmt.Errorf("API key is required for OpenAI-compatible API at %s. Configure llm_backend.api_key_env_var", llmBackend.Endpoint)
	}

	return llmBackend.Endpoint, apiKey, nil
}

// Name returns the provider name
func (p *OpenAIProvider) Name() string {
	return "openai"
}

// EnsureReady is a no-op for the OpenAI-compatible provider.
func (p *OpenAIProvider) EnsureReady(ctx context.Context) error {
	return nil
}

// Chat sends a chat request to the OpenAI-compatible API. This works with any server implementing the OpenAI Chat Completions API format.
func (p *OpenAIProvider) Chat(ctx context.Context, messages []Message, tools []*mcp.Tool, opts InferenceOptions) (*Response, <-chan StreamEvent, error) {
	stream := opts.Stream
	// Ensure the OpenAI-compatible API is ready. This is a no-op for the OpenAI-compatible provider, but
	// might as well check anyway in case we add health checks in the future.
	readyErr := p.EnsureReady(ctx)
	if readyErr != nil {
		return nil, nil, fmt.Errorf("failed to ensure OpenAI-compatible API is ready: %w", readyErr)
	}

	// Convert messages to go-openai format
	openAIMessages := make([]openai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		openAIMessages[i] = convertMessageToOpenAI(msg)
	}

	// Build request
	req := openai.ChatCompletionRequest{
		Model:    p.modelName,
		Messages: openAIMessages,
		Stream:   stream,
	}
	if opts.MaxTokens != nil {
		req.MaxTokens = *opts.MaxTokens
	}
	if opts.Temperature != nil {
		req.Temperature = float32(*opts.Temperature)
	}
	if opts.TopP != nil {
		req.TopP = float32(*opts.TopP)
	}
	if len(opts.ChatTemplateKwargs) > 0 {
		req.ChatTemplateKwargs = opts.ChatTemplateKwargs
	}
	if opts.ReasoningEffort != "" {
		req.ReasoningEffort = opts.ReasoningEffort
	}

	// Add tools if provided
	if len(tools) > 0 {
		openAITools, err := convertToolsToOpenAIFormat(tools)
		if err != nil {
			return nil, nil, err
		}
		req.Tools = openAITools
		req.ToolChoice = "auto" // Let the model decide when to use tools
	}

	// Structured output (OpenAI, vLLM, SGLang)
	if opts.ResponseFormat != nil {
		req.ResponseFormat = &openai.ChatCompletionResponseFormat{
			Type: openai.ChatCompletionResponseFormatTypeJSONSchema,
			JSONSchema: &openai.ChatCompletionResponseFormatJSONSchema{
				Name:   opts.ResponseFormat.Name,
				Strict: opts.ResponseFormat.Strict,
				Schema: opts.ResponseFormat.Schema,
			},
		}
	}

	if stream {
		return p.handleStreamingChat(ctx, req)
	}

	return p.handleNonStreamingChat(ctx, req)
}

// handleNonStreamingChat handles non-streaming chat requests.
func (p *OpenAIProvider) handleNonStreamingChat(ctx context.Context, req openai.ChatCompletionRequest) (*Response, <-chan StreamEvent, error) {
	completion, err := p.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("OpenAI-compatible API error: %w", err)
	}

	if len(completion.Choices) == 0 {
		return nil, nil, fmt.Errorf("OpenAI-compatible API returned no choices")
	}

	choice := completion.Choices[0]
	response := &Response{
		Content:  choice.Message.Content,
		Thinking: choice.Message.ReasoningContent,
		Metrics: &ResponseMetrics{
			PromptTokens:     completion.Usage.PromptTokens,
			CompletionTokens: completion.Usage.CompletionTokens,
		},
	}

	// Parse tool calls if present
	if len(choice.Message.ToolCalls) > 0 {
		toolCalls, err := convertOpenAIToolCalls(choice.Message.ToolCalls)
		if err != nil {
			return nil, nil, err
		}
		response.ToolCalls = toolCalls
		zap.L().Debug("Parsed tool calls", zap.Int("count", len(toolCalls)))
	}

	return response, nil, nil
}

// handleStreamingChat handles streaming chat requests. It records TTFT/TPOT and sets response.Metrics when the stream completes.
func (p *OpenAIProvider) handleStreamingChat(ctx context.Context, req openai.ChatCompletionRequest) (*Response, <-chan StreamEvent, error) {
	req.StreamOptions = &openai.StreamOptions{IncludeUsage: true}
	stream, err := p.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		return nil, nil, fmt.Errorf("OpenAI-compatible API error: %w", err)
	}

	ch := make(chan StreamEvent, defaultStreamBufferSize)
	response := &Response{
		Content:   "",
		Thinking:  "",
		ToolCalls: []ToolCallWithID{},
	}

	go func() {
		defer close(ch)
		defer core.LogDeferredError(stream.Close)

		start := time.Now()
		var firstContentAt, lastContentAt time.Time
		var promptTokens, completionTokens int
		var accumulatedToolCalls []openai.ToolCall

		for {
			chunk, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				zap.L().Error("Error reading stream", zap.Error(err))
				break
			}

			if chunk.Usage != nil {
				if chunk.Usage.PromptTokens > 0 {
					promptTokens = chunk.Usage.PromptTokens
				}
				if chunk.Usage.CompletionTokens > 0 {
					completionTokens = chunk.Usage.CompletionTokens
				}
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			delta := chunk.Choices[0].Delta

			// Accumulate content
			if delta.Content != "" {
				response.Content += delta.Content
				now := time.Now()
				if firstContentAt.IsZero() {
					firstContentAt = now
				}
				lastContentAt = now
				ch <- &ContentEvent{Content: delta.Content}
			}

			// Accumulate reasoning/thinking content (Ollama, DeepSeek, etc.)
			if delta.ReasoningContent != "" {
				response.Thinking += delta.ReasoningContent
				ch <- &ThinkingEvent{Content: delta.ReasoningContent}
			}

			// Accumulate tool calls if present
			if len(delta.ToolCalls) > 0 {
				// OpenAI sends tool calls incrementally, we need to accumulate them
				for _, toolCallDelta := range delta.ToolCalls {
					if toolCallDelta.Index == nil {
						continue
					}

					// Index is a pointer, so we need to dereference it
					index := *toolCallDelta.Index
					if index < 0 {
						zap.L().Error("Invalid tool call index (negative)", zap.Int("index", index), zap.Any("toolCallDelta", toolCallDelta))
						continue
					}

					// Expand slice if needed for this index
					if index >= len(accumulatedToolCalls) {
						for len(accumulatedToolCalls) <= index {
							idx := len(accumulatedToolCalls)
							accumulatedToolCalls = append(accumulatedToolCalls, openai.ToolCall{
								Index: &idx,
								ID:    "",
								Type:  openai.ToolTypeFunction,
								Function: openai.FunctionCall{
									Name:      "",
									Arguments: "",
								},
							})
						}
					}

					// Accumulate tool call data
					if toolCallDelta.ID != "" {
						accumulatedToolCalls[index].ID = toolCallDelta.ID
					}
					if toolCallDelta.Type != "" {
						accumulatedToolCalls[index].Type = toolCallDelta.Type
					}
					if toolCallDelta.Function.Name != "" {
						accumulatedToolCalls[index].Function.Name = toolCallDelta.Function.Name
					}
					if toolCallDelta.Function.Arguments != "" {
						accumulatedToolCalls[index].Function.Arguments += toolCallDelta.Function.Arguments
					}
				}
			}
		}

		// Convert accumulated tool calls to our format (after stream completes)
		if len(accumulatedToolCalls) > 0 {
			toolCalls, err := convertOpenAIToolCalls(accumulatedToolCalls)
			if err != nil {
				zap.L().Error("Failed to convert tool calls from stream", zap.Error(err))
			}
			response.ToolCalls = toolCalls
		}

		// Metrics: timing (streaming only) + token counts (always)
		response.Metrics = &ResponseMetrics{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
		}
		if !firstContentAt.IsZero() {
			response.Metrics.TTFTMs = firstContentAt.Sub(start).Milliseconds()
		}
		if completionTokens > 0 && !firstContentAt.IsZero() && !lastContentAt.IsZero() {
			decodeMs := lastContentAt.Sub(firstContentAt).Milliseconds()
			response.Metrics.TPOTMs = decodeMs / int64(completionTokens)
			if response.Metrics.TPOTMs == 0 && decodeMs > 0 {
				response.Metrics.TPOTMs = 1
			}
		}
	}()

	return response, ch, nil
}

// convertMessageToOpenAI converts our Message type to go-openai format
func convertMessageToOpenAI(msg Message) openai.ChatCompletionMessage {
	result := openai.ChatCompletionMessage{
		Role:    string(msg.Role),
		Content: msg.Content,
	}

	// Handle tool messages (they need a ToolCallID for OpenAI API)
	if msg.Role == MessageRoleTool {
		result.Role = openai.ChatMessageRoleTool
		if msg.ToolCallID != "" {
			result.ToolCallID = msg.ToolCallID
		} else {
			zap.L().Warn("Tool message missing ToolCallID",
				zap.String("tool_name", msg.ToolName))
		}
	}

	// Carry tool calls on assistant messages so the conversation history is
	// well-formed for backends that require it (e.g. Ollama).
	if msg.Role == MessageRoleAssistant && len(msg.ToolCalls) > 0 {
		for _, tc := range msg.ToolCalls {
			argsJSON, err := json.Marshal(tc.McpCallToolParams.Arguments)
			if err != nil {
				zap.L().Warn("Failed to marshal tool call arguments", zap.Error(err))
				continue
			}
			result.ToolCalls = append(result.ToolCalls, openai.ToolCall{
				ID:   tc.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      tc.McpCallToolParams.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	return result
}

// normalizeSchemaToMap converts the InputSchema of a tool to a map[string]any, or returns an error if it fails
func normalizeSchemaToMap(tool *mcp.Tool) (map[string]any, error) {
	if tool.InputSchema == nil {
		return nil, fmt.Errorf("tool input schema is nil for tool %s", tool.Name)
	}

	schemaMap, ok := tool.InputSchema.(map[string]any)
	if ok {
		return schemaMap, nil
	}

	var schemaBytes []byte

	switch s := tool.InputSchema.(type) {
	case json.RawMessage:
		schemaBytes = s
	case []byte:
		schemaBytes = s
	default:
		b, jsonErr := json.Marshal(tool.InputSchema)
		if jsonErr != nil {
			return nil, fmt.Errorf("failed to marshal Tool InputSchema for tool %s: %w", tool.Name, jsonErr)
		}
		schemaBytes = b
	}

	err := json.Unmarshal(schemaBytes, &schemaMap)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal Tool InputSchema for tool %s into map: %w", tool.Name, err)
	}

	return schemaMap, nil
}

// convertToolsToOpenAIFormat converts mcp.Tool slice to go-openai format
func convertToolsToOpenAIFormat(tools []*mcp.Tool) ([]openai.Tool, error) {
	openAITools := make([]openai.Tool, len(tools))

	for i, tool := range tools {
		schemaMap, err := normalizeSchemaToMap(tool)
		if err != nil {
			return nil, fmt.Errorf("failed to normalize tool input schema for tool %s: %w", tool.Name, err)
		}

		openAITools[i] = openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  schemaMap,
			},
		}
	}
	return openAITools, nil
}

// convertOpenAiToolCall converts a go-openai tool call to orla's format.
func convertOpenAiToolCall(call openai.ToolCall, idx int) (*ToolCallWithID, error) {
	if call.ID == "" {
		return nil, fmt.Errorf("tool call at index %d has empty id (tool %s); backend must provide an id for result matching", idx, call.Function.Name)
	}

	args := make(map[string]any)
	if call.Function.Arguments != "" {
		if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tool call arguments for tool %s: %w", call.Function.Name, err)
		}
	}

	return &ToolCallWithID{
		ID: call.ID,
		McpCallToolParams: mcp.CallToolParams{
			Name:      call.Function.Name,
			Arguments: args,
		},
	}, nil
}

// convertOpenAIToolCalls converts go-openai tool calls to orla's format (from non-streaming response)
func convertOpenAIToolCalls(openAICalls []openai.ToolCall) ([]ToolCallWithID, error) {
	toolCalls := make([]ToolCallWithID, len(openAICalls))

	for i, call := range openAICalls {
		toolCall, err := convertOpenAiToolCall(call, i)
		if err != nil {
			return nil, fmt.Errorf("failed to convert tool call for tool %s: %w", call.Function.Name, err)
		}
		toolCalls[i] = *toolCall
	}

	return toolCalls, nil
}
