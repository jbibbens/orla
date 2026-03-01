package orla

import (
	"context"
	"encoding/json"
	"fmt"
)

const oneBitPredictorName = "one_bit_predictor"

var oneBitPredictorSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"prediction": map[string]any{
			"type": "boolean",
		},
	},
}

var oneBitPredictorResponseFormat = NewStructuredOutputRequest(oneBitPredictorName, oneBitPredictorSchema)

// OneBitPredictor is a predictor that returns a single bit of information.
type OneBitPredictor struct {
	Agent *Agent
}

// NewOneBitPredictor returns a new OneBitPredictor.
func NewOneBitPredictor(client *OrlaClient, backend *LLMBackend) *OneBitPredictor {
	agent := NewAgent(client)
	stage := NewAgentStage("one_bit_predictor", backend)
	stage.SetResponseFormat(oneBitPredictorResponseFormat)
	agent.SetStage(stage)
	return &OneBitPredictor{Agent: agent}
}

// oneBitResponse is the structured response shape for OneBitPredictor.
type oneBitResponse struct {
	Prediction bool `json:"prediction"`
}

// Predict predicts a single bit of information. prompt is the text sent to the model (e.g. a question or statement to classify).
func (p *OneBitPredictor) Predict(ctx context.Context, prompt string) (bool, error) {
	response, err := p.Agent.Execute(ctx, prompt)
	if err != nil {
		return false, fmt.Errorf("failed to execute prediction: %w", err)
	}
	var out oneBitResponse
	if err := json.Unmarshal([]byte(response.Content), &out); err != nil {
		return false, fmt.Errorf("failed to unmarshal prediction response: %w (response: %s)", err, response.Content)
	}
	return out.Prediction, nil
}
