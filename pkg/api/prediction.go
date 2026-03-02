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
	Stage *Stage
}

// NewOneBitPredictor returns a new OneBitPredictor.
// The predictor uses temperature 0 for deterministic classification.
func NewOneBitPredictor(client *OrlaClient, backend *LLMBackend) *OneBitPredictor {
	stage := NewStage("one_bit_predictor", backend)
	stage.Client = client
	stage.SetResponseFormat(oneBitPredictorResponseFormat)
	stage.SetTemperature(0)
	return &OneBitPredictor{Stage: stage}
}

// oneBitResponse is the structured response shape for OneBitPredictor.
type oneBitResponse struct {
	Prediction bool `json:"prediction"`
}

// Predict predicts a single bit of information. prompt is the text sent to the model.
func (p *OneBitPredictor) Predict(ctx context.Context, prompt string) (bool, error) {
	response, err := p.Stage.Execute(ctx, prompt)
	if err != nil {
		return false, fmt.Errorf("failed to execute prediction: %w", err)
	}
	var out oneBitResponse
	if err := json.Unmarshal([]byte(response.Content), &out); err != nil {
		return false, fmt.Errorf("failed to unmarshal prediction response: %w (response: %s)", err, response.Content)
	}
	return out.Prediction, nil
}
