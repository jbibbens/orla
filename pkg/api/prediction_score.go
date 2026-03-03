package orla

import (
	"context"
	"encoding/json"
	"fmt"
)

const scorePredictorName = "score_predictor"

var scorePredictorSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"score": map[string]any{
			"type":    "integer",
			"minimum": 1,
			"maximum": 5,
		},
	},
	"required": []any{"score"},
}

var scorePredictorResponseFormat = NewStructuredOutputRequest(scorePredictorName, scorePredictorSchema)

// ScorePredictor estimates a score on a 1-5 integer scale using structured
// output. The meaning of the scale is defined by the caller's prompt (e.g.
// complexity, priority, confidence). Use the returned score as a scheduling
// signal or routing decision.
type ScorePredictor struct {
	Stage *Stage
}

// NewScorePredictor returns a new ScorePredictor.
// The predictor uses temperature 0 for deterministic scoring.
func NewScorePredictor(client *OrlaClient, backend *LLMBackend) *ScorePredictor {
	stage := NewStage("score_predictor", backend)
	stage.Client = client
	stage.SetResponseFormat(scorePredictorResponseFormat)
	stage.SetTemperature(0)
	return &ScorePredictor{Stage: stage}
}

type scoreResponse struct {
	Score int `json:"score"`
}

// Predict returns a score from 1 to 5. The semantics of the scale depend on
// the prompt provided by the caller.
func (p *ScorePredictor) Predict(ctx context.Context, prompt string) (int, error) {
	response, err := p.Stage.Execute(ctx, prompt)
	if err != nil {
		return 0, fmt.Errorf("failed to execute score prediction: %w", err)
	}
	var out scoreResponse
	if err := json.Unmarshal([]byte(response.Content), &out); err != nil {
		return 0, fmt.Errorf("failed to unmarshal score response: %w (response: %s)", err, response.Content)
	}
	if out.Score < 1 {
		out.Score = 1
	}
	if out.Score > 5 {
		out.Score = 5
	}
	return out.Score, nil
}
