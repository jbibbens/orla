// Package cost provides helpers for token-based cost estimation.
package cost

import (
	"fmt"

	"github.com/harvard-cns/orla/internal/core"
)

// EstimatedCostUSD computes the estimated cost in USD from token counts and a
// CostModel whose rates are in USD per million tokens. Returns (nil, nil) when
// cm is nil (no cost model configured), distinguishing "unknown" from "$0".
func EstimatedCostUSD(promptTokens, completionTokens int, cm *core.CostModel) (*float64, error) {
	if cm == nil {
		return nil, nil
	}
	if promptTokens < 0 || completionTokens < 0 {
		return nil, fmt.Errorf("negative token count: prompt=%d, completion=%d", promptTokens, completionTokens)
	}
	if cm.InputCostPerMToken < 0 || cm.OutputCostPerMToken < 0 {
		return nil, fmt.Errorf("negative cost rate: input=%.4f, output=%.4f", cm.InputCostPerMToken, cm.OutputCostPerMToken)
	}
	v := (float64(promptTokens)*cm.InputCostPerMToken +
		float64(completionTokens)*cm.OutputCostPerMToken) / 1_000_000
	return &v, nil
}
