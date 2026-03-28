package cost_test

import (
	"testing"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/harvard-cns/orla/internal/serving/cost"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEstimatedCostUSD(t *testing.T) {
	tests := []struct {
		name             string
		promptTokens     int
		completionTokens int
		cm               *core.CostModel
		wantNil          bool
		want             float64
	}{
		{
			name:             "nil cost model returns nil",
			promptTokens:     1000,
			completionTokens: 500,
			cm:               nil,
			wantNil:          true,
		},
		{
			name:             "zero tokens returns zero",
			promptTokens:     0,
			completionTokens: 0,
			cm:               &core.CostModel{InputCostPerMToken: 3.0, OutputCostPerMToken: 15.0},
			want:             0,
		},
		{
			name:             "known values",
			promptTokens:     1_000_000,
			completionTokens: 1_000_000,
			cm:               &core.CostModel{InputCostPerMToken: 3.0, OutputCostPerMToken: 15.0},
			want:             18.0,
		},
		{
			name:             "fractional tokens",
			promptTokens:     150,
			completionTokens: 12,
			cm:               &core.CostModel{InputCostPerMToken: 0.25, OutputCostPerMToken: 1.25},
			want:             (150*0.25 + 12*1.25) / 1_000_000,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cost.EstimatedCostUSD(tt.promptTokens, tt.completionTokens, tt.cm)
			require.NoError(t, err)
			if tt.wantNil {
				assert.Nil(t, got)
				return
			}
			require.NotNil(t, got)
			assert.InDelta(t, tt.want, *got, 1e-12)
		})
	}
}

func TestEstimatedCostUSD_NegativeTokens(t *testing.T) {
	cm := &core.CostModel{InputCostPerMToken: 3.0, OutputCostPerMToken: 15.0}

	got, err := cost.EstimatedCostUSD(-1, 100, cm)
	assert.Nil(t, got)
	assert.ErrorContains(t, err, "negative token count")

	got, err = cost.EstimatedCostUSD(100, -1, cm)
	assert.Nil(t, got)
	assert.ErrorContains(t, err, "negative token count")
}

func TestEstimatedCostUSD_NegativeRates(t *testing.T) {
	got, err := cost.EstimatedCostUSD(100, 100, &core.CostModel{InputCostPerMToken: -1, OutputCostPerMToken: 5})
	assert.Nil(t, got)
	assert.ErrorContains(t, err, "negative cost rate")

	got, err = cost.EstimatedCostUSD(100, 100, &core.CostModel{InputCostPerMToken: 5, OutputCostPerMToken: -1})
	assert.Nil(t, got)
	assert.ErrorContains(t, err, "negative cost rate")
}
