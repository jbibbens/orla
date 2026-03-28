package serving

import (
	"testing"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSelectBackendByAccuracy_PicksCheapest(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	mgr.AddLLMBackend("expensive", &core.LLMBackend{
		Endpoint: "http://a",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.9,
		CostModel: &core.CostModel{
			InputCostPerMToken:  3.0,
			OutputCostPerMToken: 15.0,
		},
	}, "openai:big")
	mgr.AddLLMBackend("cheap", &core.LLMBackend{
		Endpoint: "http://b",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.7,
		CostModel: &core.CostModel{
			InputCostPerMToken:  0.25,
			OutputCostPerMToken: 1.25,
		},
	}, "openai:small")

	name, err := mgr.SelectBackendByAccuracy(0.5)
	require.NoError(t, err)
	assert.Equal(t, "cheap", name)
}

func TestSelectBackendByAccuracy_FiltersLowQuality(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	mgr.AddLLMBackend("weak", &core.LLMBackend{
		Endpoint: "http://a",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.3,
		CostModel: &core.CostModel{
			InputCostPerMToken:  0.1,
			OutputCostPerMToken: 0.5,
		},
	}, "openai:tiny")
	mgr.AddLLMBackend("strong", &core.LLMBackend{
		Endpoint: "http://b",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.9,
		CostModel: &core.CostModel{
			InputCostPerMToken:  5.0,
			OutputCostPerMToken: 20.0,
		},
	}, "openai:big")

	name, err := mgr.SelectBackendByAccuracy(0.8)
	require.NoError(t, err)
	assert.Equal(t, "strong", name)
}

func TestSelectBackendByAccuracy_NoneQualify(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	mgr.AddLLMBackend("low", &core.LLMBackend{
		Endpoint: "http://a",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.3,
		CostModel: &core.CostModel{
			InputCostPerMToken:  0.1,
			OutputCostPerMToken: 0.5,
		},
	}, "openai:tiny")

	_, err := mgr.SelectBackendByAccuracy(0.9)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backend with quality >= 0.9")
}

func TestSelectBackendByAccuracy_SkipsNoCostModel(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	mgr.AddLLMBackend("no-cost", &core.LLMBackend{
		Endpoint: "http://a",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.9,
	}, "openai:no-cost")
	mgr.AddLLMBackend("with-cost", &core.LLMBackend{
		Endpoint: "http://b",
		Type:     core.LLMInferenceAPITypeOpenAI,
		Quality:  0.7,
		CostModel: &core.CostModel{
			InputCostPerMToken:  1.0,
			OutputCostPerMToken: 5.0,
		},
	}, "openai:costed")

	name, err := mgr.SelectBackendByAccuracy(0.5)
	require.NoError(t, err)
	assert.Equal(t, "with-cost", name)
}

func TestSelectBackendByAccuracy_EmptyBackends(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	_, err := mgr.SelectBackendByAccuracy(0.5)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backend")
}

func TestGetCostModel(t *testing.T) {
	mgr := NewLLMBackendManager(nil)
	cm := &core.CostModel{InputCostPerMToken: 1.0, OutputCostPerMToken: 5.0}
	mgr.AddLLMBackend("b", &core.LLMBackend{
		Endpoint:  "http://a",
		Type:      core.LLMInferenceAPITypeOpenAI,
		CostModel: cm,
	}, "openai:m")

	got := mgr.GetCostModel("b")
	require.NotNil(t, got)
	assert.Equal(t, 1.0, got.InputCostPerMToken)
	assert.Equal(t, 5.0, got.OutputCostPerMToken)

	assert.Nil(t, mgr.GetCostModel("nonexistent"))
}
