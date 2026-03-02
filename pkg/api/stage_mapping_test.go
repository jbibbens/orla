package orla

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestExplicitStageMapping_AllAssigned(t *testing.T) {
	backend := &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"}
	s1 := NewStage("s1", backend)
	s1.SetMaxTokens(100)
	s2 := NewStage("s2", backend)

	m := &ExplicitStageMapping{}
	out, err := m.Map(&StageMappingInput{
		Stages:   []*Stage{s1, s2},
		Backends: []*LLMBackend{backend},
	})
	require.NoError(t, err)
	require.Len(t, out.Assignments, 2)
	assert.Equal(t, backend, out.Assignments[s1.ID].Backend)
	assert.Equal(t, backend, out.Assignments[s2.ID].Backend)
	require.NotNil(t, out.Assignments[s1.ID].MaxTokens)
	assert.Equal(t, 100, *out.Assignments[s1.ID].MaxTokens)
}

func TestExplicitStageMapping_MissingBackend(t *testing.T) {
	s := &Stage{ID: "s1", Name: "s1", Tools: make(map[string]*Tool)}
	m := &ExplicitStageMapping{}
	_, err := m.Map(&StageMappingInput{Stages: []*Stage{s}})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no backend assigned")
}

func TestExplicitStageMapping_NilInput(t *testing.T) {
	m := &ExplicitStageMapping{}
	_, err := m.Map(nil)
	require.Error(t, err)
}

func TestApplyStageMappingOutput(t *testing.T) {
	backendOld := &LLMBackend{Name: "old", Endpoint: "http://old", Type: "openai", ModelID: "m"}
	backendNew := &LLMBackend{Name: "new", Endpoint: "http://new", Type: "openai", ModelID: "m2"}
	s := NewStage("s", backendOld)
	newTokens := 512

	err := ApplyStageMappingOutput([]*Stage{s}, &StageMappingOutput{
		Assignments: map[string]*StageAssignment{
			s.ID: {Backend: backendNew, MaxTokens: &newTokens},
		},
	})
	require.NoError(t, err)
	assert.Equal(t, backendNew, s.Backend)
	require.NotNil(t, s.MaxTokens)
	assert.Equal(t, 512, *s.MaxTokens)
}
