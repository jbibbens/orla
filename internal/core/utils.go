package core

import (
	"fmt"
	"maps"
	"slices"
	"strings"
)

// JoinMapKeys joins the keys of a map into a comma-separated string.
// Useful for error messages that need to list valid values.
func JoinMapKeys[T comparable](m map[T]struct{}) string {
	keys := slices.Collect(maps.Keys(m))
	sliceStrings := make([]string, len(keys))
	for i, k := range keys {
		sliceStrings[i] = fmt.Sprintf("%v", k)
	}
	return strings.Join(sliceStrings, ", ")
}
