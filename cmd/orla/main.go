// Command orla is the daemon entry point. It delegates to the cobra
// command tree in cmd/orla/commands.
package main

import (
	"fmt"
	"os"

	"github.com/harvard-cns/orla/cmd/orla/commands"
)

func main() {
	if err := commands.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "orla: %v\n", err)
		os.Exit(1)
	}
}
