package orchestrator

import "errors"

var (
	// ErrMaxSteps is returned when a run exceeds the configured step budget,
	// which guards against infinite cycles in the graph.
	ErrMaxSteps = errors.New("orchestrator: max steps exceeded")

	// ErrNoCheckpoint is returned by Resume when no checkpoint exists for the
	// requested thread.
	ErrNoCheckpoint = errors.New("orchestrator: no checkpoint for thread")

	// ErrNoCheckpointer is returned by Resume when the engine was compiled
	// without a checkpointer.
	ErrNoCheckpointer = errors.New("orchestrator: engine has no checkpointer")
)
