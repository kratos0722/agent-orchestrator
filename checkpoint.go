package orchestrator

import "sync"

// Checkpoint is an immutable snapshot of a run: the state after a completed
// step, the step index, and the name of the node that should execute next. It
// carries everything needed to resume a run.
type Checkpoint[S any] struct {
	// Step is the number of transitions completed so far.
	Step int
	// State is the state after the most recently completed step.
	State S
	// Next is the node that should execute next when resuming (may be END).
	Next string
}

// Checkpointer persists run snapshots keyed by a thread identifier so a run can
// be resumed later. The engine calls Save after every step. Implementations
// must be safe for concurrent use across threads.
type Checkpointer[S any] interface {
	// Save stores the checkpoint for the given thread, replacing any previous
	// snapshot for that thread.
	Save(threadID string, cp Checkpoint[S]) error
	// Load returns the latest checkpoint for the thread and whether one exists.
	Load(threadID string) (Checkpoint[S], bool)
}

// MemoryCheckpointer is an in-memory, concurrency-safe Checkpointer. It is
// suitable for tests and single-process runs; snapshots are lost when the
// process exits.
type MemoryCheckpointer[S any] struct {
	mu    sync.RWMutex
	store map[string]Checkpoint[S]
}

// NewMemoryCheckpointer creates an empty in-memory checkpointer.
func NewMemoryCheckpointer[S any]() *MemoryCheckpointer[S] {
	return &MemoryCheckpointer[S]{store: make(map[string]Checkpoint[S])}
}

// Save implements Checkpointer.
func (m *MemoryCheckpointer[S]) Save(threadID string, cp Checkpoint[S]) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.store[threadID] = cp
	return nil
}

// Load implements Checkpointer.
func (m *MemoryCheckpointer[S]) Load(threadID string) (Checkpoint[S], bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cp, ok := m.store[threadID]
	return cp, ok
}
