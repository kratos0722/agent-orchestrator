package orchestrator

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// Phase identifies whether an observability Event marks a node being entered or
// exited.
type Phase string

const (
	// PhaseEnter is emitted immediately before a node's first attempt.
	PhaseEnter Phase = "enter"
	// PhaseExit is emitted after a node finishes (success or terminal error),
	// with the wall-clock duration across all attempts.
	PhaseExit Phase = "exit"
)

// Event is passed to the observability Hook on each node enter and exit.
type Event struct {
	Node     string
	Step     int
	Phase    Phase
	Duration time.Duration // populated on PhaseExit
	Err      error         // terminal error on PhaseExit, if any
}

// Hook is an optional callback invoked on every node enter and exit. It must
// not block for long and must be safe for concurrent use, because parallel
// branches emit events from multiple goroutines.
type Hook func(Event)

// Engine is a compiled, runnable graph. It is safe to Run concurrently for
// distinct thread identifiers.
type Engine[S any] struct {
	g            *Graph[S]
	checkpointer Checkpointer[S]
	hook         Hook
	maxSteps     int
}

// Option configures an Engine at Compile time.
type Option[S any] func(*Engine[S])

// WithCheckpointer makes the engine save a checkpoint after every step and
// enables Resume.
func WithCheckpointer[S any](cp Checkpointer[S]) Option[S] {
	return func(e *Engine[S]) { e.checkpointer = cp }
}

// WithHook installs an observability callback.
func WithHook[S any](h Hook) Option[S] {
	return func(e *Engine[S]) { e.hook = h }
}

// WithMaxSteps sets the global step budget that guards against infinite cycles.
// A value below 1 leaves the default in place.
func WithMaxSteps[S any](n int) Option[S] {
	return func(e *Engine[S]) {
		if n >= 1 {
			e.maxSteps = n
		}
	}
}

// defaultMaxSteps is a generous ceiling used when the caller does not set one.
const defaultMaxSteps = 1_000_000

// Compile validates the graph and returns a runnable engine. It fails if the
// graph was built with errors, if START has no outgoing edge, or if any edge
// references an unknown node.
func (g *Graph[S]) Compile(opts ...Option[S]) (*Engine[S], error) {
	if len(g.errs) > 0 {
		return nil, fmt.Errorf("orchestrator: invalid graph: %w", joinErrs(g.errs))
	}
	if _, ok := g.trans[START]; !ok {
		return nil, fmt.Errorf("orchestrator: no edge from START")
	}
	if err := g.validateTargets(); err != nil {
		return nil, err
	}
	e := &Engine[S]{g: g, maxSteps: defaultMaxSteps}
	for _, opt := range opts {
		opt(e)
	}
	return e, nil
}

func (g *Graph[S]) known(name string) bool {
	if name == START || name == END {
		return true
	}
	_, ok := g.nodes[name]
	return ok
}

func (g *Graph[S]) validateTargets() error {
	for from, tr := range g.trans {
		if !g.known(from) && from != START {
			return fmt.Errorf("orchestrator: edge from unknown node %q", from)
		}
		switch tr.kind {
		case kindStatic:
			if !g.known(tr.target) {
				return fmt.Errorf("orchestrator: edge %q -> unknown node %q", from, tr.target)
			}
		case kindConditional:
			// Dynamic routers (AddConditional) leave branches nil and are
			// validated at runtime. Path-map routers (AddConditionalEdges)
			// declare their targets, so validate them here.
			for _, t := range tr.branches {
				if !g.known(t) {
					return fmt.Errorf("orchestrator: conditional edge from %q has unknown target %q", from, t)
				}
			}
		case kindParallel:
			for _, b := range tr.branches {
				if !g.known(b) || b == START || b == END {
					return fmt.Errorf("orchestrator: parallel edge from %q has invalid branch %q", from, b)
				}
			}
			if !g.known(tr.join) {
				return fmt.Errorf("orchestrator: parallel edge from %q has unknown join %q", from, tr.join)
			}
		}
	}
	return nil
}

func joinErrs(errs []error) error {
	if len(errs) == 1 {
		return errs[0]
	}
	msg := ""
	for i, e := range errs {
		if i > 0 {
			msg += "; "
		}
		msg += e.Error()
	}
	return fmt.Errorf("%s", msg)
}

// Run executes the graph from START to END starting from the initial state. The
// threadID identifies the run for checkpointing. It returns the final state, or
// an error if a node fails, the context is canceled, or the step budget is
// exceeded.
func (e *Engine[S]) Run(ctx context.Context, threadID string, initial S) (S, error) {
	entry := e.g.trans[START]
	return e.run(ctx, threadID, initial, entry.target, 0)
}

// Resume continues a previously checkpointed run for threadID from where it
// left off. It requires an engine compiled WithCheckpointer and an existing
// checkpoint.
func (e *Engine[S]) Resume(ctx context.Context, threadID string) (S, error) {
	var zero S
	if e.checkpointer == nil {
		return zero, ErrNoCheckpointer
	}
	cp, ok := e.checkpointer.Load(threadID)
	if !ok {
		return zero, ErrNoCheckpoint
	}
	return e.run(ctx, threadID, cp.State, cp.Next, cp.Step)
}

func (e *Engine[S]) run(ctx context.Context, threadID string, state S, current string, step int) (S, error) {
	for {
		if current == END {
			return state, nil
		}
		if step > e.maxSteps {
			return state, fmt.Errorf("%w (%d)", ErrMaxSteps, e.maxSteps)
		}
		if err := ctx.Err(); err != nil {
			return state, err
		}

		// START is a sentinel with no work of its own.
		if current != START {
			out, err := e.execNode(ctx, current, state, step)
			if err != nil {
				return state, fmt.Errorf("node %q: %w", current, err)
			}
			state = out
		}

		tr, ok := e.g.trans[current]
		if !ok {
			return state, fmt.Errorf("orchestrator: no outgoing edge from %q", current)
		}

		var next string
		switch tr.kind {
		case kindStatic:
			next = tr.target
		case kindConditional:
			next = tr.router(state)
			if !e.g.known(next) {
				return state, fmt.Errorf("orchestrator: router from %q returned unknown node %q", current, next)
			}
		case kindParallel:
			results, err := e.runParallel(ctx, tr.branches, state, step)
			if err != nil {
				return state, err
			}
			state = tr.reducer(state, results)
			next = tr.join
		}

		step++
		if e.checkpointer != nil {
			if err := e.checkpointer.Save(threadID, Checkpoint[S]{Step: step, State: state, Next: next}); err != nil {
				return state, fmt.Errorf("orchestrator: checkpoint save: %w", err)
			}
		}
		current = next
	}
}

// execNode runs a single node with its retry policy and timeout, emitting enter
// and exit events. Retries happen only on non-cancellation errors.
func (e *Engine[S]) execNode(ctx context.Context, name string, state S, step int) (S, error) {
	n := e.g.nodes[name]
	e.emit(Event{Node: name, Step: step, Phase: PhaseEnter})
	start := time.Now()

	var (
		out S
		err error
	)
	for attempt := 1; attempt <= n.retry.MaxAttempts; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			err = ctxErr
			break
		}

		callCtx := ctx
		var cancel context.CancelFunc
		if n.timeout > 0 {
			callCtx, cancel = context.WithTimeout(ctx, n.timeout)
		}
		out, err = n.fn(callCtx, state)
		if cancel != nil {
			cancel()
		}

		if err == nil {
			break
		}
		// Stop early if the parent context was canceled: this is an abort, not
		// a transient failure to retry.
		if ctx.Err() != nil {
			err = ctx.Err()
			break
		}
		if attempt < n.retry.MaxAttempts {
			if waitErr := sleep(ctx, n.retry.Backoff<<(attempt-1)); waitErr != nil {
				err = waitErr
				break
			}
		}
	}

	e.emit(Event{Node: name, Step: step, Phase: PhaseExit, Duration: time.Since(start), Err: err})
	return out, err
}

// sleep waits for d, returning early with the context error if the context is
// canceled. A non-positive d returns immediately.
func sleep(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// runParallel executes every branch concurrently on a copy of base. If any
// branch errors, the shared context is canceled so siblings can observe the
// abort, and the first error encountered is returned. Results are returned in
// branch-declaration order to keep the subsequent reduce deterministic.
func (e *Engine[S]) runParallel(ctx context.Context, branches []string, base S, step int) ([]S, error) {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	results := make([]S, len(branches))
	errs := make([]error, len(branches))

	var wg sync.WaitGroup
	for i, name := range branches {
		wg.Add(1)
		go func(i int, name string) {
			defer wg.Done()
			// base is passed by value, giving each branch its own copy.
			out, err := e.execNode(ctx, name, base, step)
			if err != nil {
				errs[i] = err
				cancel() // cancel siblings on the first failure
				return
			}
			results[i] = out
		}(i, name)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			return nil, fmt.Errorf("parallel branch %q: %w", branches[i], err)
		}
	}
	return results, nil
}

func (e *Engine[S]) emit(ev Event) {
	if e.hook != nil {
		e.hook(ev)
	}
}
