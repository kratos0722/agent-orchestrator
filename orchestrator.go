// Package orchestrator is a typed, deterministic graph-execution engine for
// agent workflows. It models an agent run as a directed graph of named nodes
// connected by static, conditional, and parallel (fan-out/fan-in) edges, in the
// spirit of state-machine runtimes such as LangGraph, implemented with the Go
// standard library only.
//
// A graph is generic over a user-defined State type S. Nodes are plain
// functions func(context.Context, S) (S, error). Execution starts at the START
// sentinel and stops when it reaches END.
//
// The package provides per-node retries with backoff, per-node timeouts, a
// global max-steps guard against infinite cycles, pluggable checkpointing for
// resumable runs, and an observability hook invoked on every node enter/exit.
package orchestrator

import (
	"context"
	"fmt"
	"sort"
	"time"
)

// START and END are reserved sentinel node names. Execution begins at START and
// terminates when a transition resolves to END. Neither sentinel executes a
// node function; they only anchor edges.
const (
	START = "__start__"
	END   = "__end__"
)

// NodeFunc is the unit of work in a graph. It receives the current state and
// returns the next state or an error. Implementations must respect context
// cancellation (for timeouts and aborts) and should treat the input state as
// immutable, returning a new value rather than mutating shared backing storage.
type NodeFunc[S any] func(ctx context.Context, state S) (S, error)

// Router inspects the state and returns the name of the next node to run. It is
// used by conditional edges and must be a pure function of the state so that
// routing stays deterministic.
type Router[S any] func(state S) string

// Reducer deterministically merges the states produced by the branches of a
// parallel edge back into a single state. base is the state as it entered the
// fan-out; results holds each branch's output in branch-declaration order, so a
// reducer that depends only on base and results is deterministic.
type Reducer[S any] func(base S, results []S) S

// RetryPolicy configures per-node retries. A node is retried only when its
// function returns a non-nil error that is not caused by context cancellation.
// Backoff grows exponentially: the wait before attempt n is Backoff<<(n-1).
type RetryPolicy struct {
	// MaxAttempts is the total number of attempts, including the first. Values
	// below 1 are treated as 1 (no retry).
	MaxAttempts int
	// Backoff is the base delay between attempts. Zero means retry immediately.
	Backoff time.Duration
}

const (
	kindStatic = iota
	kindConditional
	kindParallel
)

type transition[S any] struct {
	kind     int
	target   string     // kindStatic
	router   Router[S]  // kindConditional
	branches []string   // kindParallel
	join     string     // kindParallel
	reducer  Reducer[S] // kindParallel
}

type node[S any] struct {
	name    string
	fn      NodeFunc[S]
	retry   RetryPolicy
	timeout time.Duration
}

// NodeOption customizes a node when it is registered with AddNode.
type NodeOption[S any] func(*node[S])

// WithRetry attaches a retry policy to a node. attempts is the total number of
// attempts (values below 1 are treated as 1); backoff is the base delay that
// grows exponentially between attempts.
func WithRetry[S any](attempts int, backoff time.Duration) NodeOption[S] {
	return func(n *node[S]) {
		n.retry = RetryPolicy{MaxAttempts: attempts, Backoff: backoff}
	}
}

// WithTimeout bounds a single attempt of a node with a context deadline. The
// node function must observe ctx.Done() for the timeout to take effect.
func WithTimeout[S any](d time.Duration) NodeOption[S] {
	return func(n *node[S]) { n.timeout = d }
}

// Graph is a builder for an agent workflow over a user-defined state type S.
// Construct one with New, register nodes and edges, then call Compile to obtain
// a runnable Engine. A Graph is not safe for concurrent construction, but a
// compiled Engine may be run concurrently for distinct threads.
type Graph[S any] struct {
	nodes map[string]*node[S]
	trans map[string]transition[S]
	errs  []error
}

// New creates an empty graph over state type S.
func New[S any]() *Graph[S] {
	return &Graph[S]{
		nodes: make(map[string]*node[S]),
		trans: make(map[string]transition[S]),
	}
}

func (g *Graph[S]) fail(format string, args ...any) {
	g.errs = append(g.errs, fmt.Errorf(format, args...))
}

// AddNode registers a named node. The name must be unique and must not collide
// with the START or END sentinels. Options configure retries and timeouts.
func (g *Graph[S]) AddNode(name string, fn NodeFunc[S], opts ...NodeOption[S]) *Graph[S] {
	switch {
	case name == START || name == END:
		g.fail("AddNode: %q is a reserved sentinel name", name)
		return g
	case name == "":
		g.fail("AddNode: node name must not be empty")
		return g
	case fn == nil:
		g.fail("AddNode: node %q has a nil function", name)
		return g
	}
	if _, dup := g.nodes[name]; dup {
		g.fail("AddNode: node %q already registered", name)
		return g
	}
	n := &node[S]{name: name, fn: fn}
	for _, opt := range opts {
		opt(n)
	}
	if n.retry.MaxAttempts < 1 {
		n.retry.MaxAttempts = 1
	}
	g.nodes[name] = n
	return g
}

func (g *Graph[S]) setTransition(from string, tr transition[S]) {
	if _, dup := g.trans[from]; dup {
		g.fail("edge: node %q already has an outgoing edge", from)
		return
	}
	g.trans[from] = tr
}

// AddEdge adds a static edge from one node to the next. from may be START; to
// may be END. Each node may have at most one outgoing edge of any kind.
func (g *Graph[S]) AddEdge(from, to string) *Graph[S] {
	g.setTransition(from, transition[S]{kind: kindStatic, target: to})
	return g
}

// AddConditional adds a conditional edge: at runtime the router inspects the
// state and returns the name of the next node (which may be END).
func (g *Graph[S]) AddConditional(from string, router Router[S]) *Graph[S] {
	if router == nil {
		g.fail("AddConditional: node %q has a nil router", from)
		return g
	}
	g.setTransition(from, transition[S]{kind: kindConditional, router: router})
	return g
}

// AddConditionalEdges adds a conditional edge whose possible targets are known
// up front, in the style of LangGraph's path maps. At runtime the router
// returns a key, and paths resolves that key to the next node (which may be
// END). Because the full set of targets is declared here, Compile validates
// them like any other edge, so a typo'd or renamed target is caught when the
// graph is built rather than when the branch is first taken.
//
// If the router returns a key that is not present in paths, the run fails at
// that step with an unknown-node error, just like AddConditional.
func (g *Graph[S]) AddConditionalEdges(from string, router Router[S], paths map[string]string) *Graph[S] {
	switch {
	case router == nil:
		g.fail("AddConditionalEdges: node %q has a nil router", from)
		return g
	case len(paths) == 0:
		g.fail("AddConditionalEdges: node %q has an empty path map", from)
		return g
	}
	// Capture the declared targets (sorted, de-duplicated) so Compile can
	// validate them deterministically.
	seen := make(map[string]bool, len(paths))
	targets := make([]string, 0, len(paths))
	for key, target := range paths {
		if target == "" {
			g.fail("AddConditionalEdges: node %q maps key %q to an empty target", from, key)
			return g
		}
		if !seen[target] {
			seen[target] = true
			targets = append(targets, target)
		}
	}
	sort.Strings(targets)

	// Resolve the user's key to a concrete node name at runtime. An unmapped
	// key yields "" and surfaces as an unknown-node error in the engine.
	resolved := make(map[string]string, len(paths))
	for k, v := range paths {
		resolved[k] = v
	}
	wrapped := func(state S) string { return resolved[router(state)] }

	g.setTransition(from, transition[S]{kind: kindConditional, router: wrapped, branches: targets})
	return g
}

// AddParallel adds a fan-out/fan-in edge. When execution leaves from, every
// branch node runs concurrently on a copy of the current state. Once all
// branches complete, reducer merges their outputs back into a single state and
// execution continues at join. If any branch errors, its siblings are canceled
// via context and the run fails.
func (g *Graph[S]) AddParallel(from string, branches []string, join string, reducer Reducer[S]) *Graph[S] {
	switch {
	case len(branches) == 0:
		g.fail("AddParallel: node %q has no branches", from)
		return g
	case reducer == nil:
		g.fail("AddParallel: node %q has a nil reducer", from)
		return g
	case join == "":
		g.fail("AddParallel: node %q has an empty join target", from)
		return g
	}
	cp := make([]string, len(branches))
	copy(cp, branches)
	g.setTransition(from, transition[S]{
		kind:     kindParallel,
		branches: cp,
		join:     join,
		reducer:  reducer,
	})
	return g
}
