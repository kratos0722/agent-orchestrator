# agent-orchestrator

A typed, deterministic **graph-execution engine for agent workflows**, written in
idiomatic Go with the **standard library only**. It models an agent run as a
directed graph of named nodes connected by static, conditional, and parallel
(fan-out/fan-in) edges — the kind of state-machine runtime that frameworks like
[LangGraph](https://langchain-ai.github.io/langgraph/) provide, implemented
cleanly in Go with generics.

No LLM or network calls live in the engine: nodes are plain functions, so the
runtime is reproducible, unit-testable, and CI-friendly.

```
                         ┌──────────────┐
                         │   searchA    │
                 ┌──────▶│  (branch)    │──────┐
┌───────┐  ┌──────┐      └──────────────┘      ▼  reducer   ┌───────┐   ┌────────────┐   ┌─────┐
│ START │─▶│ plan │──fan-out               fan-in ────────▶│ merge │──▶│ summarize  │──▶│ END │
└───────┘  └──────┘      ┌──────────────┐      ▲            └───────┘   └────────────┘   └─────┘
                 └──────▶│   searchB    │──────┘
                         │  (branch)    │
                         └──────────────┘

Conditional routing:
                         ┌──────────┐  router(state)=="clarify"   ┌─────────┐
┌───────┐  ┌────────┐    │          │────────────────────────────▶│ clarify │──▶ END
│ START │─▶│ triage │───▶│  router  │                              └─────────┘
└───────┘  └────────┘    │          │────────────────────────────▶┌─────────┐
                         └──────────┘  router(state)=="answer"     │ answer  │──▶ END
                                                                   └─────────┘
```

## Core concepts

| Concept | What it is |
| --- | --- |
| **State `S`** | A user-defined type that flows through the graph. `Graph[S]` and `Engine[S]` are generic over it. |
| **Node** | A named unit of work: `func(ctx context.Context, S) (S, error)`. Must respect `ctx` and treat the input state as immutable. |
| **START / END** | Reserved sentinels. Execution begins at `START` and stops when a transition resolves to `END`. |
| **Static edge** | `AddEdge(from, to)` — unconditional transition. |
| **Conditional edge** | `AddConditional(from, router)` — a pure `func(S) string` inspects the state and names the next node (resolved at runtime). |
| **Conditional path map** | `AddConditionalEdges(from, router, paths)` — LangGraph-style: the router returns a key and `paths` maps keys to target nodes. Because the targets are declared, `Compile` validates them, so a mistyped target fails at build time instead of at runtime. |
| **Parallel edge** | `AddParallel(from, branches, join, reducer)` — branches run **concurrently** on copies of the state, then a **reducer** `func(base S, results []S) S` merges them deterministically before continuing at `join`. If any branch errors, siblings are canceled via context. |
| **Retry** | Per-node `WithRetry(attempts, backoff)` — retries only on non-cancellation errors, with exponential backoff. |
| **Timeout** | Per-node `WithTimeout(d)` — bounds each attempt with a `context.WithTimeout` deadline. |
| **Max-steps guard** | `WithMaxSteps(n)` — returns `ErrMaxSteps` if a run exceeds the step budget, preventing infinite cycles. |
| **Checkpointer** | `Checkpointer[S]` interface (`Save`/`Load`) with an in-memory implementation. The engine checkpoints after every step; `Resume` continues a run from its last checkpoint. |
| **Observability hook** | `WithHook(func(Event))` — invoked on each node enter/exit with node name, step index, and duration. No external deps. |

Determinism: linear and conditional paths are inherently deterministic; for
parallel edges the **reducer** receives branch results in declaration order, so
the merge does not depend on goroutine scheduling.

## Install

```sh
go get github.com/kratos0722/agent-orchestrator
```

Requires Go 1.23+ (generics). Zero third-party dependencies.

## Minimal usage

```go
package main

import (
	"context"
	"fmt"

	orch "github.com/kratos0722/agent-orchestrator"
)

type State struct {
	Query  string
	Answer string
}

func main() {
	g := orch.New[State]()

	g.AddNode("plan", func(_ context.Context, s State) (State, error) {
		s.Query = "planned: " + s.Query
		return s, nil
	})
	g.AddNode("answer", func(_ context.Context, s State) (State, error) {
		s.Answer = "done(" + s.Query + ")"
		return s, nil
	})

	g.AddEdge(orch.START, "plan")
	g.AddEdge("plan", "answer")
	g.AddEdge("answer", orch.END)

	engine, err := g.Compile()
	if err != nil {
		panic(err)
	}

	out, err := engine.Run(context.Background(), "thread-1", State{Query: "hello"})
	if err != nil {
		panic(err)
	}
	fmt.Println(out.Answer) // done(planned: hello)
}
```

### Parallel fan-out with a reducer

```go
g.AddParallel("plan", []string{"searchA", "searchB"}, "merge",
	func(base State, results []State) State {
		for _, r := range results { // deterministic: declaration order
			base.Answer += r.Answer
		}
		return base
	})
```

### Reliability options

```go
g.AddNode("call-tool", toolFn,
	orch.WithRetry[State](3, 50*time.Millisecond), // 3 attempts, exponential backoff
	orch.WithTimeout[State](2*time.Second),        // per-attempt deadline
)

engine, _ := g.Compile(
	orch.WithMaxSteps[State](100),                       // cycle guard
	orch.WithCheckpointer[State](orch.NewMemoryCheckpointer[State]()),
	orch.WithHook[State](func(ev orch.Event) { /* trace */ }),
)
```

### Resume from a checkpoint

```go
// A run that failed midway can be resumed from its last saved step.
out, err := engine.Resume(context.Background(), "thread-1")
```

## Runnable example

```sh
go run ./examples/research
```

Builds the research-agent graph
`plan → (searchA ‖ searchB) → merge → summarize → END` plus a conditional triage
graph, using deterministic mock node functions (no network/LLM).

## How it maps to real agent frameworks

| This engine | LangGraph / typical agent stacks |
| --- | --- |
| `Graph[S]` + `Compile` | `StateGraph` + `compile()` |
| `NodeFunc[S]` | node / tool callables |
| `AddConditional` + `Router` | conditional edges / routers |
| `AddParallel` + `Reducer` | fan-out super-steps + channel/state reducers |
| `WithRetry` / `WithTimeout` | node retry policies and timeouts |
| `Checkpointer[S]` / `Resume` | checkpointers + thread-level persistence |
| `WithHook` / `Event` | callbacks / tracing (LangSmith-style) |
| `WithMaxSteps` | recursion-limit guard |

## Testing

```sh
go build ./...
go vet ./...
go test -race ./...
```

The suite covers linear flow, conditional routing, concurrent fan-out with
deterministic merge, retry (success-after-N and max-attempts), per-node timeout,
the max-steps cycle guard, checkpoint save/load/resume, and context cancellation
of a run and of sibling branches — all clean under the race detector.
