package orchestrator_test

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	orch "github.com/kratos0722/agent-orchestrator"
)

// linState is a minimal state used by most tests: an ordered trace of visited
// nodes plus a counter.
type linState struct {
	trace []string
	n     int
}

func visit(name string) orch.NodeFunc[linState] {
	return func(_ context.Context, s linState) (linState, error) {
		s.trace = append(s.trace, name)
		s.n++
		return s, nil
	}
}

func mustCompile(t *testing.T, g *orch.Graph[linState], opts ...orch.Option[linState]) *orch.Engine[linState] {
	t.Helper()
	e, err := g.Compile(opts...)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	return e
}

func TestLinearFlow(t *testing.T) {
	g := orch.New[linState]()
	g.AddNode("a", visit("a"))
	g.AddNode("b", visit("b"))
	g.AddNode("c", visit("c"))
	g.AddEdge(orch.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "c")
	g.AddEdge("c", orch.END)

	e := mustCompile(t, g)
	out, err := e.Run(context.Background(), "t1", linState{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	want := []string{"a", "b", "c"}
	if fmt.Sprint(out.trace) != fmt.Sprint(want) {
		t.Fatalf("trace = %v, want %v", out.trace, want)
	}
	if out.n != 3 {
		t.Fatalf("n = %d, want 3", out.n)
	}
}

func TestConditionalRouting(t *testing.T) {
	build := func() *orch.Engine[linState] {
		g := orch.New[linState]()
		g.AddNode("triage", visit("triage"))
		g.AddNode("clarify", visit("clarify"))
		g.AddNode("answer", visit("answer"))
		g.AddEdge(orch.START, "triage")
		g.AddConditional("triage", func(s linState) string {
			if s.n < 2 { // only "triage" visited so far
				return "clarify"
			}
			return "answer"
		})
		g.AddEdge("clarify", orch.END)
		g.AddEdge("answer", orch.END)
		return mustCompile(t, g)
	}

	tests := []struct {
		name  string
		start linState
		want  []string
	}{
		{"routes to clarify", linState{}, []string{"triage", "clarify"}},
		{"routes to answer", linState{n: 5}, []string{"triage", "answer"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := build().Run(context.Background(), "t", tc.start)
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if fmt.Sprint(out.trace) != fmt.Sprint(tc.want) {
				t.Fatalf("trace = %v, want %v", out.trace, tc.want)
			}
		})
	}
}

// pstate carries a per-branch tag so the reducer can merge deterministically.
type pstate struct {
	tag    string
	merged string
}

func TestParallelMergeDeterministic(t *testing.T) {
	g := orch.New[pstate]()
	g.AddNode("start", func(_ context.Context, s pstate) (pstate, error) { return s, nil })
	// Branches finish in nondeterministic order but the reducer sorts by index.
	g.AddNode("x", func(_ context.Context, s pstate) (pstate, error) {
		time.Sleep(2 * time.Millisecond)
		s.tag = "x"
		return s, nil
	})
	g.AddNode("y", func(_ context.Context, s pstate) (pstate, error) {
		s.tag = "y"
		return s, nil
	})
	g.AddNode("done", func(_ context.Context, s pstate) (pstate, error) { return s, nil })

	g.AddEdge(orch.START, "start")
	g.AddParallel("start", []string{"x", "y"}, "done", func(base pstate, results []pstate) pstate {
		for _, r := range results { // declaration order: x then y
			base.merged += r.tag
		}
		return base
	})
	g.AddEdge("done", orch.END)

	e, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	for i := 0; i < 50; i++ {
		out, err := e.Run(context.Background(), "p", pstate{})
		if err != nil {
			t.Fatalf("run: %v", err)
		}
		if out.merged != "xy" {
			t.Fatalf("merged = %q, want %q (nondeterministic merge)", out.merged, "xy")
		}
	}
}

func TestParallelRunsConcurrently(t *testing.T) {
	// Each branch sends to the other and waits; if the branches ran
	// sequentially the first would block forever and the context would expire.
	a2b := make(chan struct{}, 1)
	b2a := make(chan struct{}, 1)

	g := orch.New[pstate]()
	g.AddNode("fork", func(_ context.Context, s pstate) (pstate, error) { return s, nil })
	g.AddNode("x", func(ctx context.Context, s pstate) (pstate, error) {
		a2b <- struct{}{}
		select {
		case <-b2a:
			s.tag = "x"
			return s, nil
		case <-ctx.Done():
			return s, ctx.Err()
		}
	})
	g.AddNode("y", func(ctx context.Context, s pstate) (pstate, error) {
		b2a <- struct{}{}
		select {
		case <-a2b:
			s.tag = "y"
			return s, nil
		case <-ctx.Done():
			return s, ctx.Err()
		}
	})
	g.AddNode("join", func(_ context.Context, s pstate) (pstate, error) { return s, nil })
	g.AddEdge(orch.START, "fork")
	g.AddParallel("fork", []string{"x", "y"}, "join", func(base pstate, _ []pstate) pstate { return base })
	g.AddEdge("join", orch.END)

	e, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if _, err := e.Run(ctx, "c", pstate{}); err != nil {
		t.Fatalf("branches did not run concurrently: %v", err)
	}
}

func TestRetrySucceedsAfterTransientFailures(t *testing.T) {
	var calls int32
	flaky := func(_ context.Context, s linState) (linState, error) {
		if atomic.AddInt32(&calls, 1) < 3 {
			return s, errors.New("transient")
		}
		s.trace = append(s.trace, "ok")
		return s, nil
	}
	g := orch.New[linState]()
	g.AddNode("flaky", flaky, orch.WithRetry[linState](3, time.Millisecond))
	g.AddEdge(orch.START, "flaky")
	g.AddEdge("flaky", orch.END)

	e := mustCompile(t, g)
	out, err := e.Run(context.Background(), "r", linState{})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Fatalf("calls = %d, want 3", got)
	}
	if fmt.Sprint(out.trace) != "[ok]" {
		t.Fatalf("trace = %v, want [ok]", out.trace)
	}
}

func TestRetryRespectsMaxAttempts(t *testing.T) {
	var calls int32
	boom := errors.New("boom")
	always := func(_ context.Context, s linState) (linState, error) {
		atomic.AddInt32(&calls, 1)
		return s, boom
	}
	g := orch.New[linState]()
	g.AddNode("bad", always, orch.WithRetry[linState](4, time.Millisecond))
	g.AddEdge(orch.START, "bad")
	g.AddEdge("bad", orch.END)

	e := mustCompile(t, g)
	_, err := e.Run(context.Background(), "r", linState{})
	if !errors.Is(err, boom) {
		t.Fatalf("err = %v, want wrapped boom", err)
	}
	if got := atomic.LoadInt32(&calls); got != 4 {
		t.Fatalf("calls = %d, want 4 (max attempts)", got)
	}
}

func TestPerNodeTimeoutFires(t *testing.T) {
	slow := func(ctx context.Context, s linState) (linState, error) {
		select {
		case <-time.After(time.Second):
			return s, nil
		case <-ctx.Done():
			return s, ctx.Err()
		}
	}
	g := orch.New[linState]()
	g.AddNode("slow", slow, orch.WithTimeout[linState](20*time.Millisecond))
	g.AddEdge(orch.START, "slow")
	g.AddEdge("slow", orch.END)

	e := mustCompile(t, g)
	start := time.Now()
	_, err := e.Run(context.Background(), "to", linState{})
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want DeadlineExceeded", err)
	}
	if elapsed := time.Since(start); elapsed > 500*time.Millisecond {
		t.Fatalf("timeout did not fire promptly: %v", elapsed)
	}
}

func TestMaxStepsGuard(t *testing.T) {
	g := orch.New[linState]()
	g.AddNode("loop", visit("loop"))
	g.AddEdge(orch.START, "loop")
	g.AddEdge("loop", "loop") // self-cycle, never reaches END
	e := mustCompile(t, g, orch.WithMaxSteps[linState](10))

	_, err := e.Run(context.Background(), "cyc", linState{})
	if !errors.Is(err, orch.ErrMaxSteps) {
		t.Fatalf("err = %v, want ErrMaxSteps", err)
	}
}

func TestCheckpointSaveLoadResume(t *testing.T) {
	cp := orch.NewMemoryCheckpointer[linState]()
	var healthy atomic.Bool // false => node "b" fails

	g := orch.New[linState]()
	g.AddNode("a", visit("a"))
	g.AddNode("b", func(_ context.Context, s linState) (linState, error) {
		if !healthy.Load() {
			return s, errors.New("b unavailable")
		}
		s.trace = append(s.trace, "b")
		return s, nil
	})
	g.AddNode("c", visit("c"))
	g.AddEdge(orch.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", "c")
	g.AddEdge("c", orch.END)

	e := mustCompile(t, g, orch.WithCheckpointer[linState](cp))

	// First run fails at b; a checkpoint should exist with Next == "b".
	if _, err := e.Run(context.Background(), "thread-1", linState{}); err == nil {
		t.Fatal("expected failure at b on first run")
	}
	saved, ok := cp.Load("thread-1")
	if !ok {
		t.Fatal("no checkpoint saved")
	}
	if saved.Next != "b" {
		t.Fatalf("checkpoint Next = %q, want b", saved.Next)
	}
	if fmt.Sprint(saved.State.trace) != "[a]" {
		t.Fatalf("checkpoint state trace = %v, want [a]", saved.State.trace)
	}

	// Recover and resume: should continue from b through to END.
	healthy.Store(true)
	out, err := e.Resume(context.Background(), "thread-1")
	if err != nil {
		t.Fatalf("resume: %v", err)
	}
	if fmt.Sprint(out.trace) != "[a b c]" {
		t.Fatalf("resumed trace = %v, want [a b c]", out.trace)
	}
}

func TestContextCancelAbortsRun(t *testing.T) {
	g := orch.New[linState]()
	g.AddNode("block", func(ctx context.Context, s linState) (linState, error) {
		<-ctx.Done()
		return s, ctx.Err()
	})
	g.AddEdge(orch.START, "block")
	g.AddEdge("block", orch.END)

	e := mustCompile(t, g)
	ctx, cancel := context.WithCancel(context.Background())
	errc := make(chan error, 1)
	go func() {
		_, err := e.Run(ctx, "cancel", linState{})
		errc <- err
	}()
	cancel()
	select {
	case err := <-errc:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("err = %v, want Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("run did not abort on cancellation")
	}
}

func TestContextCancelAbortsSiblingBranches(t *testing.T) {
	var siblingCanceled atomic.Bool
	released := make(chan struct{})

	g := orch.New[linState]()
	g.AddNode("fork", func(_ context.Context, s linState) (linState, error) { return s, nil })
	// failing branch errors once the sibling has started.
	g.AddNode("failing", func(_ context.Context, s linState) (linState, error) {
		<-released // wait until the sibling is confirmed running
		return s, errors.New("branch failed")
	})
	// sibling blocks until it observes cancellation triggered by the failure.
	g.AddNode("sibling", func(ctx context.Context, s linState) (linState, error) {
		close(released)
		<-ctx.Done()
		siblingCanceled.Store(true)
		return s, ctx.Err()
	})
	g.AddNode("join", func(_ context.Context, s linState) (linState, error) { return s, nil })
	g.AddEdge(orch.START, "fork")
	g.AddParallel("fork", []string{"failing", "sibling"}, "join",
		func(base linState, _ []linState) linState { return base })
	g.AddEdge("join", orch.END)

	e := mustCompile(t, g)
	_, err := e.Run(context.Background(), "sib", linState{})
	if err == nil {
		t.Fatal("expected error from failing branch")
	}
	if !siblingCanceled.Load() {
		t.Fatal("sibling branch was not canceled when its peer failed")
	}
}

func TestCompileValidation(t *testing.T) {
	t.Run("no START edge", func(t *testing.T) {
		g := orch.New[linState]()
		g.AddNode("a", visit("a"))
		if _, err := g.Compile(); err == nil {
			t.Fatal("expected error for missing START edge")
		}
	})
	t.Run("edge to unknown node", func(t *testing.T) {
		g := orch.New[linState]()
		g.AddNode("a", visit("a"))
		g.AddEdge(orch.START, "a")
		g.AddEdge("a", "ghost")
		if _, err := g.Compile(); err == nil {
			t.Fatal("expected error for edge to unknown node")
		}
	})
	t.Run("duplicate node", func(t *testing.T) {
		g := orch.New[linState]()
		g.AddNode("a", visit("a"))
		g.AddNode("a", visit("a"))
		g.AddEdge(orch.START, "a")
		g.AddEdge("a", orch.END)
		if _, err := g.Compile(); err == nil {
			t.Fatal("expected error for duplicate node")
		}
	})
	t.Run("reserved name", func(t *testing.T) {
		g := orch.New[linState]()
		g.AddNode(orch.END, visit("x"))
		if _, err := g.Compile(); err == nil {
			t.Fatal("expected error for reserved node name")
		}
	})
}

func TestObservabilityHook(t *testing.T) {
	var enters, exits int32
	hook := func(ev orch.Event) {
		switch ev.Phase {
		case orch.PhaseEnter:
			atomic.AddInt32(&enters, 1)
		case orch.PhaseExit:
			atomic.AddInt32(&exits, 1)
		}
	}
	g := orch.New[linState]()
	g.AddNode("a", visit("a"))
	g.AddNode("b", visit("b"))
	g.AddEdge(orch.START, "a")
	g.AddEdge("a", "b")
	g.AddEdge("b", orch.END)

	e := mustCompile(t, g, orch.WithHook[linState](hook))
	if _, err := e.Run(context.Background(), "h", linState{}); err != nil {
		t.Fatalf("run: %v", err)
	}
	if enters != 2 || exits != 2 {
		t.Fatalf("hook counts enters=%d exits=%d, want 2/2", enters, exits)
	}
}
