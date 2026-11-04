// Command research is a runnable demonstration of the orchestrator package: a
// small research-agent graph wired as
//
//	START -> plan -> (searchA || searchB) -> merge -> summarize -> END
//
// plus a conditional edge that routes to "clarify" or "answer" depending on the
// query. All node functions are deterministic string transforms with no network
// or LLM calls, so the program is reproducible and CI-friendly.
package main

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"time"

	orch "github.com/kratos0722/agent-orchestrator"
)

// ResearchState flows through the graph. Each parallel branch writes its own
// field so branches never touch shared memory, which keeps the run race-free.
type ResearchState struct {
	Query    string
	Plan     string
	HitsA    []string // produced by searchA
	HitsB    []string // produced by searchB
	Findings []string // merged, sorted hits
	Answer   string
}

func plan(_ context.Context, s ResearchState) (ResearchState, error) {
	s.Plan = "decompose: " + s.Query
	return s, nil
}

func searchA(_ context.Context, s ResearchState) (ResearchState, error) {
	s.HitsA = []string{"A:intro(" + s.Query + ")", "A:details"}
	return s, nil
}

func searchB(_ context.Context, s ResearchState) (ResearchState, error) {
	s.HitsB = []string{"B:survey(" + s.Query + ")", "B:benchmarks"}
	return s, nil
}

// merge is the fan-in join node. The reducer has already combined the branch
// states by the time this runs; here we normalize the result deterministically.
func merge(_ context.Context, s ResearchState) (ResearchState, error) {
	sort.Strings(s.Findings)
	return s, nil
}

func summarize(_ context.Context, s ResearchState) (ResearchState, error) {
	s.Answer = fmt.Sprintf("summary of %d findings: %s",
		len(s.Findings), strings.Join(s.Findings, " | "))
	return s, nil
}

// reduceFindings deterministically merges the parallel branch outputs back into
// the base state by concatenating each branch's hits in declaration order.
func reduceFindings(base ResearchState, results []ResearchState) ResearchState {
	for _, r := range results {
		base.HitsA = append(base.HitsA, r.HitsA...)
		base.HitsB = append(base.HitsB, r.HitsB...)
	}
	base.Findings = append(base.Findings, base.HitsA...)
	base.Findings = append(base.Findings, base.HitsB...)
	return base
}

func buildResearchGraph() (*orch.Engine[ResearchState], error) {
	g := orch.New[ResearchState]()
	g.AddNode("plan", plan)
	g.AddNode("searchA", searchA)
	g.AddNode("searchB", searchB)
	g.AddNode("merge", merge)
	g.AddNode("summarize", summarize)

	g.AddEdge(orch.START, "plan")
	g.AddParallel("plan", []string{"searchA", "searchB"}, "merge", reduceFindings)
	g.AddEdge("merge", "summarize")
	g.AddEdge("summarize", orch.END)

	return g.Compile(orch.WithHook[ResearchState](logEvent))
}

// buildTriageGraph shows a conditional edge: route to clarify when the query is
// too short, otherwise straight to answer.
func buildTriageGraph() (*orch.Engine[ResearchState], error) {
	g := orch.New[ResearchState]()
	g.AddNode("triage", func(_ context.Context, s ResearchState) (ResearchState, error) {
		return s, nil
	})
	g.AddNode("clarify", func(_ context.Context, s ResearchState) (ResearchState, error) {
		s.Answer = "please provide more detail on: " + s.Query
		return s, nil
	})
	g.AddNode("answer", func(_ context.Context, s ResearchState) (ResearchState, error) {
		s.Answer = "answering: " + s.Query
		return s, nil
	})

	g.AddEdge(orch.START, "triage")
	g.AddConditional("triage", func(s ResearchState) string {
		if len(strings.Fields(s.Query)) < 3 {
			return "clarify"
		}
		return "answer"
	})
	g.AddEdge("clarify", orch.END)
	g.AddEdge("answer", orch.END)

	return g.Compile()
}

func logEvent(ev orch.Event) {
	if ev.Phase == orch.PhaseExit {
		fmt.Printf("  [step %d] %-10s done in %s\n", ev.Step, ev.Node, ev.Duration.Round(time.Microsecond))
	}
}

func main() {
	ctx := context.Background()

	research, err := buildResearchGraph()
	if err != nil {
		panic(err)
	}
	fmt.Println("research graph:")
	out, err := research.Run(ctx, "demo-thread", ResearchState{Query: "vector databases"})
	if err != nil {
		panic(err)
	}
	fmt.Println("  =>", out.Answer)

	triage, err := buildTriageGraph()
	if err != nil {
		panic(err)
	}
	fmt.Println("triage graph:")
	for _, q := range []string{"hi", "how do vector databases work"} {
		r, err := triage.Run(ctx, "triage-"+q, ResearchState{Query: q})
		if err != nil {
			panic(err)
		}
		fmt.Printf("  %-32q => %s\n", q, r.Answer)
	}
}
