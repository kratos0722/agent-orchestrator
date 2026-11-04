package orchestrator_test

import (
	"context"
	"fmt"
	"strings"
	"testing"

	orch "github.com/kratos0722/agent-orchestrator"
)

// routeState carries a label that a router maps to a branch key.
type routeState struct {
	label string
	trace []string
}

func tag(name string) orch.NodeFunc[routeState] {
	return func(_ context.Context, s routeState) (routeState, error) {
		s.trace = append(s.trace, name)
		return s, nil
	}
}

// buildTriage wires START -> classify -> (conditional by label) -> {urgent|normal} -> END.
func buildTriage() *orch.Graph[routeState] {
	g := orch.New[routeState]()
	g.AddNode("classify", tag("classify"))
	g.AddNode("urgent", tag("urgent"))
	g.AddNode("normal", tag("normal"))
	g.AddEdge(orch.START, "classify")
	g.AddConditionalEdges("classify",
		func(s routeState) string { return s.label },
		map[string]string{"hot": "urgent", "cold": "normal"},
	)
	g.AddEdge("urgent", orch.END)
	g.AddEdge("normal", orch.END)
	return g
}

func TestConditionalEdgesRouteByKey(t *testing.T) {
	cases := []struct {
		label string
		want  []string
	}{
		{"hot", []string{"classify", "urgent"}},
		{"cold", []string{"classify", "normal"}},
	}
	for _, tc := range cases {
		t.Run(tc.label, func(t *testing.T) {
			e, err := buildTriage().Compile()
			if err != nil {
				t.Fatalf("compile: %v", err)
			}
			out, err := e.Run(context.Background(), "t", routeState{label: tc.label})
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if fmt.Sprint(out.trace) != fmt.Sprint(tc.want) {
				t.Fatalf("trace = %v, want %v", out.trace, tc.want)
			}
		})
	}
}

// The whole point of path maps: a bad target is rejected at Compile, not at runtime.
func TestConditionalEdgesValidateTargetsAtCompile(t *testing.T) {
	g := orch.New[routeState]()
	g.AddNode("classify", tag("classify"))
	g.AddNode("urgent", tag("urgent"))
	g.AddEdge(orch.START, "classify")
	g.AddConditionalEdges("classify",
		func(s routeState) string { return s.label },
		map[string]string{"hot": "urgent", "cold": "typo_node"}, // typo_node never registered
	)
	g.AddEdge("urgent", orch.END)

	_, err := g.Compile()
	if err == nil {
		t.Fatal("expected Compile to reject an unknown conditional target")
	}
	if !strings.Contains(err.Error(), "typo_node") {
		t.Fatalf("error should name the unknown target, got: %v", err)
	}
}

// END is a legal path-map target and must pass validation.
func TestConditionalEdgesAllowEndTarget(t *testing.T) {
	g := orch.New[routeState]()
	g.AddNode("classify", tag("classify"))
	g.AddNode("work", tag("work"))
	g.AddEdge(orch.START, "classify")
	g.AddConditionalEdges("classify",
		func(s routeState) string {
			if s.label == "done" {
				return "stop"
			}
			return "go"
		},
		map[string]string{"stop": orch.END, "go": "work"},
	)
	g.AddEdge("work", orch.END)

	e, err := g.Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	out, err := e.Run(context.Background(), "t", routeState{label: "done"})
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if fmt.Sprint(out.trace) != fmt.Sprint([]string{"classify"}) {
		t.Fatalf("trace = %v, want [classify]", out.trace)
	}
}

// An unmapped key resolves to "" and fails at runtime (documented behavior).
func TestConditionalEdgesUnmappedKeyFailsAtRuntime(t *testing.T) {
	e, err := buildTriage().Compile()
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	_, err = e.Run(context.Background(), "t", routeState{label: "unknown-label"})
	if err == nil {
		t.Fatal("expected a runtime error for an unmapped router key")
	}
}

func TestConditionalEdgesRejectsEmptyPaths(t *testing.T) {
	g := orch.New[routeState]()
	g.AddNode("classify", tag("classify"))
	g.AddEdge(orch.START, "classify")
	g.AddConditionalEdges("classify", func(s routeState) string { return s.label }, nil)
	if _, err := g.Compile(); err == nil {
		t.Fatal("expected Compile to fail for an empty path map")
	}
}
