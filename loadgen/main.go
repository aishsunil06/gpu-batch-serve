// Command loadgen drives any of the three serving arms at a fixed concurrency.
//
// A second load generator, in a second language, exists for a reason beyond
// novelty. The Python generator shares an interpreter and an event loop with
// nothing, but it does share a machine, and at high concurrency its own
// scheduling becomes a plausible confound for the latency it reports. Go's
// goroutine-per-client model holds concurrency with far less client-side
// overhead, so agreement between the two is evidence that a measured
// difference belongs to the server rather than the client.
//
// Closed loop, matching the Python generator: each worker waits for its
// response before sending again, holding exactly -concurrency requests in
// flight. An open-loop generator at a fixed arrival rate would let a slow arm
// build an unbounded queue and then report the backlog rather than the server.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type tokenRequest struct {
	InputIDs      []int64 `json:"input_ids"`
	AttentionMask []int64 `json:"attention_mask"`
	TokenTypeIDs  []int64 `json:"token_type_ids"`
}

type fixture struct {
	Meta struct {
		Size           int `json:"size"`
		SequenceLength int `json:"sequence_length"`
	} `json:"meta"`
	Requests []tokenRequest `json:"requests"`
}

// tritonRequest is the KServe v2 inference payload Triton expects on its HTTP
// endpoint. The local and Ray arms take the simpler tokenRequest shape, so the
// generator builds whichever the target arm speaks.
type tritonRequest struct {
	Inputs []tritonInput `json:"inputs"`
}

type tritonInput struct {
	Name     string  `json:"name"`
	Shape    []int   `json:"shape"`
	Datatype string  `json:"datatype"`
	Data     []int64 `json:"data"`
}

type result struct {
	latencies []float64
	failures  int64
	wall      time.Duration
}

func buildPayload(arm string, req tokenRequest, seqLen int) ([]byte, error) {
	if arm == "triton" {
		return json.Marshal(tritonRequest{
			Inputs: []tritonInput{
				{Name: "input_ids", Shape: []int{1, seqLen}, Datatype: "INT64", Data: req.InputIDs},
				{Name: "attention_mask", Shape: []int{1, seqLen}, Datatype: "INT64", Data: req.AttentionMask},
				{Name: "token_type_ids", Shape: []int{1, seqLen}, Datatype: "INT64", Data: req.TokenTypeIDs},
			},
		})
	}
	return json.Marshal(req)
}

func percentile(sorted []float64, fraction float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	// Nearest rank, matching the Python harness. An interpolated percentile
	// over a few hundred samples reports a value no request experienced.
	rank := int(fraction*float64(len(sorted)) + 0.5)
	if rank < 1 {
		rank = 1
	}
	if rank > len(sorted) {
		rank = len(sorted)
	}
	return sorted[rank-1]
}

func runWindow(url, arm string, fx *fixture, concurrency int, duration time.Duration, client *http.Client) result {
	var (
		mu        sync.Mutex
		latencies = make([]float64, 0, 4096)
		failures  int64
		wg        sync.WaitGroup
	)

	start := time.Now()
	deadline := start.Add(duration)

	for worker := 0; worker < concurrency; worker++ {
		wg.Add(1)
		go func(offset int) {
			defer wg.Done()
			local := make([]float64, 0, 1024)
			index := offset
			for time.Now().Before(deadline) {
				req := fx.Requests[index%len(fx.Requests)]
				index += concurrency

				body, err := buildPayload(arm, req, fx.Meta.SequenceLength)
				if err != nil {
					atomic.AddInt64(&failures, 1)
					continue
				}

				sent := time.Now()
				resp, err := client.Post(url, "application/json", bytes.NewReader(body))
				if err != nil {
					atomic.AddInt64(&failures, 1)
					continue
				}
				// The body must be drained and closed, not just closed, or the
				// connection is not returned to the keep-alive pool and every
				// request pays a fresh TCP handshake. That would show up as
				// server latency and it is not.
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()

				if resp.StatusCode != http.StatusOK {
					atomic.AddInt64(&failures, 1)
					continue
				}
				local = append(local, float64(time.Since(sent).Microseconds())/1000.0)
			}
			mu.Lock()
			latencies = append(latencies, local...)
			mu.Unlock()
		}(worker)
	}

	wg.Wait()
	return result{latencies: latencies, failures: failures, wall: time.Since(start)}
}

func main() {
	var (
		url         = flag.String("url", "http://127.0.0.1:8100/embed-tokens", "inference endpoint")
		arm         = flag.String("arm", "local", "local | ray | triton")
		fixturePath = flag.String("fixture", "bench/fixtures/workload_tokens.json", "pre-tokenized workload")
		concurrency = flag.Int("concurrency", 8, "virtual clients held in flight")
		durationSec = flag.Float64("duration", 20, "measured window, seconds")
		warmupSec   = flag.Float64("warmup", 5, "unmeasured window, seconds")
		jsonOut     = flag.Bool("json", false, "emit a single JSON object")
	)
	flag.Parse()

	raw, err := os.ReadFile(*fixturePath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "cannot read fixture %s: %v\n", *fixturePath, err)
		os.Exit(1)
	}
	var fx fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		fmt.Fprintf(os.Stderr, "cannot parse fixture: %v\n", err)
		os.Exit(1)
	}
	if len(fx.Requests) == 0 {
		fmt.Fprintln(os.Stderr, "fixture contains no requests")
		os.Exit(1)
	}

	client := &http.Client{
		Timeout: 60 * time.Second,
		Transport: &http.Transport{
			// Default MaxIdleConnsPerHost is 2, which would throttle the
			// generator itself and cap measured throughput well below what the
			// server can do.
			MaxIdleConns:        *concurrency + 16,
			MaxIdleConnsPerHost: *concurrency + 16,
			MaxConnsPerHost:     *concurrency + 16,
			IdleConnTimeout:     90 * time.Second,
		},
	}

	if *warmupSec > 0 {
		runWindow(*url, *arm, &fx, *concurrency, time.Duration(*warmupSec*float64(time.Second)), client)
	}

	res := runWindow(*url, *arm, &fx, *concurrency, time.Duration(*durationSec*float64(time.Second)), client)
	sort.Float64s(res.latencies)

	completed := len(res.latencies)
	rps := 0.0
	if res.wall.Seconds() > 0 {
		rps = float64(completed) / res.wall.Seconds()
	}

	if *jsonOut {
		out := map[string]any{
			"arm":            *arm,
			"concurrency":    *concurrency,
			"completed":      completed,
			"failed":         res.failures,
			"wall_seconds":   res.wall.Seconds(),
			"throughput_rps": rps,
			"latency_ms": map[string]float64{
				"p50": percentile(res.latencies, 0.50),
				"p95": percentile(res.latencies, 0.95),
				"p99": percentile(res.latencies, 0.99),
			},
			"generator": "go",
		}
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		enc.Encode(out)
		return
	}

	fmt.Printf("arm=%s conc=%d completed=%d failed=%d rps=%.1f p50=%.2fms p95=%.2fms p99=%.2fms\n",
		*arm, *concurrency, completed, res.failures, rps,
		percentile(res.latencies, 0.50),
		percentile(res.latencies, 0.95),
		percentile(res.latencies, 0.99),
	)
}
