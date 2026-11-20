package main

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestRouteAuto(t *testing.T) {
	user := func(text string) message { return message{Role: "user", Content: content{text: text}} }
	assistant := func(text string) message { return message{Role: "assistant", Content: content{text: text}} }
	withImage := message{Role: "user", Content: content{parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURL{URL: "data:image/png;base64,"}}, {Type: "text", Text: "What is this?"}}}}
	for _, tc := range []struct {
		name          string
		messages      []message
		model, reason string
	}{
		{"simple question", []message{user("What is AKS Automatic in one sentence?")}, cheapModel, "simple"},
		{"creative writing", []message{user("Write a short poem about the sea.")}, cheapModel, "simple"},
		{"dress code", []message{user("What is the dress code for a summer wedding?")}, cheapModel, "simple"},
		{"travel", []message{user("Plan a trip from Paris to Rome in 3 days.")}, cheapModel, "simple"},
		{"opening hours", []message{user("Is the store open 9-5, or 24/7 on holidays?")}, cheapModel, "simple"},
		{"code fence", []message{user("Why does this fail?\n```go\nfmt.Println(x)\n```")}, strongModel, "code"},
		{"coding request", []message{user("Write Python code to reverse a linked list.")}, strongModel, "code"},
		{"function request", []message{user("Write a Go function that reverses a string, handling Unicode correctly.")}, strongModel, "code"},
		{"ambiguous language name", []message{user("Why does my Rust program panic on startup?")}, strongModel, "code"},
		{"exception name", []message{user("What causes a NullPointerException here?")}, strongModel, "code"},
		{"go as a verb", []message{user("Should I go hiking or write a letter to my aunt?")}, cheapModel, "simple"},
		{"stack trace", []message{user("Traceback (most recent call last):\n  File \"app.py\", line 3")}, strongModel, "code"},
		{"arithmetic", []message{user("What is 17 * 23?")}, strongModel, "math"},
		{"equation", []message{user("Solve 2x + 3 = 11 for x.")}, strongModel, "math"},
		{"latex", []message{user(`Evaluate \int_0^1 x dx`)}, strongModel, "math"},
		{"follow-up to code", []message{user("Write a Go function that sums a slice."),
			assistant("```go\nfunc sum(xs []int) int\n```"), user("Now make it generic.")}, strongModel, "code"},
		{"older code ignored", []message{user("Write code to sort a list in Python."), assistant("Done."),
			user("Thanks! What is the capital of France?")}, cheapModel, "simple"},
		{"image", []message{withImage}, visionModel, "image"},
		{"earlier image", []message{withImage, assistant("A cat."), user("What color is it?")}, visionModel, "image"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if m, reason := routeByRules(tc.messages); m.ID != tc.model || reason != tc.reason {
				t.Fatalf("routed to %s (%s), want %s (%s)", m.ID, reason, tc.model, tc.reason)
			}
		})
	}
}

func TestModelRouter(t *testing.T) {
	var calls atomic.Int32
	var received atomic.Value
	router := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
		}
		received.Store(body)
		switch body["text"] {
		case "fail":
			http.Error(w, "unavailable", http.StatusInternalServerError)
		case "unknown":
			io.WriteString(w, `{"label":"poetry"}`)
		case "slow":
			time.Sleep(200 * time.Millisecond)
		default:
			io.WriteString(w, `{"label":"math","scores":{"simple":0.1,"code":0.2,"math":0.7}}`)
		}
	}))
	defer router.Close()
	s := newService(ledger(t), nil, &url.URL{})
	s.router = &promptRouter{url: router.URL, client: &http.Client{Timeout: 50 * time.Millisecond}}
	user := func(text string) message { return message{Role: "user", Content: content{text: text}} }
	assistant := func(text string) message { return message{Role: "assistant", Content: content{text: text}} }
	withImage := message{Role: "user", Content: content{parts: []contentPart{
		{Type: "image_url", ImageURL: &imageURL{URL: "data:image/png;base64,"}}, {Type: "text", Text: "Hi"}}}}
	for _, tc := range []struct {
		name                  string
		messages              []message
		model, reason, source string
		calls                 int32
	}{
		// The rules alone would call "Say hello." simple; the router's answer wins.
		{"router decides", []message{user("Say hello.")}, strongModel, "math", "model", 1},
		{"router error", []message{user("fail")}, cheapModel, "simple", "rules", 1},
		{"unknown label", []message{user("unknown")}, cheapModel, "simple", "rules", 1},
		{"timeout", []message{user("slow")}, cheapModel, "simple", "rules", 1},
		{"images skip the router", []message{withImage}, visionModel, "image", "rules", 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := calls.Load()
			m, reason, source := s.route(context.Background(), tc.messages)
			if m.ID != tc.model || reason != tc.reason || source != tc.source || calls.Load()-before != tc.calls {
				t.Fatalf("got %s (%s, %s) after %d calls", m.ID, reason, source, calls.Load()-before)
			}
		})
	}
	long := strings.Repeat("é", maxRouterChars+10)
	s.route(context.Background(), []message{user("old question"), assistant(long), user("new question")})
	if body := received.Load().(map[string]string); body["text"] != "new question" ||
		body["context"] != strings.Repeat("é", maxRouterChars) {
		t.Fatalf("router received text %q and %d-byte context", body["text"], len(body["context"]))
	}
}

func TestModelSelectionAndPricing(t *testing.T) {
	var forwarded atomic.Value
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		body, _ := io.ReadAll(r.Body)
		forwarded.Store(body)
		io.WriteString(w, `{"choices":[]}`)
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	img := dataURL(t, "image/png", image.NewGray(image.Rect(0, 0, 8, 8)))
	chat := func(model, content string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":` + content + `}],"max_tokens":4096}`
	}
	imageContent := `[{"type":"image_url","image_url":{"url":"` + img + `"}},{"type":"text","text":"Describe it."}]`
	for _, tc := range []struct {
		name, body, model, reason, effort string
		status                            int
		multiplier                        int64
	}{
		{"auto simple", chat("auto", `"Say hello."`), cheapModel, "simple", "low", 200, 1},
		{"auto code", chat("auto", `"Write a Python function to merge two sorted lists."`), strongModel, "code", "medium", 200, 3},
		{"auto image", chat("auto", imageContent), visionModel, "image", "", 200, 1},
		{"explicit model", chat(strongModel, `"Say hello."`), strongModel, "requested", "medium", 200, 3},
		{"image to text-only model", chat(cheapModel, imageContent), "", "", "", 400, 0},
		{"output above cap", strings.Replace(chat("auto", `"Hi"`), "4096", "4097", 1), "", "", "", 400, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := ledger(t)
			c := testCaller("customer", "secret")
			s := newService(db, map[string]caller{c.KeySHA256: c}, upstream)
			var logs bytes.Buffer
			s.audit = slog.New(slog.NewJSONHandler(&logs, nil))
			before := calls.Load()
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			r.Header.Set("Authorization", "Bearer secret")
			w := httptest.NewRecorder()
			s.ServeHTTP(w, r)
			if w.Code != tc.status {
				t.Fatalf("status=%d (%s), want %d", w.Code, w.Body, tc.status)
			}
			if tc.status != 200 {
				if calls.Load() != before {
					t.Fatal("rejected request reached Ray")
				}
				return
			}
			var sent map[string]any
			if err := json.Unmarshal(forwarded.Load().([]byte), &sent); err != nil {
				t.Fatal(err)
			}
			effort, _ := sent["reasoning_effort"].(string)
			if sent["model"] != tc.model || effort != tc.effort {
				t.Fatalf("forwarded model=%v reasoning_effort=%q, want %s %q", sent["model"], effort, tc.model, tc.effort)
			}
			charge := contextTokens * c.TokenPriceMicrousd * tc.multiplier
			if w.Header().Get("X-Routed-Model") != tc.model || w.Header().Get("X-Route-Reason") != tc.reason ||
				w.Header().Get("X-Quota-Reserved-Microusd") != strconv.FormatInt(charge, 10) {
				t.Fatalf("headers: %v", w.Header())
			}
			var ledgerCharge int64
			if err := db.QueryRow("SELECT charge FROM daily WHERE caller=?", c.ID).Scan(&ledgerCharge); err != nil || ledgerCharge != charge {
				t.Fatalf("ledger charge=%d (%v), want %d", ledgerCharge, err, charge)
			}
			records := auditRecords(t, &logs)
			a := records[len(records)-1].Call
			wantRequested, wantSource := tc.model, ""
			if tc.reason != "requested" {
				wantRequested, wantSource = autoModel, "rules"
			}
			if a.RequestedModel != wantRequested || a.Model != tc.model || a.RouteReason != tc.reason ||
				a.RouteSource != wantSource || w.Header().Get("X-Route-Source") != wantSource ||
				a.PriceMultiplier != tc.multiplier || a.ChargeMicrousd != charge {
				t.Fatalf("audit routing fields: %+v", a)
			}
		})
	}
}

func TestModelListIncludesAuto(t *testing.T) {
	c := testCaller("customer", "secret")
	upstream, _ := url.Parse("http://backend.invalid")
	s := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)
	s.proxy.Transport = auditRoundTripper(func(*http.Request) (*http.Response, error) {
		t.Fatal("model listing reached Ray")
		return nil, nil
	})
	s.audit = slog.New(slog.NewJSONHandler(io.Discard, nil))
	r := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	r.Header.Set("Authorization", "Bearer secret")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	var list struct {
		Data []struct{ ID string } `json:"data"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &list); err != nil || w.Code != 200 {
		t.Fatalf("status=%d err=%v", w.Code, err)
	}
	var ids []string
	for _, m := range list.Data {
		ids = append(ids, m.ID)
	}
	if strings.Join(ids, ",") != strings.Join(modelList, ",") || ids[0] != autoModel {
		t.Fatalf("models=%v", ids)
	}
}
