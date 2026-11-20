package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"image"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"
)

type auditRecord struct {
	Message string    `json:"msg"`
	Call    callAudit `json:"call"`
}

func auditRecords(t *testing.T, logs *bytes.Buffer) []auditRecord {
	t.Helper()
	var records []auditRecord
	dec := json.NewDecoder(strings.NewReader(logs.String()))
	for {
		var record auditRecord
		if err := dec.Decode(&record); err == io.EOF {
			return records
		} else if err != nil {
			t.Fatalf("invalid JSON audit log: %v", err)
		}
		records = append(records, record)
	}
}

func TestCallAudit(t *testing.T) {
	const response = `{"choices":[{"message":{"content":"private-answer"}}]}`
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Request-ID") == "" || r.Header.Get("X-Request-ID") == "untrusted-id" {
			t.Error("generated request ID not forwarded")
		}
		w.Header().Set("X-Request-ID", "backend-id")
		if r.Header.Get("Accept") == "fail" {
			w.WriteHeader(http.StatusInternalServerError)
		}
		if r.Header.Get("Accept") == "text/event-stream" {
			w.Header().Set("Content-Type", "text/event-stream")
			io.WriteString(w, "data: "+response+"\n\n")
			w.(http.Flusher).Flush()
			io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		io.WriteString(w, response)
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	for _, tc := range []struct {
		name, key, body, accept, outcome string
		status                           int
		admitted                         bool
	}{
		{"stream", "private-key", validChat, "text/event-stream", "completed", 200, true},
		{"json", "private-key", strings.Replace(validChat, "true", "false", 1), "", "completed", 200, true},
		{"upstream error", "private-key", validChat, "fail", "upstream_error", 500, true},
		{"unauthenticated", "wrong-private-key", validChat, "", "rejected", 401, false},
		{"invalid request", "private-key", `{"private-invalid":true}`, "", "rejected", 400, false},
		{"quota denied", "private-key", validChat, "", "quota_exceeded", 429, false},
		{"ledger unavailable", "private-key", validChat, "", "ledger_error", 503, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			db := ledger(t)
			c := testCaller("customer-42", "private-key")
			if tc.name == "quota denied" {
				c.DailyTokens = 1
			}
			if tc.name == "ledger unavailable" {
				db.Close()
			}
			s := newService(db, map[string]caller{c.KeySHA256: c}, upstream)
			var logs bytes.Buffer
			s.audit = slog.New(slog.NewJSONHandler(&logs, nil))
			now := time.Date(2026, 11, 20, 1, 0, 0, 0, time.UTC)
			s.now = func() time.Time { return now }
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			r.Header.Set("Authorization", "Bearer "+tc.key)
			r.Header.Set("Accept", tc.accept)
			r.Header.Set("X-Request-ID", "untrusted-id")
			r.Header.Set("Cookie", "private-cookie")
			w := httptest.NewRecorder()
			s.ServeHTTP(w, r)
			if w.Code != tc.status {
				t.Fatalf("status=%d, want %d", w.Code, tc.status)
			}
			records := auditRecords(t, &logs)
			wantEvents := 1
			if tc.admitted {
				wantEvents = 2
			}
			if len(records) != wantEvents {
				t.Fatalf("events=%d, want %d", len(records), wantEvents)
			}
			final := records[len(records)-1]
			a := final.Call
			if final.Message != "inference_request" || a.Outcome != tc.outcome || a.Status != tc.status ||
				a.RequestID == "" || a.RequestID != w.Header().Get("X-Request-ID") ||
				!a.StartedAt.Equal(now) || a.ResponseBytes != int64(w.Body.Len()) || a.DurationMS < 0 {
				t.Fatalf("incorrect completion: %+v", final)
			}
			if tc.admitted {
				first := records[0]
				if first.Message != "inference_admitted" || first.Call.Outcome != "admitted" ||
					first.Call.RequestID != a.RequestID || first.Call.Status != 0 ||
					a.ReservedTokens != contextTokens || a.ChargeMicrousd != contextTokens*c.TokenPriceMicrousd ||
					first.Call.ChargeMicrousd != a.ChargeMicrousd || a.TokenPriceMicrousd != c.TokenPriceMicrousd ||
					a.Model != visionModel || a.CallerID != c.ID || a.MessageCount != 1 || a.MaxTokens != 32 ||
					a.Stream != strings.Contains(tc.body, `"stream":true`) || a.TextBytes != len("Hello") {
					t.Fatalf("incorrect admitted call: %+v / %+v", first, a)
				}
				var charge int64
				if err := db.QueryRow("SELECT charge FROM daily WHERE caller=?", c.ID).Scan(&charge); err != nil {
					t.Fatal(err)
				}
				if charge != a.ChargeMicrousd {
					t.Fatalf("audit charge %d differs from ledger %d", a.ChargeMicrousd, charge)
				}
			} else if a.ChargeMicrousd != 0 || a.ReservedTokens != 0 {
				t.Fatalf("rejected request logged a charge: %+v", a)
			}
			if tc.status == 401 && a.CallerID != "" {
				t.Fatal("unauthenticated caller was attributed an identity")
			}
			for _, secret := range []string{"private-key", "private-cookie", "Hello", "private-answer", "private-invalid", c.KeySHA256, "untrusted-id", "backend-id"} {
				if strings.Contains(logs.String(), secret) {
					t.Fatalf("sensitive content %q leaked to logs", secret)
				}
			}
		})
	}
}

type auditRoundTripper func(*http.Request) (*http.Response, error)

func (f auditRoundTripper) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type brokenAuditBody struct{}

func (brokenAuditBody) Read([]byte) (int, error) { return 0, errors.New("broken stream") }
func (brokenAuditBody) Close() error             { return nil }

type failedAuditWriter struct{ *httptest.ResponseRecorder }

func (failedAuditWriter) Write([]byte) (int, error) { return 0, errors.New("client disconnected") }

func TestAuditTransportFailures(t *testing.T) {
	for _, mode := range []string{"transport", "canceled", "interrupted", "write_error"} {
		t.Run(mode, func(t *testing.T) {
			c := testCaller("customer", "secret")
			upstream, _ := url.Parse("http://backend.invalid")
			s := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)
			var logs bytes.Buffer
			s.audit = slog.New(slog.NewJSONHandler(&logs, nil))
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			s.proxy.Transport = auditRoundTripper(func(*http.Request) (*http.Response, error) {
				records := auditRecords(t, &logs)
				if len(records) != 1 || records[0].Message != "inference_admitted" {
					t.Fatal("admission not logged before upstream call")
				}
				if mode == "interrupted" {
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: brokenAuditBody{}}, nil
				}
				if mode == "write_error" {
					return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("data: chunk\n\n"))}, nil
				}
				if mode == "canceled" {
					cancel()
				}
				return nil, errors.New("backend unavailable")
			})
			// ReverseProxy aborts incomplete responses when running under an HTTP server.
			ctx = context.WithValue(ctx, http.ServerContextKey, &http.Server{})
			r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(validChat)).WithContext(ctx)
			r.Header.Set("Authorization", "Bearer secret")
			func() {
				defer func() {
					if p := recover(); p != nil && ((mode != "interrupted" && mode != "write_error") || p != http.ErrAbortHandler) {
						panic(p)
					}
				}()
				var w http.ResponseWriter = httptest.NewRecorder()
				if mode == "write_error" {
					w = failedAuditWriter{httptest.NewRecorder()}
				}
				s.ServeHTTP(w, r)
			}()
			records := auditRecords(t, &logs)
			wantOutcome, wantStatus := "upstream_error", 502
			if mode == "canceled" {
				wantOutcome = "canceled"
			} else if mode == "interrupted" {
				wantOutcome, wantStatus = "interrupted", 200
			} else if mode == "write_error" {
				wantOutcome, wantStatus = "write_error", 200
			}
			if len(records) != 2 || records[1].Call.Outcome != wantOutcome ||
				records[1].Call.Status != wantStatus || records[1].Call.ChargeMicrousd != contextTokens*c.TokenPriceMicrousd {
				t.Fatalf("incorrect failure audit: %+v", records)
			}
		})
	}
}

func TestAuditPreservesIncrementalStreaming(t *testing.T) {
	release := make(chan struct{})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: first\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-release:
			io.WriteString(w, "data: [DONE]\n\n")
		case <-r.Context().Done():
		}
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	c := testCaller("customer", "secret")
	s := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)
	s.audit = slog.New(slog.NewJSONHandler(io.Discard, nil))
	proxy := httptest.NewServer(s)
	defer func() {
		close(release)
		proxy.Close()
	}()
	req, err := http.NewRequest(http.MethodPost, proxy.URL+"/v1/chat/completions", strings.NewReader(validChat))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer secret")
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	line, err := bufio.NewReader(resp.Body).ReadString('\n')
	if err != nil || line != "data: first\n" {
		t.Fatalf("first chunk not delivered before stream finishes: %q, %v", line, err)
	}
}

func TestAuditSafeContextAndUnchargedRoutes(t *testing.T) {
	c := testCaller("customer", "secret")
	upstream, _ := url.Parse("http://backend.invalid")
	s := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)
	s.proxy.Transport = auditRoundTripper(func(*http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: 200, Header: make(http.Header), Body: io.NopCloser(strings.NewReader("{}"))}, nil
	})
	var logs bytes.Buffer
	s.audit = slog.New(slog.NewJSONHandler(&logs, nil))
	ids := make(map[string]bool)
	for _, path := range []string{"/healthz", "/v1/models", "/private-path", "/v1/chat/completions?private-query=yes"} {
		logs.Reset()
		r := httptest.NewRequest(http.MethodGet, path, nil)
		r.Header.Set("Authorization", "Bearer secret")
		s.ServeHTTP(httptest.NewRecorder(), r)
		records := auditRecords(t, &logs)
		if path == "/healthz" {
			if len(records) != 0 {
				t.Fatal("health probe logged")
			}
			continue
		}
		if len(records) != 1 || records[0].Call.ChargeMicrousd != 0 || records[0].Call.ReservedTokens != 0 {
			t.Fatalf("uncharged route audit: %+v", records)
		}
		id := records[0].Call.RequestID
		if id == "" || ids[id] {
			t.Fatal("missing or reused request ID")
		}
		ids[id] = true
		if strings.Contains(logs.String(), "private-") {
			t.Fatal("arbitrary path or query leaked")
		}
	}
	logs.Reset()
	img := dataURL(t, "image/png", image.NewGray(image.Rect(0, 0, 1, 1)))
	body := `{"model":"qwen3.5-4b","messages":[{"role":"user","content":[{"type":"text","text":"private-text"},{"type":"image_url","image_url":{"url":"` + img + `"}}]}],"max_tokens":12,"temperature":0.5}`
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
	r.Header.Set("Authorization", "Bearer secret")
	s.ServeHTTP(httptest.NewRecorder(), r)
	records := auditRecords(t, &logs)
	if len(records) != 2 || records[1].Call.ImageCount != 1 || records[1].Call.TextBytes != len("private-text") ||
		records[1].Call.Temperature == nil || *records[1].Call.Temperature != 0.5 ||
		strings.Contains(logs.String(), img) || strings.Contains(logs.String(), "private-text") {
		t.Fatalf("unsafe or inaccurate image metadata: %s", logs.String())
	}
}
