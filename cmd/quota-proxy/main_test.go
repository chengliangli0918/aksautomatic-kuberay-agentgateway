package main

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testCaller(id, key string) caller {
	hash := sha256.Sum256([]byte(key))
	return caller{
		ID: id, KeySHA256: hex.EncodeToString(hash[:]), RequestsPerMinute: 60,
		DailyTokens: 100000, DailyBudgetMicrousd: 1000000, TokenPriceMicrousd: 10,
	}
}

func ledger(t *testing.T) *sql.DB {
	t.Helper()
	db, err := openLedger(filepath.Join(t.TempDir(), "usage.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestAtomicQuota(t *testing.T) {
	for _, limit := range []string{"requests", "tokens", "budget"} {
		t.Run(limit, func(t *testing.T) {
			db := ledger(t)
			c := testCaller("a", "secret")
			switch limit {
			case "requests":
				c.RequestsPerMinute = 2
			case "tokens":
				c.DailyTokens = 2*contextTokens + contextTokens - 1
			case "budget":
				c.DailyBudgetMicrousd = 2*contextTokens*c.TokenPriceMicrousd + 1
			}
			var accepted atomic.Int32
			var wg sync.WaitGroup
			now := time.Date(2026, 11, 20, 1, 0, 0, 0, time.UTC)
			for range 20 {
				wg.Add(1)
				go func() {
					defer wg.Done()
					err := reserve(context.Background(), db, c, now)
					if err == nil {
						accepted.Add(1)
					} else if !errors.Is(err, errQuota) {
						t.Errorf("unexpected reservation error: %v", err)
					}
				}()
			}
			wg.Wait()
			if accepted.Load() != 2 {
				t.Fatalf("accepted %d concurrent requests, want 2", accepted.Load())
			}
			var tokens, charge int64
			if err := db.QueryRow("SELECT tokens, charge FROM daily WHERE caller='a'").Scan(&tokens, &charge); err != nil {
				t.Fatal(err)
			}
			if tokens != 2*contextTokens || charge != 2*contextTokens*c.TokenPriceMicrousd {
				t.Fatalf("incorrect ledger totals: tokens=%d charge=%d", tokens, charge)
			}
		})
	}
}

func TestWindowsIdentityAndPersistence(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.db")
	db, err := openLedger(path)
	if err != nil {
		t.Fatal(err)
	}
	c := testCaller("stable-id", "old-key")
	c.RequestsPerMinute = 1
	c.DailyTokens = 2 * contextTokens
	now := time.Date(2026, 11, 20, 23, 57, 0, 0, time.UTC)
	if err := reserve(context.Background(), db, c, now); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	db, err = openLedger(path)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	rotated := testCaller("stable-id", "new-key")
	rotated.RequestsPerMinute, rotated.DailyTokens = c.RequestsPerMinute, c.DailyTokens
	if err := reserve(context.Background(), db, rotated, now); !errors.Is(err, errQuota) {
		t.Fatalf("restart/key rotation reset minute quota: %v", err)
	}
	if err := reserve(context.Background(), db, rotated, now.Add(time.Minute)); err != nil {
		t.Fatalf("minute did not reset: %v", err)
	}
	if err := reserve(context.Background(), db, rotated, now.Add(2*time.Minute)); !errors.Is(err, errQuota) {
		t.Fatalf("daily quota was not retained: %v", err)
	}
	if err := reserve(context.Background(), db, rotated, now.Add(3*time.Minute)); err != nil {
		t.Fatalf("UTC day did not reset: %v", err)
	}
	other := testCaller("other", "other-key")
	if err := reserve(context.Background(), db, other, now); err != nil {
		t.Fatalf("callers are not isolated: %v", err)
	}
}

const validChat = `{"model":"qwen3.5-4b","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"stream":true}`

func TestHTTPAdmissionAndStreaming(t *testing.T) {
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		if r.Header.Get("Authorization") != "" || r.Header.Get("Cookie") != "" ||
			r.Header.Get("X-Api-Key") != "" || r.Header.Get("X-Forwarded-For") != "" {
			t.Error("caller credentials or forwarding headers leaked to Ray")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	db := ledger(t)
	c := testCaller("a", "secret")
	c.DailyTokens = contextTokens
	s := newService(db, map[string]caller{c.KeySHA256: c}, upstream)
	for _, tc := range []struct {
		name, key, body string
		status          int
	}{
		{"missing auth", "", validChat, 401},
		{"invalid auth", "wrong", validChat, 401},
		{"multiple outputs", "secret", strings.TrimSuffix(validChat, "}") + `,"n":2}`, 400},
		{"missing cap", "secret", strings.Replace(validChat, `"max_tokens":32,`, "", 1), 400},
		{"excess output", "secret", strings.Replace(validChat, `"max_tokens":32`, `"max_tokens":513`, 1), 400},
		{"other model", "secret", strings.Replace(validChat, modelID, "other-model", 1), 400},
		{"system not first", "secret", strings.Replace(validChat, `"content":"Hello"}`,
			`"content":"Hello"},{"role":"system","content":"Be brief."}`, 1), 400},
		{"no user message", "secret", strings.Replace(validChat, `"role":"user"`, `"role":"system"`, 1), 400},
		{"trailing JSON", "secret", validChat + `{}`, 400},
		{"oversized", "secret", strings.Replace(validChat, "Hello", strings.Repeat("x", 65537), 1), 400},
		{"accepted stream", "secret", validChat, 200},
		{"exhausted", "secret", validChat, 429},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(tc.body))
			if tc.key != "" {
				req.Header.Set("Authorization", "Bearer "+tc.key)
			}
			req.Header.Set("Cookie", "sensitive")
			req.Header.Set("X-Api-Key", "sensitive")
			req.Header.Set("X-Forwarded-For", "untrusted")
			w := httptest.NewRecorder()
			s.ServeHTTP(w, req)
			if w.Code != tc.status {
				t.Fatalf("got %d (%s), want %d", w.Code, w.Body, tc.status)
			}
			if tc.status == 200 && (!strings.Contains(w.Body.String(), "data: [DONE]") ||
				w.Header().Get("X-Quota-Reserved-Tokens") != "8192") {
				t.Fatal("stream or reservation metadata was lost")
			}
		})
	}
	if calls.Load() != 1 {
		t.Fatalf("backend invoked %d times, want 1", calls.Load())
	}
}

// dataURL encodes img as a base64 data URL of mediaType (image/png or image/jpeg).
func dataURL(t *testing.T, mediaType string, img image.Image) string {
	t.Helper()
	var buf bytes.Buffer
	encode := func() error { return png.Encode(&buf, img) }
	if mediaType == "image/jpeg" {
		encode = func() error { return jpeg.Encode(&buf, img, nil) }
	}
	if err := encode(); err != nil {
		t.Fatal(err)
	}
	return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func TestImageMessages(t *testing.T) {
	var forwarded atomic.Value
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		forwarded.Store(string(body))
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	c := testCaller("a", "secret")
	s := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)

	small := image.NewGray(image.Rect(0, 0, 64, 48))
	pngURL, jpegURL := dataURL(t, "image/png", small), dataURL(t, "image/jpeg", small)
	raw, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(pngURL, "data:image/png;base64,"))
	oversized := "data:image/png;base64," + base64.StdEncoding.EncodeToString(append(raw, make([]byte, maxImageBytes)...))
	img := func(url string) string { return `{"type":"image_url","image_url":{"url":"` + url + `"}}` }
	user := func(parts ...string) string { return `{"role":"user","content":[` + strings.Join(parts, ",") + `]}` }
	const text = `{"type":"text","text":"What is this?"}`
	for _, tc := range []struct {
		name, messages string
		status         int
	}{
		{"png with text", user(img(pngURL), text), 200},
		{"jpeg only", user(img(jpegURL)), 200},
		{"remote URL", user(img("https://example.com/cat.png")), 400},
		{"mismatched type", user(img(strings.Replace(pngURL, "image/png", "image/jpeg", 1))), 400},
		{"not an image", user(img("data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte("hello")))), 400},
		{"line break in base64", user(img(strings.Replace(pngURL, "base64,", `base64,\n`, 1))), 400},
		{"too many bytes", user(img(oversized)), 400},
		{"too many pixels", user(img(dataURL(t, "image/png", image.NewGray(image.Rect(0, 0, 4097, 4096))))), 400},
		{"extreme aspect ratio", user(img(dataURL(t, "image/png", image.NewGray(image.Rect(0, 0, 201, 1))))), 400},
		{"too many images", user(img(pngURL), img(pngURL)) + `,{"role":"assistant","content":"A square."},` + user(img(pngURL)), 400},
		{"image from assistant", `{"role":"user","content":"Hi"},{"role":"assistant","content":[` + img(pngURL) + `]}`, 400},
		{"empty parts", `{"role":"user","content":[]}`, 400},
		{"empty text part", user(`{"type":"text","text":""}`), 400},
		{"unknown part field", user(`{"type":"text","text":"Hi","detail":"high"}`), 400},
	} {
		t.Run(tc.name, func(t *testing.T) {
			forwarded.Store("")
			body := `{"model":"` + modelID + `","messages":[` + tc.messages + `],"max_tokens":32}`
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(body))
			req.Header.Set("Authorization", "Bearer secret")
			w := httptest.NewRecorder()
			s.ServeHTTP(w, req)
			if w.Code != tc.status {
				t.Fatalf("got %d (%s), want %d", w.Code, w.Body, tc.status)
			}
			got := forwarded.Load().(string)
			if tc.status == 200 && !strings.Contains(got, `"image_url":{"url":"data:image/`) {
				t.Fatalf("image part was not forwarded: %.200s", got)
			}
			if tc.status != 200 && got != "" {
				t.Fatal("rejected request reached the backend")
			}
		})
	}
}

func TestFailClosedAndNoRefund(t *testing.T) {
	db := ledger(t)
	c := testCaller("a", "secret")
	c.DailyTokens = contextTokens
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "failed", http.StatusInternalServerError)
	}))
	defer backend.Close()
	upstream, _ := url.Parse(backend.URL)
	s := newService(db, map[string]caller{c.KeySHA256: c}, upstream)
	request := func() int {
		r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(validChat))
		r.Header.Set("Authorization", "Bearer secret")
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		return w.Code
	}
	if status := request(); status != 500 {
		t.Fatalf("backend error not preserved: %d", status)
	}
	if status := request(); status != 429 {
		t.Fatalf("failed inference refunded reservation: %d", status)
	}
	db.Close()
	if status := request(); status != 503 {
		t.Fatalf("unavailable ledger did not fail closed: %d", status)
	}
}

func TestCallerConfiguration(t *testing.T) {
	c := testCaller("a", "secret")
	encode := func(entries []caller) string {
		b, err := json.Marshal(entries)
		if err != nil {
			t.Fatal(err)
		}
		return string(b)
	}
	if _, err := loadCallers(strings.NewReader(encode([]caller{c}))); err != nil {
		t.Fatal(err)
	}
	bad := c
	bad.TokenPriceMicrousd = 0
	for _, input := range []string{
		`[]`, encode([]caller{bad}), encode([]caller{c, c}),
		encode([]caller{c}) + `[]`, `[{"unknown":1}]`,
	} {
		if _, err := loadCallers(strings.NewReader(input)); err == nil {
			t.Fatalf("invalid configuration accepted: %s", input)
		}
	}
}

func TestDeploymentCallers(t *testing.T) {
	path := os.Getenv("CALLERS_FILE")
	if path == "" {
		t.Skip("CALLERS_FILE is not set")
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := loadCallers(f); err != nil {
		t.Fatalf("invalid deployment caller policy: %v", err)
	}
}
