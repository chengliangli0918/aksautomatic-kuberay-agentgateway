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
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/signal"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	_ "modernc.org/sqlite"
)

// Reserve the complete engine context, not a tokenizer-dependent estimate.
const contextTokens int64 = 8192
const modelID = "qwen3.5-4b"

const (
	maxBodyBytes = 12 << 20
	maxTextBytes = 64 << 10
	// Matches limit_mm_per_prompt in deploy/rayservice.yaml.
	maxImages      = 2
	maxImageBytes  = 4 << 20
	maxImagePixels = 4096 * 4096
	// Bodies decoded at once; each 12 MiB body needs tens of MiB while it is validated.
	decodeSlots = 4
)

type caller struct {
	ID                  string `json:"id"`
	KeySHA256           string `json:"key_sha256"`
	RequestsPerMinute   int64  `json:"requests_per_minute"`
	DailyTokens         int64  `json:"daily_tokens"`
	DailyBudgetMicrousd int64  `json:"daily_budget_microusd"`
	TokenPriceMicrousd  int64  `json:"token_price_microusd"`
}

func loadCallers(r io.Reader) (map[string]caller, error) {
	var entries []caller
	dec := json.NewDecoder(r)
	dec.DisallowUnknownFields()
	if err := dec.Decode(&entries); err != nil {
		return nil, err
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return nil, errors.New("caller configuration must contain one JSON value")
	}
	if len(entries) == 0 {
		return nil, errors.New("at least one caller is required")
	}
	result := make(map[string]caller)
	ids := make(map[string]bool)
	for _, c := range entries {
		key, err := hex.DecodeString(c.KeySHA256)
		if err != nil || len(key) != sha256.Size || c.KeySHA256 != strings.ToLower(c.KeySHA256) ||
			c.KeySHA256 == strings.Repeat("0", 64) || strings.TrimSpace(c.ID) == "" ||
			c.RequestsPerMinute <= 0 || c.DailyTokens <= 0 || c.DailyBudgetMicrousd <= 0 ||
			c.TokenPriceMicrousd <= 0 || c.TokenPriceMicrousd > 1_000_000_000 {
			return nil, errors.New("invalid caller identity, key hash, or positive limits")
		}
		if _, exists := result[c.KeySHA256]; exists || ids[c.ID] {
			return nil, errors.New("duplicate caller ID or key hash")
		}
		result[c.KeySHA256] = c
		ids[c.ID] = true
	}
	return result, nil
}

func openLedger(path string) (*sql.DB, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	_, err = db.Exec(`
		PRAGMA journal_mode=WAL;
		PRAGMA synchronous=FULL;
		PRAGMA busy_timeout=5000;
		CREATE TABLE IF NOT EXISTS daily (
			caller TEXT NOT NULL, day TEXT NOT NULL,
			tokens INTEGER NOT NULL, charge INTEGER NOT NULL,
			PRIMARY KEY (caller, day)
		);
		CREATE TABLE IF NOT EXISTS minute (
			caller TEXT NOT NULL, minute TEXT NOT NULL,
			requests INTEGER NOT NULL,
			PRIMARY KEY (caller, minute)
		);
	`)
	if err != nil {
		db.Close()
		return nil, err
	}
	return db, nil
}

var errQuota = errors.New("caller quota exceeded")

func reserve(ctx context.Context, db *sql.DB, c caller, now time.Time) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()
	day := now.UTC().Format("2006-01-02")
	minute := now.UTC().Format("2006-01-02T15:04")
	var tokens, charge, requests int64
	err = tx.QueryRowContext(ctx, "SELECT tokens, charge FROM daily WHERE caller=? AND day=?", c.ID, day).
		Scan(&tokens, &charge)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return err
	}
	err = tx.QueryRowContext(ctx, "SELECT requests FROM minute WHERE caller=? AND minute=?", c.ID, minute).
		Scan(&requests)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return err
	}
	cost := contextTokens * c.TokenPriceMicrousd
	if requests >= c.RequestsPerMinute || tokens > c.DailyTokens-contextTokens ||
		charge > c.DailyBudgetMicrousd-cost {
		return errQuota
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO daily VALUES (?, ?, ?, ?)
		ON CONFLICT(caller, day) DO UPDATE SET tokens=tokens+excluded.tokens, charge=charge+excluded.charge`,
		c.ID, day, contextTokens, cost)
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO minute VALUES (?, ?, 1)
		ON CONFLICT(caller, minute) DO UPDATE SET requests=requests+1`, c.ID, minute)
	if err != nil {
		return err
	}
	return tx.Commit()
}

type imageURL struct {
	URL string `json:"url"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

// content is a string or, in user messages, a list of text and image parts.
type content struct {
	text  string
	parts []contentPart
}

func (c *content) UnmarshalJSON(data []byte) error {
	if len(data) > 0 && data[0] == '"' {
		return json.Unmarshal(data, &c.text)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c.parts); err != nil {
		return err
	}
	if len(c.parts) == 0 {
		return errors.New("content parts must not be empty")
	}
	return nil
}

func (c content) MarshalJSON() ([]byte, error) {
	if c.parts != nil {
		return json.Marshal(c.parts)
	}
	return json.Marshal(c.text)
}

type message struct {
	Role    string  `json:"role"`
	Content content `json:"content"`
}

type chatRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Stream      bool      `json:"stream,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
}

// checkMessages rejects what the model's chat template or the engine's image limits would
// reject, so callers are not charged for requests that cannot succeed.
func checkMessages(messages []message) error {
	hasUser, images, textBytes := false, 0, 0
	for i, m := range messages {
		if m.Role != "user" && m.Role != "assistant" && (m.Role != "system" || i > 0) {
			return errors.New("messages require system/user/assistant roles and the system message first")
		}
		hasUser = hasUser || m.Role == "user"
		textBytes += len(m.Content.text)
		if m.Content.parts == nil {
			if m.Content.text == "" {
				return errors.New("messages require nonempty content")
			}
			continue
		}
		if m.Role != "user" {
			return errors.New("only user messages can contain content parts")
		}
		for _, p := range m.Content.parts {
			textBytes += len(p.Text)
			switch {
			case p.Type == "text" && p.Text != "" && p.ImageURL == nil:
			case p.Type == "image_url" && p.Text == "" && p.ImageURL != nil:
				if images++; images > maxImages {
					return fmt.Errorf("at most %d images are allowed per request", maxImages)
				}
				if err := checkImage(p.ImageURL.URL); err != nil {
					return err
				}
			default:
				return errors.New("content parts must be nonempty text or image_url parts")
			}
		}
	}
	if !hasUser {
		return errors.New("messages require at least one user message")
	}
	if textBytes > maxTextBytes {
		return errors.New("message text must total at most 64 KiB")
	}
	return nil
}

// checkImage accepts only inline PNG or JPEG data URLs, because vLLM would fetch any other URL
// from inside the cluster. It also bounds the images that vLLM decodes and resizes.
func checkImage(url string) error {
	mediaType, encoded, ok := strings.Cut(strings.TrimPrefix(url, "data:"), ";base64,")
	if !strings.HasPrefix(url, "data:") || !ok || (mediaType != "image/png" && mediaType != "image/jpeg") {
		return errors.New("images must be base64 PNG or JPEG data URLs")
	}
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || strings.ContainsAny(encoded, "\r\n") || len(data) > maxImageBytes {
		return errors.New("images must be valid base64 of at most 4 MiB")
	}
	config, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || "image/"+format != mediaType {
		return errors.New("image data is not a valid image of its declared type")
	}
	w, h := config.Width, config.Height
	if w < 1 || h < 1 || w*h > maxImagePixels || max(w, h) > 200*min(w, h) {
		return errors.New("images must be at most 16,777,216 pixels with an aspect ratio of at most 200:1")
	}
	return nil
}

// payloadBody drops the forwarded payload once the transport has sent and closed it, so a large
// image request does not hold that memory while its response streams.
type payloadBody struct{ reader atomic.Pointer[bytes.Reader] }

func newPayloadBody(payload []byte) *payloadBody {
	b := &payloadBody{}
	b.reader.Store(bytes.NewReader(payload))
	return b
}

func (b *payloadBody) Read(p []byte) (int, error) {
	reader := b.reader.Load()
	if reader == nil {
		return 0, io.EOF
	}
	return reader.Read(p)
}

func (b *payloadBody) Close() error {
	b.reader.Store(nil)
	return nil
}

type service struct {
	db       *sql.DB
	callers  map[string]caller
	proxy    *httputil.ReverseProxy
	now      func() time.Time
	decoding chan struct{}
	audit    *slog.Logger
}

func newService(db *sql.DB, callers map[string]caller, upstream *url.URL) *service {
	proxy := &httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(upstream)
			// Caller credentials and client-controlled forwarding headers never reach Ray.
			r.Out.Header = make(http.Header)
			r.Out.Header.Set("Content-Type", "application/json")
			r.Out.Header.Set("Accept", r.In.Header.Get("Accept"))
			if a, ok := r.In.Context().Value(auditContextKey{}).(*callAudit); ok {
				r.Out.Header.Set("X-Request-ID", a.RequestID)
			}
		},
		ModifyResponse: func(r *http.Response) error {
			// Keep the proxy-generated correlation ID on the client response.
			r.Header.Del("X-Request-ID")
			return nil
		},
		Transport: &http.Transport{
			Proxy:                 nil,
			MaxIdleConnsPerHost:   32,
			ResponseHeaderTimeout: 5 * time.Minute,
		},
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, _ *http.Request, err error) {
			log.Printf("inference upstream failure: %v", err)
			reject(w, http.StatusBadGateway, "inference backend unavailable; reservation retained")
		},
	}
	return &service{
		db: db, callers: callers, proxy: proxy, now: time.Now, decoding: make(chan struct{}, decodeSlots),
		audit: slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}
}

func reject(w http.ResponseWriter, status int, reason string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]string{"message": reason, "type": http.StatusText(status)},
	}); err != nil {
		log.Printf("writing error response: %v", err)
	}
}

func (s *service) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Path == "/healthz" {
		if err := s.db.PingContext(r.Context()); err != nil {
			log.Printf("ledger health failure: %v", err)
			reject(w, http.StatusServiceUnavailable, "quota ledger unavailable")
			return
		}
		w.WriteHeader(http.StatusOK)
		return
	}
	start := time.Now()
	a := newCallAudit(r, s.now())
	r = r.WithContext(context.WithValue(r.Context(), auditContextKey{}, a))
	aw := &auditWriter{ResponseWriter: w}
	w = aw
	w.Header().Set("X-Request-ID", a.RequestID)
	defer s.finishAudit(r, aw, a, start)
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		reject(w, http.StatusUnauthorized, "Bearer API key required")
		return
	}
	hash := sha256.Sum256([]byte(strings.TrimPrefix(auth, "Bearer ")))
	c, ok := s.callers[hex.EncodeToString(hash[:])]
	if !ok {
		reject(w, http.StatusUnauthorized, "invalid API key")
		return
	}
	a.CallerID, a.TokenPriceMicrousd = c.ID, c.TokenPriceMicrousd
	if r.URL.RawQuery != "" {
		reject(w, http.StatusBadRequest, "query parameters are not supported")
		return
	}
	if r.Method == http.MethodGet && r.URL.Path == "/v1/models" {
		a.Outcome = "forwarding"
		s.proxy.ServeHTTP(w, r)
		a.Outcome = "completed"
		if aw.status >= 400 {
			a.Outcome = "upstream_error"
		}
		return
	}
	if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
		reject(w, http.StatusNotFound, "only chat completions and model listing are supported")
		return
	}
	payload, status, reason := s.readChat(w, r)
	if status != 0 {
		reject(w, status, reason)
		return
	}
	if err := reserve(r.Context(), s.db, c, s.now()); err != nil {
		if errors.Is(err, errQuota) {
			a.Outcome = "quota_exceeded"
			reject(w, http.StatusTooManyRequests, "request, daily token, or daily chargeback budget exceeded")
		} else {
			a.Outcome = "ledger_error"
			log.Printf("quota reservation failed: %v", err)
			reject(w, http.StatusServiceUnavailable, "quota ledger unavailable")
		}
		return
	}
	a.ReservedTokens = contextTokens
	a.ChargeMicrousd = contextTokens * c.TokenPriceMicrousd
	a.Outcome = "admitted"
	s.audit.InfoContext(r.Context(), "inference_admitted", "call", a)
	r.Body = newPayloadBody(payload)
	r.ContentLength = int64(len(payload))
	w.Header().Set("X-Quota-Reserved-Tokens", fmt.Sprint(contextTokens))
	w.Header().Set("X-Quota-Reserved-Microusd", fmt.Sprint(contextTokens*c.TokenPriceMicrousd))
	a.Outcome = "forwarding"
	s.proxy.ServeHTTP(w, r)
	a.Outcome = "completed"
	if aw.status >= 400 {
		a.Outcome = "upstream_error"
	}
}

// readChat decodes and validates a chat request and returns the payload to forward, or an
// HTTP status and reason for rejecting it.
func (s *service) readChat(w http.ResponseWriter, r *http.Request) ([]byte, int, string) {
	select {
	case s.decoding <- struct{}{}:
		defer func() { <-s.decoding }()
	case <-r.Context().Done():
		return nil, http.StatusServiceUnavailable, "request canceled while waiting to be read"
	}
	r.Body = http.MaxBytesReader(w, r.Body, maxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	var body chatRequest
	if err := dec.Decode(&body); err != nil {
		return nil, http.StatusBadRequest, "invalid or oversized chat request"
	}
	if err := dec.Decode(new(any)); err != io.EOF {
		return nil, http.StatusBadRequest, "expected exactly one JSON object"
	}
	if body.Model != modelID || len(body.Messages) == 0 || len(body.Messages) > 128 ||
		body.MaxTokens < 1 || body.MaxTokens > 512 ||
		(body.Temperature != nil && (*body.Temperature < 0 || *body.Temperature > 2)) {
		return nil, http.StatusBadRequest, "unsupported model, messages, temperature, or max_tokens (1..512 required)"
	}
	if err := checkMessages(body.Messages); err != nil {
		return nil, http.StatusBadRequest, err.Error()
	}
	if a, ok := r.Context().Value(auditContextKey{}).(*callAudit); ok {
		a.setRequest(body)
	}
	payload, err := json.Marshal(body)
	if err != nil {
		log.Printf("encoding validated request: %v", err)
		return nil, http.StatusInternalServerError, "request encoding failed"
	}
	return payload, 0, ""
}

func run() error {
	f, err := os.Open("/config/callers.json")
	if err != nil {
		return err
	}
	defer f.Close()
	callers, err := loadCallers(f)
	if err != nil {
		return err
	}
	db, err := openLedger("/data/usage.db")
	if err != nil {
		return err
	}
	defer db.Close()
	upstream, err := url.Parse("http://vllm-serve-svc:8000")
	if err != nil {
		return err
	}
	server := &http.Server{
		Addr: ":8080", Handler: newService(db, callers, upstream),
		ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 30 * time.Second,
		WriteTimeout: 10 * time.Minute, IdleTimeout: 60 * time.Second,
		MaxHeaderBytes: 16 << 10,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()
	done := make(chan error, 1)
	go func() { done <- server.ListenAndServe() }()
	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()
		return server.Shutdown(shutdown)
	}
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}
