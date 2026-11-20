package main

import (
	"crypto/rand"
	"log/slog"
	"net/http"
	"time"
)

type auditContextKey struct{}

// Audit metadata deliberately excludes message contents and client-controlled headers.
type callAudit struct {
	RequestID          string    `json:"request_id"`
	StartedAt          time.Time `json:"started_at"`
	Endpoint           string    `json:"endpoint"`
	CallerID           string    `json:"caller_id,omitempty"`
	RequestedModel     string    `json:"requested_model,omitempty"`
	Model              string    `json:"model,omitempty"`
	RouteReason        string    `json:"route_reason,omitempty"`
	RouteSource        string    `json:"route_source,omitempty"`
	Stream             bool      `json:"stream"`
	MessageCount       int       `json:"message_count"`
	ImageCount         int       `json:"image_count"`
	TextBytes          int       `json:"text_bytes"`
	MaxTokens          int       `json:"max_tokens"`
	Temperature        *float64  `json:"temperature,omitempty"`
	ReservedTokens     int64     `json:"reserved_tokens"`
	TokenPriceMicrousd int64     `json:"token_price_microusd"`
	PriceMultiplier    int64     `json:"price_multiplier,omitempty"`
	ChargeMicrousd     int64     `json:"charge_microusd"`
	CostBasis          string    `json:"cost_basis"`
	Outcome            string    `json:"outcome"`
	Status             int       `json:"status"`
	ResponseBytes      int64     `json:"response_bytes"`
	DurationMS         int64     `json:"duration_ms"`
}

func newCallAudit(r *http.Request, now time.Time) *callAudit {
	endpoint := "unsupported"
	switch r.URL.Path {
	case "/v1/chat/completions", "/v1/models":
		endpoint = r.URL.Path
	}
	return &callAudit{
		RequestID: rand.Text(), StartedAt: now.UTC(), Endpoint: endpoint,
		CostBasis: "reserved_context_chargeback", Outcome: "rejected",
	}
}

// setRequest records the validated request and the model selected for it.
func (a *callAudit) setRequest(body chatRequest, m modelSpec, route, source string) {
	a.RequestedModel, a.Model, a.RouteReason, a.RouteSource = body.Model, m.ID, route, source
	a.PriceMultiplier, a.Stream = m.PriceMultiplier, body.Stream
	a.MessageCount, a.MaxTokens, a.Temperature = len(body.Messages), body.MaxTokens, body.Temperature
	for _, m := range body.Messages {
		a.TextBytes += len(m.Content.text)
		for _, p := range m.Content.parts {
			a.TextBytes += len(p.Text)
			if p.Type == "image_url" {
				a.ImageCount++
			}
		}
	}
}

func (s *service) finishAudit(r *http.Request, w *auditWriter, a *callAudit, start time.Time) {
	if a.Outcome == "forwarding" {
		a.Outcome = "interrupted"
	}
	if r.Context().Err() != nil {
		a.Outcome = "canceled"
	} else if w.writeFailed {
		a.Outcome = "write_error"
	}
	a.Status, a.ResponseBytes = w.status, w.bytes
	a.DurationMS = time.Since(start).Milliseconds()
	s.audit.LogAttrs(r.Context(), slog.LevelInfo, "inference_request", slog.Any("call", a))
}

type auditWriter struct {
	http.ResponseWriter
	status      int
	bytes       int64
	writeFailed bool
}

func (w *auditWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *auditWriter) WriteHeader(status int) {
	if w.status == 0 && status >= 200 {
		w.status = status
	}
	w.ResponseWriter.WriteHeader(status)
}

func (w *auditWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	n, err := w.ResponseWriter.Write(p)
	w.bytes += int64(n)
	w.writeFailed = w.writeFailed || err != nil
	return n, err
}

func (w *auditWriter) FlushError() error {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	err := http.NewResponseController(w.ResponseWriter).Flush()
	w.writeFailed = w.writeFailed || err != nil
	return err
}

func (w *auditWriter) Flush() { _ = w.FlushError() }
