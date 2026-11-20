package main

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"image"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const publicHost = "inference.test.example"

func freePort(t *testing.T) int {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func writeTestCertificate(t *testing.T, dir string) *x509.CertPool {
	t.Helper()
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	template := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject:      pkix.Name{CommonName: publicHost},
		DNSNames:     []string{publicHost},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, template, template, &key.PublicKey, key)
	if err != nil {
		t.Fatal(err)
	}
	keyDER, err := x509.MarshalPKCS8PrivateKey(key)
	if err != nil {
		t.Fatal(err)
	}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	if err := os.WriteFile(filepath.Join(dir, "tls.crt"), certPEM, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tls.key"),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: keyDER}), 0o600); err != nil {
		t.Fatal(err)
	}
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(certPEM)
	return pool
}

// renderGatewayConfig rewrites only cluster addresses and file paths in the
// deployed config, failing if any expected value is missing or ambiguous.
func renderGatewayConfig(t *testing.T, replacements map[string]string) string {
	t.Helper()
	raw, err := os.ReadFile("../../config/agentgateway.yaml")
	if err != nil {
		t.Fatal(err)
	}
	config := string(raw)
	for old, replacement := range replacements {
		if n := strings.Count(config, old); n != 1 {
			t.Fatalf("deployed gateway config contains %q %d times, want exactly once", old, n)
		}
		config = strings.Replace(config, old, replacement, 1)
	}
	return config
}

func clientFrom(sourceIP string, port int, tlsConfig *tls.Config) *http.Client {
	dialer := &net.Dialer{Timeout: 2 * time.Second, LocalAddr: &net.TCPAddr{IP: net.ParseIP(sourceIP)}}
	return &http.Client{
		Timeout: 5 * time.Second,
		Transport: &http.Transport{
			DialContext: func(ctx context.Context, network, _ string) (net.Conn, error) {
				return dialer.DialContext(ctx, network, "127.0.0.1:"+strconv.Itoa(port))
			},
			TLSClientConfig:   tlsConfig,
			ForceAttemptHTTP2: tlsConfig != nil,
		},
	}
}

func send(t *testing.T, client *http.Client, method, target, key, body string) (int, string) {
	t.Helper()
	req, err := http.NewRequest(method, target, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	res, err := client.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	data, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	return res.StatusCode, string(data)
}

func TestPublicGatewayConfig(t *testing.T) {
	binary := os.Getenv("AGENTGATEWAY_BIN")
	if binary == "" {
		t.Skip("AGENTGATEWAY_BIN is not set")
	}
	var rayCalls atomic.Int32
	ray := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		rayCalls.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[]}\n\n")
		w.(http.Flusher).Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer ray.Close()
	upstream, _ := url.Parse(ray.URL)
	c := testCaller("a", "secret")
	c.DailyTokens = contextTokens
	var quotaCalls atomic.Int32
	quotaService := newService(ledger(t), map[string]caller{c.KeySHA256: c}, upstream)
	quota := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		quotaCalls.Add(1)
		quotaService.ServeHTTP(w, r)
	}))
	defer quota.Close()
	var solverHosts sync.Map
	// Reached through a hostname backend, as in the cluster, because agentgateway
	// rewrites Host for hostname backends unless the route disables it.
	solver := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		solverHosts.Store(r.Host, true)
		// Mirrors cert-manager's acmesolver, which rejects any other Host.
		if r.Host != publicHost || r.URL.Path != "/.well-known/acme-challenge/token-1" {
			http.NotFound(w, r)
			return
		}
		fmt.Fprint(w, "token-1.key-authorization")
	}))
	defer solver.Close()

	dir := t.TempDir()
	roots := writeTestCertificate(t, dir)
	acmePort, httpsPort := freePort(t), freePort(t)
	config := renderGatewayConfig(t, map[string]string{
		"port: 8080":   "port: " + strconv.Itoa(acmePort),
		"port: 8443":   "port: " + strconv.Itoa(httpsPort),
		"/tls/tls.crt": filepath.Join(dir, "tls.crt"),
		"/tls/tls.key": filepath.Join(dir, "tls.key"),
		"acme-http01-solver.inference-edge.svc.cluster.local:8089": "localhost:" + solver.URL[strings.LastIndex(solver.URL, ":")+1:],
		"quota-proxy.inference.svc.cluster.local:8080":             strings.TrimPrefix(quota.URL, "http://"),
	})
	configPath := filepath.Join(dir, "config.yaml")
	// Read-only, like the Kubernetes ConfigMap mount.
	if err := os.WriteFile(configPath, []byte(config), 0o400); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, "-f", configPath)
	output := &strings.Builder{}
	var outputMu sync.Mutex
	writer := writerFunc(func(p []byte) (int, error) {
		outputMu.Lock()
		defer outputMu.Unlock()
		return output.Write(p)
	})
	cmd.Stdout, cmd.Stderr = writer, writer
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cancel()
		_ = cmd.Wait()
		if t.Failed() {
			outputMu.Lock()
			t.Logf("gateway output:\n%s", output.String())
			outputMu.Unlock()
		}
	})

	tlsConfig := &tls.Config{RootCAs: roots, ServerName: publicHost, MinVersion: tls.VersionTLS12}
	https := clientFrom("127.0.0.1", httpsPort, tlsConfig)
	plain := clientFrom("127.0.0.1", acmePort, nil)
	apiURL := "https://" + publicHost
	ready := false
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); time.Sleep(50 * time.Millisecond) {
		req, _ := http.NewRequest(http.MethodGet, apiURL+"/", nil)
		if res, err := https.Do(req); err == nil {
			res.Body.Close()
			if res.StatusCode == http.StatusNotFound {
				ready = true
				break
			}
		}
	}
	if !ready {
		t.Fatal("gateway HTTPS listener did not become ready with the trusted certificate")
	}

	if status, body := send(t, plain, http.MethodGet, "http://"+publicHost+"/.well-known/acme-challenge/token-1", "", ""); status != 200 || body != "token-1.key-authorization" {
		t.Fatalf("ACME challenge through gateway: status=%d body=%q", status, body)
	}
	if _, ok := solverHosts.Load(publicHost); !ok {
		t.Fatal("gateway did not preserve the public Host header for the ACME solver")
	}
	if status, _ := send(t, plain, http.MethodPost, "http://"+publicHost+"/v1/chat/completions", "secret", validChat); status != 400 {
		t.Fatalf("plaintext API request was not rejected: %d", status)
	}
	if status, _ := send(t, https, http.MethodGet, apiURL+"/healthz", "", ""); status != 404 {
		t.Fatalf("non-API path was exposed: %d", status)
	}
	if quotaCalls.Load() != 0 {
		t.Fatalf("plaintext or non-API traffic reached quota-proxy %d times", quotaCalls.Load())
	}

	// A chat with a 3 MB image checks that the gateway passes bodies of pasted-image size.
	noise := image.NewRGBA(image.Rect(0, 0, 1000, 1000))
	if _, err := rand.Read(noise.Pix); err != nil {
		t.Fatal(err)
	}
	for i := 3; i < len(noise.Pix); i += 4 {
		noise.Pix[i] = 255
	}
	imageChat := `{"model":"` + modelID + `","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"` +
		dataURL(t, "image/png", noise) + `"}},{"type":"text","text":"Describe it."}]}],"max_tokens":32,"stream":true}`
	for _, tc := range []struct {
		key, body string
		status    int
	}{{"", validChat, 401}, {"secret", imageChat, 200}, {"secret", imageChat, 429}} {
		status, body := send(t, https, http.MethodPost, apiURL+"/v1/chat/completions", tc.key, tc.body)
		if status != tc.status {
			t.Fatalf("HTTPS chat with key %q: status=%d want=%d body=%s", tc.key, status, tc.status, body)
		}
		if status == 200 && !strings.Contains(body, "data: [DONE]") {
			t.Fatal("SSE stream lost through HTTPS gateway")
		}
		if status == 429 && !strings.Contains(body, "daily chargeback budget exceeded") {
			t.Fatalf("second chat was not rejected by the caller quota: %s", body)
		}
	}
	if rayCalls.Load() != 1 {
		t.Fatalf("inference backend called %d times, want 1", rayCalls.Load())
	}

	limited := 0
	for i := 1; i <= 400; i++ {
		status, _ := send(t, https, http.MethodGet, apiURL+"/v1/models", "", "")
		if status == http.StatusTooManyRequests {
			limited = i
			break
		}
		if status != http.StatusUnauthorized {
			t.Fatalf("unexpected status %d before per-IP limit", status)
		}
	}
	if limited == 0 {
		t.Fatal("per-IP rate limit never triggered within 400 requests")
	}
	if limited < 100 {
		t.Fatalf("per-IP limit triggered after %d requests; expected a burst of about 120", limited)
	}
	other := clientFrom("127.0.0.2", httpsPort, tlsConfig)
	if status, _ := send(t, other, http.MethodGet, apiURL+"/v1/models", "", ""); status != http.StatusUnauthorized {
		t.Fatalf("rate limit was not keyed by client IP: other client got %d", status)
	}
}

type writerFunc func([]byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }
