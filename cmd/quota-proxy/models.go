package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"
)

// modelSpec describes one model in deploy/rayservice.yaml. IDs, context lengths, and image
// support must match its llm_configs.
type modelSpec struct {
	ID string
	// Admission reserves the complete engine context (vLLM max_model_len) for every call.
	ContextTokens int64
	// Scales the caller's token_price_microusd, so calls to stronger models cost more.
	PriceMultiplier int64
	Vision          bool
	// Forwarded as reasoning_effort to gpt-oss models; callers cannot set it.
	ReasoningEffort string
}

// Callers that send model "auto" let the proxy pick one of these models for each request.
const (
	autoModel   = "auto"
	cheapModel  = "gpt-oss-20b"
	strongModel = "gpt-oss-120b"
	visionModel = "qwen3.5-4b"
)

// Matches max_model_len of every model in deploy/rayservice.yaml.
const contextTokens int64 = 8192

// Reasoning tokens count against max_tokens, so gpt-oss needs a larger output limit.
const maxOutputTokens = 4096

var models = map[string]modelSpec{
	cheapModel:  {ID: cheapModel, ContextTokens: contextTokens, PriceMultiplier: 1, ReasoningEffort: "low"},
	strongModel: {ID: strongModel, ContextTokens: contextTokens, PriceMultiplier: 3, ReasoningEffort: "medium"},
	visionModel: {ID: visionModel, ContextTokens: contextTokens, PriceMultiplier: 1, Vision: true},
}

// modelList is the /v1/models order; clients that pick the first entry get routing.
var modelList = []string{autoModel, cheapModel, strongModel, visionModel}

func pattern(alternatives ...string) *regexp.Regexp {
	return regexp.MustCompile(`(?im)` + strings.Join(alternatives, "|"))
}

// These are deliberately simple keyword and syntax checks, used when the model router
// (router/app.py) is not configured or fails. A miss sends a coding or math question to the
// cheap model; a false match only costs the stronger model's price.
var (
	codePattern = pattern(
		"```", "~~~",
		`\b(def|func|fn)\s+\w+\s*\(`, `\bfunction\s*\w*\s*\(`, `\bclass\s+\w+\s*[:({]`, `#include\s*<`,
		`^\s*import\s+(\w+(\.\w+)*(\s+as\s+\w+)?\s*$|[\w.*]+;|\(|"|.+\s+from\s+['"])`,
		`^\s*from\s+[\w.]+\s+import\s+\w`,
		`\b(console\.log|System\.out\.print\w*|printf|println)\s*\(`,
		`\bselect\b.+\bfrom\b.+\b(where|join|group\s+by|order\s+by)\b`,
		`=>|::|&&|\|\||!=|==`, `\{[^{}]*;[^{}]*\}`, `</?[a-z][a-z0-9]*(\s[^<>]*)?>`,
		`traceback \(most recent call last\)|exception in thread|\bpanic:|\bgoroutine \d+|segmentation fault|npm err!|\w+(error|exception):`,
		`\b(write|writing|fix|review|explain|debug|optimi[sz]e|refactor|convert|generate|complete)\b.{0,40}\bcode\b`,
		`\b(write|writing|implement|create|build|fix|debug|refactor|optimi[sz]e|review|test)\b.{0,40}\b(functions?|methods?|class(es)?|scripts?|programs?|query|queries|regex|endpoints?|api|cli|unit tests?|algorithms?|modules?|library)\b`,
		`\b(go|rust|ruby|swift|dart|elixir|lua|zig|julia|r)\s+(code|functions?|programs?|scripts?|structs?|modules?|packages?|snippets?|class(es)?|methods?|library|app|services?)\b`,
		`\b\w+exception\b`,
		`\b(this|my|the|following|your|sample|example|source)\s+code\b`, `\bcode\s+(that|which|snippet|block|review)\b`,
		`\b(coding|programming|compile|compiler|debug|debugging|refactor|regex|regexp|algorithms?|unit tests?|stack trace|syntax error|runtime error|null pointer|recursion|linked list|binary tree|hash ?map|big-?o|leetcode|api endpoint|shell script|bash script)\b`,
		`\b(python|javascript|typescript|java|golang|kotlin|php|scala|haskell|perl|sql|html|css|json|yaml|dockerfile|terraform|kubectl|numpy|pytorch|tensorflow)\b`,
		`\bc\+\+|\bc#|\bnode\.js\b|(^|\s)\.net\b`,
	)
	mathPattern = pattern(
		`[∫∑∏√∞≤≥≠≈π∂∇∈∀∃±×÷²³]`,
		`\\(frac|int|sum|sqrt|lim|prod|cdot|times|infty|alpha|beta|theta|pi|begin)`, `\$\$`,
		`\d\s*[+*^×÷]\s*\(?\d`, `\d\s*/\s*\d+\s*[-+*=]`, `[a-z0-9)]\s*\^\s*[-a-z0-9(]`,
		`\b\d*[a-z]\s*[-+*/]\s*\d+\s*=\s*-?\d`, `\b(sin|cos|tan|ln|log)\s*\(`, `\bf\s*\(\s*x\s*\)`, `\bd[xy]\s*/\s*d[xy]\b`,
		`\d\s*%\s*of\b`, `\b\d+\s+(times|plus|minus|divided by|multiplied by)\s+\d+`,
		`\b(math|maths|mathematics|arithmetic|solve|equations?|integral|integrate|derivative|differentiate|calculus|algebra|geometry|trigonometry|theorem|lemma|prove|proof that|probability|statistics|variance|standard deviation|matrix|matrices|eigen\w*|polynomial|logarithm|factorial|prime numbers?|modulo|combinatorics|permutations?|square root|cube root|calculate|calculation)\b`,
	)
)

func contentText(c content) string {
	if c.parts == nil {
		return c.text
	}
	texts := make([]string, 0, len(c.parts))
	for _, p := range c.parts {
		texts = append(texts, p.Text)
	}
	return strings.Join(texts, "\n")
}

func hasImages(messages []message) bool {
	for _, m := range messages {
		for _, p := range m.Content.parts {
			if p.Type == "image_url" {
				return true
			}
		}
	}
	return false
}

// newestUserHasImages reports whether the image belongs to the question being asked now,
// rather than to an earlier turn that the caller resends as history.
func newestUserHasImages(messages []message) bool {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == "user" {
			return hasImages(messages[i : i+1])
		}
	}
	return false
}

const imageOmitted = "[image omitted: this model cannot read images]"

// withoutImages replaces every image with a short note, for models that cannot read images.
// It copies what it changes, because the audit log records the caller's own messages.
func withoutImages(messages []message) []message {
	if !hasImages(messages) {
		return messages
	}
	out := make([]message, len(messages))
	for i, m := range messages {
		out[i] = m
		if !hasImages(messages[i : i+1]) {
			continue
		}
		texts := make([]string, len(m.Content.parts))
		for j, p := range m.Content.parts {
			texts[j] = p.Text
			if p.Type == "image_url" {
				texts[j] = imageOmitted
			}
		}
		out[i].Content = content{text: strings.Join(texts, "\n")}
	}
	return out
}

// routingTexts returns the newest user message and the reply before it, if any, so follow-ups
// to a coding or math answer can stay on the strong model.
func routingTexts(messages []message) (user, previous string) {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != "user" {
			continue
		}
		if i > 0 && messages[i-1].Role == "assistant" {
			previous = contentText(messages[i-1].Content)
		}
		return contentText(messages[i].Content), previous
	}
	return "", ""
}

// routeByRules picks a model for an "auto" request with keyword and syntax checks.
func routeByRules(messages []message) (modelSpec, string) {
	user, previous := routingTexts(messages)
	text := user + "\n" + previous
	label := "simple"
	switch {
	case codePattern.MatchString(text):
		label = "code"
	case mathPattern.MatchString(text):
		label = "math"
	}
	return routeLabel(messages, label)
}

// routeLabel selects the model for a text label. An image in the newest user message needs the
// only vision model. An earlier image keeps a simple follow-up, such as "What color is it?", on
// the vision model, but code and math still go to the strong model, which gets the earlier
// images as a note (see withoutImages) instead of failing on them.
func routeLabel(messages []message, label string) (modelSpec, string) {
	if newestUserHasImages(messages) || (label == "simple" && hasImages(messages)) {
		return models[visionModel], "image"
	}
	return models[routerModels[label]], label
}

// The router's labels and the models they select.
var routerModels = map[string]string{"simple": cheapModel, "code": strongModel, "math": strongModel}

const (
	routerTimeout = 500 * time.Millisecond
	// The router's embedding model reads at most 512 tokens, so longer text changes nothing.
	maxRouterChars = 2000
)

// promptRouter asks the model-router service (router/app.py) to classify a prompt.
type promptRouter struct {
	url    string
	client *http.Client
}

func newPromptRouter(url string) *promptRouter {
	return &promptRouter{url: url, client: &http.Client{Timeout: routerTimeout}}
}

func truncateRunes(s string, n int) string {
	for i := range s {
		if n == 0 {
			return s[:i]
		}
		n--
	}
	return s
}

func (p *promptRouter) classify(ctx context.Context, text, previous string) (string, error) {
	body, err := json.Marshal(map[string]string{
		"text": truncateRunes(text, maxRouterChars), "context": truncateRunes(previous, maxRouterChars),
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := p.client.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return "", fmt.Errorf("model router returned HTTP %d", res.StatusCode)
	}
	var out struct {
		Label string `json:"label"`
	}
	if err := json.NewDecoder(io.LimitReader(res.Body, 4096)).Decode(&out); err != nil {
		return "", err
	}
	if _, ok := routerModels[out.Label]; !ok {
		return "", fmt.Errorf("model router returned unknown label %q", out.Label)
	}
	return out.Label, nil
}

// route picks a model for an "auto" request and returns why and what decided. An image in the
// newest user message needs the only vision model. Otherwise the model router classifies the
// text; if it is not configured or fails, the keyword rules decide. Earlier images only matter
// for simple text; see routeLabel.
func (s *service) route(ctx context.Context, messages []message) (modelSpec, string, string) {
	if s.router != nil && !newestUserHasImages(messages) {
		user, previous := routingTexts(messages)
		label, err := s.router.classify(ctx, user, previous)
		if err == nil {
			m, reason := routeLabel(messages, label)
			return m, reason, "model"
		}
		log.Printf("model router failed, using keyword rules: %v", err)
	}
	m, reason := routeByRules(messages)
	return m, reason, "rules"
}
