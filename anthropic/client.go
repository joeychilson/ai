package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
)

const (
	defaultBaseURL = "https://api.anthropic.com/v1"
)

// Client is a client for the Anthropic API.
type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// New creates a new Client using the given token.
func New(token string) *Client {
	return &Client{
		baseURL:    defaultBaseURL,
		token:      token,
		httpClient: http.DefaultClient,
	}
}

// LanguageModel represents the Anthropic language model.
type LanguageModel string

const (
	ModelClaude3_Opus   LanguageModel = "claude-3-opus-20240229"
	ModelClaude3_Sonnet LanguageModel = "claude-3-sonnet-20240229"
	ModelClaude3_Haiku  LanguageModel = "claude-3-haiku-20240307"
	ModelClaude2Dot1    LanguageModel = "claude-2.1"
	ModelClaude2        LanguageModel = "claude-2"
	ModelClaude1Dot3    LanguageModel = "claude-1.3"
)

// Role represents conversational roles.
type Role string

const (
	// RoleAssistant is the assistant conversational role.
	RoleAssistant Role = "assistant"
	// RoleUser is the user conversational role.
	RoleUser Role = "user"
)

// Content represents the content of a message.
type Content interface {
	Type() string
}

// TextContent represents a text content in the chat.
type TextContent struct {
	Text string `json:"text"`
}

// Type returns the type of the text content.
func (c TextContent) Type() string {
	return "text"
}

// MarshalJSON marshals the text content to JSON.
func (c TextContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}{
		Type: c.Type(),
		Text: c.Text,
	})
}

// ImageSource represents an image source in the chat.
type ImageSource struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

// ImageContent represents an image content in the chat.
type ImageContent struct {
	Source ImageSource `json:"source"`
}

// Type returns the type of the image content.
func (c ImageContent) Type() string {
	return "image"
}

// MarshalJSON marshals the image content to JSON.
func (c ImageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type   string      `json:"type"`
		Source ImageSource `json:"source"`
	}{
		Type:   c.Type(),
		Source: c.Source,
	})
}

// Message represents a message in the chat.
type Message interface {
	Role() Role
}

// AssistantMessage represents an assistant message in the chat.
type AssistantMessage struct {
	Content []Content `json:"content"`
}

// Role returns the assistant role for the assistant message.
func (m AssistantMessage) Role() Role {
	return RoleAssistant
}

// MarshalJSON marshals the assistant message to JSON.
func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    Role      `json:"role"`
		Content []Content `json:"content"`
	}{
		Role:    m.Role(),
		Content: m.Content,
	})
}

// UserMessage represents a user message in the chat.
type UserMessage struct {
	Content []Content `json:"content"`
}

// Role returns the user role for the user message.
func (m UserMessage) Role() Role {
	return RoleUser
}

// MarshalJSON marshals the user message to JSON.
func (m UserMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    Role      `json:"role"`
		Content []Content `json:"content"`
	}{
		Role:    m.Role(),
		Content: m.Content,
	})
}

// Metadata describes metadata about the request.
type Metadata struct {
	UserID string `json:"user_id"`
}

// ChatRequest describes a request to the messages API.
type ChatRequest struct {
	Model         LanguageModel `json:"model"`
	Messages      []Message     `json:"messages"`
	System        string        `json:"system,omitempty"`
	MaxTokens     int           `json:"max_tokens"`
	Metadata      Metadata      `json:"metadata,omitempty"`
	StopSequences []string      `json:"stop_sequences,omitempty"`
	Stream        bool          `json:"stream,omitempty"`
	Temperature   float32       `json:"temperature,omitempty"`
	TopP          float32       `json:"top_p,omitempty"`
	TopK          int           `json:"top_k,omitempty"`
}

// Usage describes the usage billing and limits usage.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// ChatMessage describes a response from the messages API.
type ChatMessage struct {
	ID           string        `json:"id"`
	Type         string        `json:"type"`
	Role         Role          `json:"role"`
	Content      []TextContent `json:"content"`
	Model        LanguageModel `json:"model"`
	StopReason   string        `json:"stop_reason"`
	StopSequence string        `json:"stop_sequence"`
	Usage        Usage         `json:"usage"`
}

// ErrorResponse describes an error response.
type ErrorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Chat creates a new chat with the given messages and returns the response.
func (c *Client) Chat(ctx context.Context, req *ChatRequest) (*ChatMessage, error) {
	req.Stream = false

	resp, err := c.request(ctx, req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var messageResp ChatMessage
	if err := json.NewDecoder(resp.Body).Decode(&messageResp); err != nil {
		return nil, err
	}
	return &messageResp, nil
}

// Event represents an event received from the SSE stream.
type Event interface {
	EventType() string
}

// PingEvent represents a ping event.
type PingEvent struct {
	Type string `json:"type"`
}

// Type returns the type of the ping event.
func (e PingEvent) EventType() string {
	return "ping"
}

// MessageStartEvent represents the message_start event.
type MessageStartEvent struct {
	Type    string      `json:"type"`
	Message ChatMessage `json:"message"`
}

// EventType returns the type of the message_start event.
func (e MessageStartEvent) EventType() string {
	return "message_start"
}

// ContentBlockStartEvent represents the content_block_start event.
type ContentBlockStartEvent struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content_block"`
}

// EventType returns the type of the content_block_start event.
func (e ContentBlockStartEvent) EventType() string {
	return "content_block_start"
}

// ContentBlockDeltaEvent represents the content_block_delta event.
type ContentBlockDeltaEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"delta"`
}

// EventType returns the type of the content_block_delta event.
func (e ContentBlockDeltaEvent) EventType() string {
	return "content_block_delta"
}

// ContentBlockStopEvent represents the content_block_stop event.
type ContentBlockStopEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

// EventType returns the type of the content_block_stop event.
func (e ContentBlockStopEvent) EventType() string {
	return "content_block_stop"
}

// MessageDeltaEvent represents the message_delta event.
type MessageDeltaEvent struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason   string `json:"stop_reason"`
		StopSequence string `json:"stop_sequence"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// EventType returns the type of the message_delta event.
func (e MessageDeltaEvent) EventType() string {
	return "message_delta"
}

// MessageStopEvent represents the message_stop event.
type MessageStopEvent struct {
	Type string `json:"type"`
}

// EventType returns the type of the message_stop event.
func (e MessageStopEvent) EventType() string {
	return "message_stop"
}

// StreamCallback is a callback function for streaming responses.
type StreamCallback func(ctx context.Context, event Event)

// ChatStream streams the chat with the given messages and calls the callback for each response.
func (c *Client) ChatStream(ctx context.Context, req *ChatRequest, callback StreamCallback) error {
	req.Stream = true

	resp, err := c.request(ctx, req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.decodeError(resp)
	}

	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			return err
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 || bytes.HasPrefix(line, []byte("event:")) {
			continue
		}

		eventData := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data:")))

		var event struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal(eventData, &event); err != nil {
			return err
		}

		switch event.Type {
		case "ping":
			var pingEvent PingEvent
			if err := json.Unmarshal(eventData, &pingEvent); err != nil {
				return err
			}
			callback(ctx, pingEvent)
		case "message_start":
			var messageStartEvent MessageStartEvent
			if err := json.Unmarshal(eventData, &messageStartEvent); err != nil {
				return err
			}
			callback(ctx, messageStartEvent)
		case "content_block_start":
			var contentBlockStartEvent ContentBlockStartEvent
			if err := json.Unmarshal(eventData, &contentBlockStartEvent); err != nil {
				return err
			}
			callback(ctx, contentBlockStartEvent)
		case "content_block_delta":
			var contentBlockDeltaEvent ContentBlockDeltaEvent
			if err := json.Unmarshal(eventData, &contentBlockDeltaEvent); err != nil {
				return err
			}
			callback(ctx, contentBlockDeltaEvent)
		case "content_block_stop":
			var contentBlockStopEvent ContentBlockStopEvent
			if err := json.Unmarshal(eventData, &contentBlockStopEvent); err != nil {
				return err
			}
			callback(ctx, contentBlockStopEvent)
		case "message_delta":
			var messageDeltaEvent MessageDeltaEvent
			if err := json.Unmarshal(eventData, &messageDeltaEvent); err != nil {
				return err
			}
			callback(ctx, messageDeltaEvent)
		case "message_stop":
			var messageStopEvent MessageStopEvent
			if err := json.Unmarshal(eventData, &messageStopEvent); err != nil {
				return err
			}
			callback(ctx, messageStopEvent)
			return nil
		case "error":
			var errResp ErrorResponse
			if err := json.Unmarshal(eventData, &errResp); err != nil {
				return err
			}
			return fmt.Errorf("%s: %s", errResp.Error.Type, errResp.Error.Message)
		default:
			log.Printf("unknown event type: %s", event.Type)
		}
	}
	return nil
}

func (c *Client) request(ctx context.Context, req any) (*http.Response, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	url := c.baseURL + "/messages"

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", c.token)
	httpReq.Header.Set("anthropic-version", "2023-06-01")

	return c.httpClient.Do(httpReq)
}

func (c *Client) decodeError(resp *http.Response) error {
	var errResp ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		return fmt.Errorf("error decoding error response: %w", err)
	}
	return fmt.Errorf("%s: %s", errResp.Error.Type, errResp.Error.Message)
}
