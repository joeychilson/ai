package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
)

const (
	defaultBaseURL = "https://api.openai.com/v1"
)

// Client is a client for the OpenAI API.
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

// SpeechModel represents the Speech model to use for the request.
type SpeechModel string

const (
	ModelTTS1    SpeechModel = "tts-1"
	ModelTTS1_HD SpeechModel = "tts-1-hd"
)

// SpeechVoice represents the Speech voice to use for the request.
type SpeechVoice string

const (
	VoiceAlloy   SpeechVoice = "alloy"
	VoiceEcho    SpeechVoice = "echo"
	VoiceFable   SpeechVoice = "fable"
	VoiceNova    SpeechVoice = "nova"
	VoiceOnyx    SpeechVoice = "onyx"
	VoiceShimmer SpeechVoice = "shimmer"
)

// SpeechFormat represents the Speech format to use for the request.
type SpeechFormat string

const (
	FormatAAC  SpeechFormat = "aac"
	FormatFLAC SpeechFormat = "flac"
	FormatMP3  SpeechFormat = "mp3"
	FormatOpus SpeechFormat = "opus"
	FormatPCM  SpeechFormat = "pcm"
	FormatWAV  SpeechFormat = "wav"
)

// SpeechResponse describes a speech response.
type SpeechResponse struct {
	Format  SpeechFormat `json:"format"`
	Content []byte       `json:"content"`
}

// CreateSpeachRequest describes a speech request.
type CreateSpeachRequest struct {
	Model          SpeechModel  `json:"model"`
	Input          string       `json:"input"`
	Voice          SpeechVoice  `json:"voice"`
	ResponseFormat SpeechFormat `json:"response_format,omitempty"`
	Speed          float32      `json:"speed,omitempty"`
}

// CreateSpeech performs a speech request and returns the file type and content.
func (c *Client) CreateSpeech(ctx context.Context, req *CreateSpeachRequest) (*SpeechResponse, error) {
	url := fmt.Sprintf("%s/audio/speech", c.baseURL)

	resp, err := c.requestJSON(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	audioData, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %v", err)
	}

	return &SpeechResponse{
		Format:  req.ResponseFormat,
		Content: audioData,
	}, nil
}

// TranscriptionModel represents the Transcript model to use for the request.
type TranscriptionModel string

const (
	ModelWhisper1 TranscriptionModel = "whisper-1"
)

// TranscriptFormat represents the Transcript format to use for the request.
type TranscriptFormat string

const (
	FormatJSON TranscriptFormat = "json"
	FormatText TranscriptFormat = "text"
)

// TranscriptionResponse describes a transcription response.
type TranscriptionResponse struct {
	Task     string  `json:"task"`
	Language string  `json:"language"`
	Duration float32 `json:"duration"`
	Text     string  `json:"text"`
	Words    []struct {
		Word  string  `json:"word"`
		Start float32 `json:"start"`
		End   float32 `json:"end"`
	} `json:"words"`
	Segments []struct {
		ID               int     `json:"id"`
		Seek             int     `json:"seek"`
		Start            float32 `json:"start"`
		End              float32 `json:"end"`
		Text             string  `json:"text"`
		Tokens           []int   `json:"tokens"`
		Temperature      float32 `json:"temperature"`
		AvgLogProb       float32 `json:"avg_log_prob"`
		CompressionRatio float32 `json:"compression_ratio"`
		NoSpeechProb     float32 `json:"no_speech_prob"`
	} `json:"segments"`
}

// CreateTranscribeRequest describes a transcription request.
type CreateTranscriptionRequest struct {
	File                   string             `json:"file"`
	Model                  TranscriptionModel `json:"model"`
	Language               string             `json:"language,omitempty"`
	Prompt                 string             `json:"prompt,omitempty"`
	ResponseFormat         TranscriptFormat   `json:"response_format,omitempty"`
	Temperature            float32            `json:"temperature,omitempty"`
	TimestampGranularities []string           `json:"timestamp_granularities,omitempty"`
}

// AddFields adds fields to the multipart form data.
func (req *CreateTranscriptionRequest) AddFields(writer *multipart.Writer) error {
	file, err := os.Open(req.File)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	part, err := writer.CreateFormFile("file", filepath.Base(req.File))
	if err != nil {
		return fmt.Errorf("failed to create form file: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return fmt.Errorf("failed to copy file: %v", err)
	}

	if req.Model != "" {
		_ = writer.WriteField("model", string(req.Model))
	}
	if req.Language != "" {
		_ = writer.WriteField("language", req.Language)
	}
	if req.Prompt != "" {
		_ = writer.WriteField("prompt", req.Prompt)
	}
	if req.ResponseFormat != "" {
		_ = writer.WriteField("response_format", string(req.ResponseFormat))
	}
	if req.Temperature != 0 {
		_ = writer.WriteField("temperature", fmt.Sprintf("%f", req.Temperature))
	}
	for _, granularity := range req.TimestampGranularities {
		_ = writer.WriteField("timestamp_granularities[]", granularity)
	}
	return nil
}

// CreateTranscription performs a transcription request and returns the transcript.
func (c *Client) CreateTranscription(ctx context.Context, req *CreateTranscriptionRequest) (*TranscriptionResponse, error) {
	url := fmt.Sprintf("%s/audio/transcriptions", c.baseURL)

	resp, err := c.requestMultipartFormData(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var transcriptionResp TranscriptionResponse
	if err := json.NewDecoder(resp.Body).Decode(&transcriptionResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &transcriptionResp, nil
}

// TranslationResponse describes a translation response.
type TranslationResponse struct {
	Text string `json:"text"`
}

// CreateTranslationRequest describes a translation request.
type CreateTranslationRequest struct {
	File           string             `json:"file"`
	Model          TranscriptionModel `json:"model"`
	Prompt         string             `json:"prompt,omitempty"`
	ResponseFormat TranscriptFormat   `json:"response_format,omitempty"`
	Temperature    float32            `json:"temperature,omitempty"`
}

// AddFields adds fields to the multipart form data.
func (req *CreateTranslationRequest) AddFields(writer *multipart.Writer) error {
	file, err := os.Open(req.File)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	part, err := writer.CreateFormFile("file", filepath.Base(req.File))
	if err != nil {
		return fmt.Errorf("failed to create form file: %v", err)
	}

	_, err = io.Copy(part, file)
	if err != nil {
		return fmt.Errorf("failed to copy file: %v", err)
	}

	if req.Model != "" {
		_ = writer.WriteField("model", string(req.Model))
	}
	if req.Prompt != "" {
		_ = writer.WriteField("prompt", req.Prompt)
	}
	if req.ResponseFormat != "" {
		_ = writer.WriteField("response_format", string(req.ResponseFormat))
	}
	if req.Temperature != 0 {
		_ = writer.WriteField("temperature", fmt.Sprintf("%f", req.Temperature))
	}
	return nil
}

// CreateTranslation performs a translation request and returns the translated text.
func (c *Client) CreateTranslation(ctx context.Context, req *CreateTranslationRequest) (*TranslationResponse, error) {
	url := fmt.Sprintf("%s/audio/translations", c.baseURL)

	resp, err := c.requestMultipartFormData(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var translationResp TranslationResponse
	if err := json.NewDecoder(resp.Body).Decode(&translationResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &translationResp, nil
}

// LanguageModel represents the Language model to use for the request.
type LanguageModel string

const (
	ModelGPT3Dot5_Turbo           LanguageModel = "gpt-3.5-turbo"
	ModelGPT3Dot5_Turbo_16k       LanguageModel = "gpt-3.5-turbo-16k"
	ModelGPT3Dot5_Turbo_16k_0613  LanguageModel = "gpt-3.5-turbo-16k-0613"
	ModelGPT3Dot5_Turbo_0125      LanguageModel = "gpt-3.5-turbo-0125"
	ModelGPT3Dot5_Turbo_0613      LanguageModel = "gpt-3.5-turbo-0613"
	ModelGPT3Dot5_Turbo_1106      LanguageModel = "gpt-3.5-turbo-1106"
	ModelGPT3Dot5_Turbo_Instruct  LanguageModel = "gpt-3.5-turbo-instruct"
	ModelGPT4                     LanguageModel = "gpt-4"
	ModelGPT4_0613                LanguageModel = "gpt-4-0613"
	ModelGPT4_32k                 LanguageModel = "gpt-4-32k"
	ModelGPT4_32k_0613            LanguageModel = "gpt-4-32k-0613"
	ModelGPT4_0125_Preview        LanguageModel = "gpt-4-0125-preview"
	ModelGPT4_1106_Preview        LanguageModel = "gpt-4-1106-preview"
	ModelGPT4_Turbo               LanguageModel = "gpt-4-turbo"
	ModelGPT4_Turbo_Preview       LanguageModel = "gpt-4-turbo-preview"
	ModelGPT4_Vision_Preview      LanguageModel = "gpt-4-vision-preview"
	ModelGPT4_1106_Vision_Preview LanguageModel = "gpt-4-vision-preview-0613"
)

// Role represents the role of the user in the chat.
type Role string

const (
	RoleUser      Role = "user"
	RoleSystem    Role = "system"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// Message represents a message in the chat.
type Message interface {
	Role() Role
}

// SystemMessage represents a system message in the chat.
type SystemMessage struct {
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// Role returns the system role for the system message.
func (m SystemMessage) Role() Role {
	return RoleSystem
}

// MarshalJSON marshals the system message to JSON.
func (m SystemMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    Role   `json:"role"`
		Content string `json:"content"`
		Name    string `json:"name,omitempty"`
	}{
		Role:    m.Role(),
		Content: m.Content,
		Name:    m.Name,
	})
}

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

// ImageContent represents an image content in the chat.
type ImageContent struct {
	URL string `json:"url"`
}

// Type returns the type of the image content.
func (c ImageContent) Type() string {
	return "image_url"
}

// MarshalJSON marshals the image content to JSON.
func (c ImageContent) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Type     string `json:"type"`
		ImageURL struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}{
		Type: c.Type(),
		ImageURL: struct {
			URL string `json:"url"`
		}{URL: c.URL},
	})
}

// UserMessage represents a user message in the chat.
type UserMessage struct {
	Content []Content `json:"content"`
	Name    string    `json:"name,omitempty"`
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
		Name    string    `json:"name,omitempty"`
	}{
		Role:    m.Role(),
		Content: m.Content,
		Name:    m.Name,
	})
}

// AssistantMessage represents an assistant message in the chat.
type AssistantMessage struct {
	Content   string     `json:"content,omitempty"`
	Name      string     `json:"name,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// Role returns the assistant role for the assistant message.
func (m AssistantMessage) Role() Role {
	return RoleAssistant
}

// MarshalJSON marshals the assistant message to JSON.
func (m AssistantMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role      Role       `json:"role"`
		Content   string     `json:"content,omitempty"`
		Name      string     `json:"name,omitempty"`
		ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	}{
		Role:      m.Role(),
		Content:   m.Content,
		Name:      m.Name,
		ToolCalls: m.ToolCalls,
	})
}

// ToolMessage represents a tool message in the chat.
type ToolMessage struct {
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id"`
}

// Role returns the tool role for the tool message.
func (m ToolMessage) Role() Role {
	return RoleTool
}

// MarshalJSON marshals the tool message to JSON.
func (m ToolMessage) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role       Role   `json:"role"`
		Content    string `json:"content"`
		ToolCallID string `json:"tool_call_id"`
	}{
		Role:       m.Role(),
		Content:    m.Content,
		ToolCallID: m.ToolCallID,
	})
}

// Tool represents a tool in the chat.
type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// ToolCall represents a tool call in the chat.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// ToolChoice represents a tool choice in the chat.
type ToolChoice struct {
	Type     string `json:"type"`
	Function struct {
		Name string `json:"name"`
	} `json:"function"`
}

// LogProb describes a log probability.
type LogProb struct {
	Token       string  `json:"token"`
	LogProb     float32 `json:"logprob"`
	Bytes       []byte  `json:"bytes"`
	TopLogProbs []struct {
		Token   string  `json:"token"`
		LogProb float32 `json:"logprob"`
		Bytes   []byte  `json:"bytes"`
	} `json:"top_logprobs"`
}

// ChatResponse describes a chat completion response.
type ChatResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		FinishReason string `json:"finish_reason"`
		Index        int    `json:"index"`
		Message      struct {
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Role      string     `json:"role"`
		} `json:"message"`
		LogProbs []LogProb `json:"logprobs"`
	} `json:"choices"`
	Created           int    `json:"created"`
	Model             string `json:"model"`
	SystemFingerprint string `json:"system_fingerprint"`
	Object            string `json:"object"`
	Usage             struct {
		CompletionTokens int `json:"completion_tokens"`
		PromptTokens     int `json:"prompt_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ChatRequest describes a chat completion request.
type ChatRequest struct {
	Messages         []Message     `json:"messages"`
	Model            LanguageModel `json:"model"`
	FrequencyPenalty float32       `json:"frequency_penalty,omitempty"`
	LogitBias        float32       `json:"logit_bias,omitempty"`
	LogProbs         bool          `json:"logprobs,omitempty"`
	TopLogProbs      int           `json:"top_logprobs,omitempty"`
	MaxTokens        int           `json:"max_tokens,omitempty"`
	N                int           `json:"n,omitempty"`
	PresencePenalty  float32       `json:"presence_penalty,omitempty"`
	ResponseFormat   string        `json:"response_format,omitempty"` // TODO
	Seed             int           `json:"seed,omitempty"`
	Stop             []string      `json:"stop,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
	Temperature      float32       `json:"temperature,omitempty"`
	TopP             float32       `json:"top_p,omitempty"`
	Tools            []Tool        `json:"tools,omitempty"`
	ToolChoices      []ToolChoice  `json:"tool_choices,omitempty"`
	User             string        `json:"user,omitempty"`
}

// Chat performs a chat completion request and returns the completion.
func (c *Client) Chat(ctx context.Context, req *ChatRequest) (*ChatResponse, error) {
	req.Stream = false

	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	resp, err := c.requestJSON(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var chatCompletionResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatCompletionResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &chatCompletionResp, nil
}

// ChatChunk represents a chat chunk in the stream chat completion.
type ChatChunk struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// Choice represents a chat completion chunk choice in the stream chat completion.
type Choice struct {
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason"`
	Index        int    `json:"index"`
}

// Delta represents a streaming delta in the stream chat completion.
type Delta struct {
	Content string `json:"content"`
	Role    string `json:"role"`
}

// StreamCallback is a callback function for streaming chat completion.
type StreamCallback func(ctx context.Context, chunk *ChatChunk)

// ChatStream performs a chat completion request and streams the completion to the callback.
func (c *Client) ChatStream(ctx context.Context, req *ChatRequest, callback StreamCallback) error {
	req.Stream = true

	url := fmt.Sprintf("%s/chat/completions", c.baseURL)

	resp, err := c.requestJSON(ctx, url, req)
	if err != nil {
		return fmt.Errorf("failed to perform request: %v", err)
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
			return fmt.Errorf("failed to read response: %v", err)
		}

		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}

		line = bytes.TrimPrefix(line, []byte("data: "))

		var event struct {
			ID      string   `json:"id"`
			Object  string   `json:"object"`
			Created int64    `json:"created"`
			Model   string   `json:"model"`
			Choices []Choice `json:"choices"`
		}
		if err := json.Unmarshal(line, &event); err != nil {
			return fmt.Errorf("failed to unmarshal event: %v", err)
		}

		for _, choice := range event.Choices {
			chunk := &ChatChunk{
				ID:      event.ID,
				Object:  event.Object,
				Created: event.Created,
				Model:   event.Model,
				Choices: []Choice{
					{
						Delta:        choice.Delta,
						FinishReason: choice.FinishReason,
						Index:        choice.Index,
					},
				},
			}
			callback(ctx, chunk)

			if choice.FinishReason != "" {
				return nil
			}
		}
	}
	return nil
}

// EmbeddingModel represents the Embedding model to use for the request.
type EmbeddingModel string

const (
	ModelTextEmbeddingADA_002 EmbeddingModel = "text-embedding-ada-002"
	ModelTextEmbedding3_Small EmbeddingModel = "text-embedding-3-small"
	ModelTextEmbedding3_Large EmbeddingModel = "text-embedding-3-large"
)

// EncodingFormat represents the encoding format to use for the request.
type EncodingFormat string

const (
	FormatFloat  EncodingFormat = "float"
	FormatBase64 EncodingFormat = "base64"
)

// Embedding describes an embedding.
type Embedding struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// EmbedResponse describes an embedding response.
type EmbedResponse struct {
	Object string      `json:"object"`
	Data   []Embedding `json:"data"`
	Model  string      `json:"model"`
	Usage  struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// EmbedRequest describes an embedding request.
type EmbedRequest struct {
	Input          []string       `json:"input"`
	Model          EmbeddingModel `json:"model"`
	EncodingFormat EncodingFormat `json:"encoding_format,omitempty"`
	Dimensions     int            `json:"dimensions,omitempty"`
	User           string         `json:"user,omitempty"`
}

// Embed performs an embedding request and returns the embeddings.
func (c *Client) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	url := fmt.Sprintf("%s/embeddings", c.baseURL)

	resp, err := c.requestJSON(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var embeddingResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &embeddingResp, nil
}

// ImageModel represents the Image model to use for the request.
type ImageModel string

const (
	ModelDallE2 ImageModel = "dall-e-2"
	ModelDallE3 ImageModel = "dall-e-3"
)

// ImageFormat represents the Image format to use for the request.
type ImageFormat string

const (
	FormatURL        ImageFormat = "url"
	FormatBase64JSON ImageFormat = "b64_json"
)

// ImageSize represents the Image size to use for the request.
type ImageSize string

const (
	Size256x256   ImageSize = "256x256"
	Size512x512   ImageSize = "512x512"
	Size1024x1024 ImageSize = "1024x1024"
	Size1792x1024 ImageSize = "1792x1024"
	Size1024x1792 ImageSize = "1024x1792"
)

// ImageResponse describes an image response.
type ImageResponse struct {
	Created int `json:"created"`
	Data    []struct {
		B64JSON       string `json:"b64_json"`
		URL           string `json:"url"`
		RevisedPrompt string `json:"revised_prompt"`
	} `json:"data"`
}

// CreateImageRequest describes an image request.
type CreateImageRequest struct {
	Prompt         string      `json:"prompt"`
	Model          ImageModel  `json:"model,omitempty"`
	N              int         `json:"n,omitempty"`
	Quality        string      `json:"quality,omitempty"`
	ResponseFormat ImageFormat `json:"response_format,omitempty"`
	Size           ImageSize   `json:"size,omitempty"`
	Style          string      `json:"style,omitempty"`
	User           string      `json:"user,omitempty"`
}

// CreateImage performs an image request and returns the images.
func (c *Client) CreateImage(ctx context.Context, req *CreateImageRequest) (*ImageResponse, error) {
	url := fmt.Sprintf("%s/images/generations", c.baseURL)

	resp, err := c.requestJSON(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var imageResp ImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&imageResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &imageResp, nil
}

// EditImageRequest describes an image editing request.
type EditImageRequest struct {
	Image          string      `json:"image"`
	Prompt         string      `json:"prompt"`
	Mask           string      `json:"mask,omitempty"`
	Model          ImageModel  `json:"model,omitempty"`
	N              int         `json:"n,omitempty"`
	Size           ImageSize   `json:"size,omitempty"`
	ResponseFormat ImageFormat `json:"response_format,omitempty"`
	User           string      `json:"user,omitempty"`
}

// AddFields adds fields to the multipart form data.
func (req *EditImageRequest) AddFields(writer *multipart.Writer) error {
	image, err := os.Open(req.Image)
	if err != nil {
		return fmt.Errorf("failed to open file: %v", err)
	}
	imagePart, err := writer.CreateFormFile("image", filepath.Base(req.Image))
	if err != nil {
		return fmt.Errorf("failed to create form file: %v", err)
	}
	_, err = io.Copy(imagePart, image)
	if err != nil {
		return fmt.Errorf("failed to copy file: %v", err)
	}

	if req.Mask != "" {
		mask, err := os.Open(req.Mask)
		if err != nil {
			return fmt.Errorf("failed to open mask: %v", err)
		}

		maskPart, err := writer.CreateFormFile("mask", filepath.Base(req.Mask))
		if err != nil {
			return fmt.Errorf("failed to create form file: %v", err)
		}

		_, err = io.Copy(maskPart, mask)
		if err != nil {
			return fmt.Errorf("failed to copy mask: %v", err)
		}
	}

	if req.Prompt != "" {
		_ = writer.WriteField("prompt", req.Prompt)
	}
	if req.Model != "" {
		_ = writer.WriteField("model", string(req.Model))
	}
	if req.N != 0 {
		_ = writer.WriteField("n", fmt.Sprintf("%d", req.N))
	}
	if req.Size != "" {
		_ = writer.WriteField("size", string(req.Size))
	}
	if req.ResponseFormat != "" {
		_ = writer.WriteField("response_format", string(req.ResponseFormat))
	}
	if req.User != "" {
		_ = writer.WriteField("user", req.User)
	}
	return nil
}

// EditImage performs an image editing request and returns the edited images.
func (c *Client) EditImage(ctx context.Context, req *EditImageRequest) (*ImageResponse, error) {
	url := fmt.Sprintf("%s/images/edits", c.baseURL)

	resp, err := c.requestMultipartFormData(ctx, url, req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var imageResp ImageResponse
	if err := json.NewDecoder(resp.Body).Decode(&imageResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &imageResp, nil
}

// Model represents a model.
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int    `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ListModelsResponse describes a model list response.
type ListModelsResponse struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}

// ListModels performs a model list request and returns the models.
func (c *Client) ListModels(ctx context.Context) (*ListModelsResponse, error) {
	url := fmt.Sprintf("%s/models", c.baseURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var modelListResp ListModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&modelListResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &modelListResp, nil
}

// RetrieveModel performs a model retrieve request and returns the model.
func (c *Client) RetrieveModel(ctx context.Context, id string) (*Model, error) {
	url := fmt.Sprintf("%s/models/%s", c.baseURL, id)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to perform request: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var model Model
	if err := json.NewDecoder(resp.Body).Decode(&model); err != nil {
		return nil, fmt.Errorf("failed to decode response: %v", err)
	}
	return &model, nil
}

func (c *Client) requestJSON(ctx context.Context, url string, req any) (*http.Response, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+c.token)

	return c.httpClient.Do(httpReq)
}

// MultipartFormDataRequest is an interface for requests that require multipart form data.
type MultipartFormDataRequest interface {
	AddFields(writer *multipart.Writer) error
}

func (c *Client) requestMultipartFormData(ctx context.Context, url string, req MultipartFormDataRequest) (*http.Response, error) {
	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)

	err := req.AddFields(writer)
	if err != nil {
		return nil, fmt.Errorf("failed to add fields to form data: %v", err)
	}

	err = writer.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to close writer: %v", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, body)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	httpReq.Header.Set("Authorization", "Bearer "+c.token)

	return c.httpClient.Do(httpReq)
}

// ErrorResponse describes an error response.
type ErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Code    string `json:"code"`
		Message string `json:"message"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (c *Client) decodeError(resp *http.Response) error {
	var errResp ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		return fmt.Errorf("error decoding error response: %w", err)
	}
	return fmt.Errorf("%s: %s", errResp.Error.Type, errResp.Error.Message)
}
