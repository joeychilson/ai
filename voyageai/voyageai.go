package voyageai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

const (
	defaultBaseURL = "https://api.voyageai.com/v1"
)

// Client is a client for the VoyageAI API.
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

// Usage is the usage of an embedding.
type Usage struct {
	TotalTokens int `json:"total_tokens"`
}

// EmbeddingModel is a model that can be used for embeddings.
type EmbeddingModel string

const (
	ModelVoyage2      EmbeddingModel = "voyage-2"
	ModelVoyageLarge2 EmbeddingModel = "voyage-large-2"
	ModelVoyageLaw2   EmbeddingModel = "voyage-law-2"
	ModelVoyageCode2  EmbeddingModel = "voyage-code-2"
)

// InputType is the type of input to embed.
type InputType string

const (
	InputTypeQuery    InputType = "query"
	InputTypeDocument InputType = "document"
)

// EncodingFormat is the format to encode the input in.
type EncodingFormat string

const (
	EncodingFormatBase64 EncodingFormat = "base64"
)

// EmbedRequest is a request to embed text.
type EmbedRequest struct {
	Model          EmbeddingModel `json:"model"`
	Input          []string       `json:"input"`
	InputType      InputType      `json:"input_type,omitempty"`
	Truncation     *bool          `json:"truncation,omitempty"`
	EncodingFormat EncodingFormat `json:"encoding_format,omitempty"`
}

// EmbeddingData is the data for an embedding.
type EmbeddingData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbedResponse is a response to an embedding request.
type EmbedResponse struct {
	Object string          `json:"object"`
	Data   []EmbeddingData `json:"data"`
	Model  string          `json:"model"`
	Usage  Usage           `json:"usage"`
}

// Embed sends a request to create embeddings for the given text.
func (c *Client) Embed(ctx context.Context, req *EmbedRequest) (*EmbedResponse, error) {
	resp, err := c.request(ctx, "/embeddings", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var embeddingResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, err
	}

	return &embeddingResp, nil
}

// RerankModel is a model that can be used for reranking.
type RerankModel string

const (
	ModelRerankLite1 RerankModel = "rerank-lite-1"
)

// RerankRequest is a request to rerank documents.
type RerankRequest struct {
	Model           RerankModel `json:"model"`
	Query           string      `json:"query"`
	Documents       []string    `json:"documents"`
	TopK            int         `json:"top_k,omitempty"`
	ReturnDocuments bool        `json:"return_documents,omitempty"`
	Truncation      bool        `json:"truncation,omitempty"`
}

// RerankData is the data for a rerank.
type RerankData struct {
	Index           int     `json:"index"`
	RelenvanceScore float32 `json:"relevance_score"`
	Document        string  `json:"document,omitempty"`
}

// RerankResponse is a response to a rerank request.
type RerankResponse struct {
	Object string       `json:"object"`
	Data   []RerankData `json:"data"`
	Model  string       `json:"model"`
	Usage  Usage        `json:"usage"`
}

// Rerank sends a request to rerank the given documents.
func (c *Client) Rerank(ctx context.Context, req *RerankRequest) (*RerankResponse, error) {
	resp, err := c.request(ctx, "/rerank", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, c.decodeError(resp)
	}

	var rerankResp RerankResponse
	if err := json.NewDecoder(resp.Body).Decode(&rerankResp); err != nil {
		return nil, err
	}

	return &rerankResp, nil
}

func (c *Client) request(ctx context.Context, path string, body any) (*http.Response, error) {
	url := c.baseURL + path

	reqBody, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Authorization", "Bearer "+c.token)
	httpReq.Header.Set("Content-Type", "application/json")

	return c.httpClient.Do(httpReq)
}

// ErrorResponse is an error response from the API.
type ErrorResponse struct {
	Detail string `json:"detail"`
}

func (c *Client) decodeError(resp *http.Response) error {
	var errResp ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		return fmt.Errorf("error decoding error response: %w", err)
	}
	return fmt.Errorf("%s: %s", resp.Status, errResp.Detail)
}
