package pinecone

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
)

const (
	defaultBaseURL = "https://api.pinecone.io"
)

// ControlClient is a client for the Pinecone control API.
type ControlClient struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewControlClient creates a new Client for the control API using the given token.
func NewControlClient(token string) *ControlClient {
	return &ControlClient{
		baseURL:    defaultBaseURL,
		token:      token,
		httpClient: http.DefaultClient,
	}
}

// Metric is the distance metric used by the index.
type Metric string

const (
	MetricCosine     Metric = "cosine"
	MetricEuclidean  Metric = "euclidean"
	MetricDotProduct Metric = "dotproduct"
)

// CloudProvider is the cloud provider used by the serverless deployment.
type CloudProvider string

const (
	CloudProviderAWS   CloudProvider = "aws"
	CloudProviderGCP   CloudProvider = "gcp"
	CloudProviderAzure CloudProvider = "azure"
)

// MetadataConfig is the metadata configuration.
type MetadataConfig struct {
	Indexed []string `json:"indexed"`
}

// Pod is the pod configuration.
type Pod struct {
	Environment      string         `json:"environment"`
	Replicas         int            `json:"replicas,omitempty"`
	PodType          string         `json:"pod_type,omitempty"`
	Pods             int            `json:"pods,omitempty"`
	Shards           int            `json:"shards,omitempty"`
	MetadataConfig   MetadataConfig `json:"metadata_config,omitempty"`
	SourceCollection string         `json:"source_collection,omitempty"`
}

// Serverless is the serverless deployment configuration.
type Serverless struct {
	CloudProvider CloudProvider `json:"cloud"`
	Region        string        `json:"region"`
}

// Spec is the index specification.
type Spec struct {
	Pod        *Pod        `json:"pod,omitempty"`
	Serverless *Serverless `json:"serverless,omitempty"`
}

// Status is the index status.
type Status struct {
	Ready bool   `json:"ready"`
	State string `json:"state"`
}

// Index is the index configuration.
type Index struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    Metric `json:"metric"`
	Host      string `json:"host"`
	Spec      Spec   `json:"spec"`
	Status    Status `json:"status"`
}

// ListIndexesResponse is the response for listing indexes.
type ListIndexesResponse struct {
	Indexes []Index `json:"indexes"`
}

// ListIndexes lists all indexes.
func (c *ControlClient) ListIndexes(ctx context.Context) (*ListIndexesResponse, error) {
	resp, err := c.request(ctx, "GET", "/indexes", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var indexes ListIndexesResponse
	err = json.NewDecoder(resp.Body).Decode(&indexes)
	if err != nil {
		return nil, err
	}
	return &indexes, nil
}

// CreateIndexRequest is the request for creating an index.
type CreateIndexRequest struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Spec      Spec   `json:"spec"`
	Metric    Metric `json:"metric,omitempty"`
}

// Validate validates the request.
func (r *CreateIndexRequest) Validate() error {
	if r.Name == "" {
		return fmt.Errorf("name is required")
	}
	if r.Dimension <= 0 {
		return fmt.Errorf("dimension must be greater than 0")
	}
	return nil
}

// CreateIndexResponse is the response for creating an index.
type CreateIndexResponse struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    Metric `json:"metric"`
	Host      string `json:"host"`
	Spec      struct {
		Pod        Pod        `json:"pod"`
		Serverless Serverless `json:"serverless"`
	} `json:"spec"`
	Status Status `json:"status"`
}

// CreateIndex creates a new index with the given configuration.
func (c *ControlClient) CreateIndex(ctx context.Context, req *CreateIndexRequest) (*CreateIndexResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	resp, err := c.request(ctx, "POST", "/indexes", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return nil, decodeError(resp)
	}

	var index CreateIndexResponse
	err = json.NewDecoder(resp.Body).Decode(&index)
	if err != nil {
		return nil, err
	}
	return &index, nil
}

// DecribeIndexResponse is the response for describing an index.
type DecribeIndexResponse struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    Metric `json:"metric"`
	Host      string `json:"host"`
	Spec      struct {
		Pod        Pod        `json:"pod"`
		Serverless Serverless `json:"serverless"`
	} `json:"spec"`
	Status Status `json:"status"`
}

// DescribeIndex describes an index by name.
func (c *ControlClient) DescribeIndex(ctx context.Context, indexName string) (*DecribeIndexResponse, error) {
	resp, err := c.request(ctx, "GET", "/indexes/"+url.PathEscape(indexName), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var index DecribeIndexResponse
	err = json.NewDecoder(resp.Body).Decode(&index)
	if err != nil {
		return nil, err
	}
	return &index, nil
}

// DeleteIndex deletes an index by name.
func (c *ControlClient) DeleteIndex(ctx context.Context, indexName string) error {
	resp, err := c.request(ctx, "DELETE", "/indexes/"+url.PathEscape(indexName), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		return decodeError(resp)
	}
	return nil
}

// ConfigureIndexSpec is the index configuration specification.
type ConfigureIndexSpec struct {
	Replicas int    `json:"replicas"`
	PodType  string `json:"pod_type"`
}

// ConfigureIndexRequest is the request for configuring an index.
type ConfigureIndexRequest struct {
	IndexName string             `json:"index_name"`
	Spec      ConfigureIndexSpec `json:"spec"`
}

// ConfigureIndexResponse is the response for configuring an index.
type ConfigureIndexResponse struct {
	Name      string `json:"name"`
	Dimension int    `json:"dimension"`
	Metric    Metric `json:"metric"`
	Host      string `json:"host"`
	Spec      struct {
		Pod        Pod        `json:"pod"`
		Serverless Serverless `json:"serverless"`
	} `json:"spec"`
	Status Status `json:"status"`
}

func (c *ControlClient) ConfigureIndex(ctx context.Context, req *ConfigureIndexRequest) (*ConfigureIndexResponse, error) {
	resp, err := c.request(ctx, "PATCH", "/indexes/"+url.PathEscape(req.IndexName), req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		return nil, decodeError(resp)
	}

	var index ConfigureIndexResponse
	err = json.NewDecoder(resp.Body).Decode(&index)
	if err != nil {
		return nil, err
	}
	return &index, nil
}

// Collection is the collection configuration.
type Collection struct {
	Name        string `json:"name"`
	Size        int    `json:"size"`
	Status      string `json:"status"`
	Dimension   int    `json:"dimension"`
	VectorCount int    `json:"vector_count"`
	Environment string `json:"environment"`
}

// ListCollectionsResponse is the response for listing collections.
type ListCollectionsResponse struct {
	Collections []Collection `json:"collections"`
}

// ListCollections lists all collections.
func (c *ControlClient) ListCollections(ctx context.Context) (*ListCollectionsResponse, error) {
	resp, err := c.request(ctx, "GET", "/collections", nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var collections ListCollectionsResponse
	err = json.NewDecoder(resp.Body).Decode(&collections)
	if err != nil {
		return nil, err
	}
	return &collections, nil
}

// CreateCollectionRequest is the request for creating a collection.
type CreateCollectionRequest struct {
	Name   string `json:"name"`
	Source string `json:"source"`
}

// CreateCollectionResponse is the response for creating a collection.
type CreateCollectionResponse struct {
	Name        string `json:"name"`
	Size        int    `json:"size"`
	Status      string `json:"status"`
	Dimension   int    `json:"dimension"`
	VectorCount int    `json:"vector_count"`
	Environment string `json:"environment"`
}

// CreateCollection creates a new collection with the given configuration.
func (c *ControlClient) CreateCollection(ctx context.Context, req *CreateCollectionRequest) (*CreateCollectionResponse, error) {
	resp, err := c.request(ctx, "POST", "/collections", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		return nil, decodeError(resp)
	}

	var collection CreateCollectionResponse
	err = json.NewDecoder(resp.Body).Decode(&collection)
	if err != nil {
		return nil, err
	}
	return &collection, nil
}

// DecribeCollectionResponse is the response for describing a collection.
type DescribeCollectionResponse struct {
	Name        string `json:"name"`
	Size        int    `json:"size"`
	Status      string `json:"status"`
	Dimension   int    `json:"dimension"`
	VectorCount int    `json:"vector_count"`
	Environment string `json:"environment"`
}

// DescribeCollection describes a collection by name.
func (c *ControlClient) DescribeCollection(ctx context.Context, collectionName string) (*DescribeCollectionResponse, error) {
	resp, err := c.request(ctx, "GET", "/collections/"+url.PathEscape(collectionName), nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var collection DescribeCollectionResponse
	err = json.NewDecoder(resp.Body).Decode(&collection)
	if err != nil {
		return nil, err
	}
	return &collection, nil
}

// DeleteCollection deletes a collection by name.
func (c *ControlClient) DeleteCollection(ctx context.Context, collectionName string) error {
	resp, err := c.request(ctx, "DELETE", "/collections/"+url.PathEscape(collectionName), nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusAccepted {
		return decodeError(resp)
	}
	return nil
}

func (c *ControlClient) request(ctx context.Context, method string, path string, body any) (*http.Response, error) {
	url := c.baseURL + path

	var buf io.ReadWriter
	if body != nil {
		buf = &bytes.Buffer{}
		err := json.NewEncoder(buf).Encode(body)
		if err != nil {
			return nil, err
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, url, buf)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Api-Key", c.token)
	if body != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	return c.httpClient.Do(httpReq)
}

// DataClient is a client for the Pinecone data API.
type DataClient struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

// NewDataClient creates a new DataClient for the data API using the given index host and token.
func NewDataClient(indexHost, token string) *DataClient {
	return &DataClient{
		baseURL:    indexHost,
		token:      token,
		httpClient: http.DefaultClient,
	}
}

// SparseValue is a sparse vector.
type SparseValue struct {
	Indices []int     `json:"indices"`
	Values  []float32 `json:"values"`
}

// Usage represents the usage of the index.
type Usage struct {
	ReadUnits int `json:"readUnits"`
}

// Vector is a representation of a vector in the index.
type Vector struct {
	ID           string         `json:"id"`
	Values       []float32      `json:"values"`
	SparseValues []SparseValue  `json:"sparseValues,omitempty"`
	Metadata     map[string]any `json:"metadata,omitempty"`
}

// UpsertVectorsRequest is the request to upsert vectors.
type UpsertVectorsRequest struct {
	Vectors   []Vector `json:"vectors"`
	Namespace string   `json:"namespace,omitempty"`
}

// UpsertVectorsResponse is the response from the UpsertVectors API.
type UpsertVectorsResponse struct {
	UpsertedCount int `json:"upsertedCount"`
}

// UpsertVectors upserts vectors to the index.
func (c *DataClient) UpsertVectors(ctx context.Context, req *UpsertVectorsRequest) (*UpsertVectorsResponse, error) {
	resp, err := c.request(ctx, "POST", "/vectors/upsert", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var upserted UpsertVectorsResponse
	err = json.NewDecoder(resp.Body).Decode(&upserted)
	if err != nil {
		return nil, err
	}
	return &upserted, nil
}

// QueryVectorsRequest is the request to query vectors.
type QueryVectorsRequest struct {
	Vector          []float32      `json:"vector"`
	TopK            int            `json:"topK"`
	ID              string         `json:"id,omitempty"`
	Namespace       string         `json:"namespace,omitempty"`
	Filter          map[string]any `json:"filter,omitempty"`
	IncludeValues   bool           `json:"includeValues,omitempty"`
	IncludeMetadata bool           `json:"includeMetadata,omitempty"`
	SparseVector    *SparseVector  `json:"sparseVector,omitempty"`
}

// Validate checks if the request is valid.
func (r *QueryVectorsRequest) Validate() error {
	if r.TopK <= 0 {
		return fmt.Errorf("top k must be greater than 0")
	}
	return nil
}

// SparseVector is a sparse vector.
type SparseVector struct {
	Indices []int     `json:"indices"`
	Values  []float32 `json:"values"`
}

// QueryVectorsResponse is the response from the QueryVectors API.
type QueryVectorsResponse struct {
	Namespace string  `json:"namespace"`
	Matches   []Match `json:"matches"`
	Usage     Usage   `json:"usage"`
}

// Match represents a matching vector.
type Match struct {
	ID            string         `json:"id"`
	Score         float32        `json:"score,omitempty"`
	Values        []float32      `json:"values,omitempty"`
	SpareseValues []SparseValue  `json:"sparseValues,omitempty"`
	Metadata      map[string]any `json:"metadata,omitempty"`
}

// QueryVectors queries the index for vectors.
func (c *DataClient) QueryVectors(ctx context.Context, req *QueryVectorsRequest) (*QueryVectorsResponse, error) {
	if err := req.Validate(); err != nil {
		return nil, err
	}

	resp, err := c.request(ctx, "POST", "/query", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var queryResp QueryVectorsResponse
	err = json.NewDecoder(resp.Body).Decode(&queryResp)
	if err != nil {
		return nil, err
	}
	return &queryResp, nil
}

// FetchVectorsRequest is the request to fetch vectors.
type FetchVectorsRequest struct {
	IDs       []string `json:"ids"`
	Namespace string   `json:"namespace,omitempty"`
}

// FetchVectorsResponse is the response from the FetchVectors API.
type FetchVectorsResponse struct {
	Vectors map[string]struct {
		ID       string         `json:"id"`
		Values   []float32      `json:"values"`
		Metadata map[string]any `json:"metadata,omitempty"`
	} `json:"vectors"`
	Namespace string `json:"namespace,omitempty"`
	Usage     Usage  `json:"usage,omitempty"`
}

// FetchVectors fetches vectors from the index.
func (c *DataClient) FetchVectors(ctx context.Context, req *FetchVectorsRequest) (*FetchVectorsResponse, error) {
	resp, err := c.request(ctx, "GET", "/vectors/fetch", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var fetchResp FetchVectorsResponse
	err = json.NewDecoder(resp.Body).Decode(&fetchResp)
	if err != nil {
		return nil, err
	}
	return &fetchResp, nil
}

// UpdateVectorRequest is the request to update a vector.
type UpdateVectorRequest struct {
	ID           string         `json:"id"`
	Values       []float32      `json:"values,omitempty"`
	SparseValues []SparseValue  `json:"sparseValues,omitempty"`
	Metadata     map[string]any `json:"setMetadata,omitempty"`
	Namespace    string         `json:"namespace,omitempty"`
}

// UpdateVector updates a vector in the index.
func (c *DataClient) UpdateVector(ctx context.Context, req *UpdateVectorRequest) error {
	resp, err := c.request(ctx, "POST", "/vectors/update", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return decodeError(resp)
	}
	return nil
}

// DeleteVectorsRequest is the request to delete vectors.
type DeleteVectorsRequest struct {
	IDs       []string       `json:"ids"`
	DeleteAll *bool          `json:"deleteAll,omitempty"`
	Namespace string         `json:"namespace,omitempty"`
	Filter    map[string]any `json:"filter,omitempty"`
}

// DeleteVectors deletes vectors from the index.
func (c *DataClient) DeleteVectors(ctx context.Context, req *DeleteVectorsRequest) error {
	resp, err := c.request(ctx, "POST", "/vectors/delete", req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return decodeError(resp)
	}
	return nil
}

// ListVectorIDsRequest is the request to list vector IDs.
type ListVectorIDsRequest struct {
	Namespace       string `json:"namespace,omitempty"`
	Prefix          string `json:"prefix,omitempty"`
	Limit           int    `json:"limit,omitempty"`
	PaginationToken string `json:"paginationToken,omitempty"`
}

// ListVectorIDsResponse is the response from the ListVectorIDs API.
type ListVectorIDsResponse struct {
	Vectors []struct {
		ID string `json:"id"`
	} `json:"vectors"`
	Pagination struct {
		Next string `json:"next"`
	} `json:"pagination"`
	Namespace string `json:"namespace,omitempty"`
}

// ListVectorIDs lists the IDs of vectors in a single namespace of a serverless index.
func (c *DataClient) ListVectorIDs(ctx context.Context, req *ListVectorIDsRequest) (*ListVectorIDsResponse, error) {
	query := make(url.Values)
	if req.Namespace != "" {
		query.Set("namespace", req.Namespace)
	}
	if req.Prefix != "" {
		query.Set("prefix", req.Prefix)
	}
	if req.Limit > 0 {
		query.Set("limit", strconv.Itoa(req.Limit))
	}
	if req.PaginationToken != "" {
		query.Set("paginationToken", req.PaginationToken)
	}

	url := fmt.Sprintf("/vectors/list?%s", query.Encode())

	resp, err := c.request(ctx, "GET", url, nil)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var listResp ListVectorIDsResponse
	err = json.NewDecoder(resp.Body).Decode(&listResp)
	if err != nil {
		return nil, err
	}

	return &listResp, nil
}

// IndexStatsRequest is the request to get index stats.
type IndexStatsRequest struct {
	Filter map[string]any `json:"filter,omitempty"`
}

// IndexStatsResponse is the response from the IndexStats API.
type IndexStatsResponse struct {
	Namespaces       map[string]NamespaceSummary `json:"namespaces"`
	Dimension        int                         `json:"dimension"`
	IndexFullness    float32                     `json:"indexFullness"`
	TotalVectorCount int                         `json:"totalVectorCount"`
}

// NamespaceSummary represents a summary of a namespace's contents.
type NamespaceSummary struct {
	VectorCount int `json:"vectorCount"`
}

// IndexStats gets statistics about the index.
func (c *DataClient) IndexStats(ctx context.Context, req *IndexStatsRequest) (*IndexStatsResponse, error) {
	resp, err := c.request(ctx, "POST", "/describe_index_stats", req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, decodeError(resp)
	}

	var statsResp IndexStatsResponse
	err = json.NewDecoder(resp.Body).Decode(&statsResp)
	if err != nil {
		return nil, err
	}

	return &statsResp, nil
}

func (c *DataClient) request(ctx context.Context, method string, path string, body any) (*http.Response, error) {
	url := c.baseURL + path

	var buf io.ReadWriter
	if body != nil {
		buf = &bytes.Buffer{}
		err := json.NewEncoder(buf).Encode(body)
		if err != nil {
			return nil, err
		}
	}

	httpReq, err := http.NewRequestWithContext(ctx, method, url, buf)
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Api-Key", c.token)
	if body != nil {
		httpReq.Header.Set("Content-Type", "application/json")
	}
	return c.httpClient.Do(httpReq)
}

// ErrorResponse is an error response.
type ErrorResponse struct {
	Status int `json:"status"`
	Error  struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	}
}

func decodeError(resp *http.Response) error {
	var errResp ErrorResponse
	if err := json.NewDecoder(resp.Body).Decode(&errResp); err != nil {
		return fmt.Errorf("error decoding error response: %w", err)
	}
	return fmt.Errorf("%s: %s", errResp.Error.Code, errResp.Error.Message)
}
