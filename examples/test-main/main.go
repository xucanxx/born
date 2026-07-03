package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/ebitengine/purego"
	"github.com/xucanxx/born/backend/webgpu"
	"github.com/xucanxx/born/generate"
	"github.com/xucanxx/born/models/llama"
	"github.com/xucanxx/born/tokenizer"
)

type MessageContext struct {
	Role    string
	Content string
}

type LLMClient struct{}

// BornClient 基于 born 框架的 LLM 客户端实现
type BornClient struct {
	model     *llama.Model[*webgpu.Backend] // 加载的 GGUF 模型
	tokenizer tokenizer.Tokenizer           // 分词器
	generator *generate.TextGenerator       // 文本生成器
	config    BornConfig                    // 配置信息
	LLMClient
}
type BornConfig struct {
	ModelPath   string   // GGUF 模型文件路径
	TokenType   string   // 分词器类型: "tiktoken", "bpe", "huggingface"
	VocabPath   string   // 词典文件路径（BPE 需要）
	MergesPath  string   // merges 文件路径（BPE 需要）
	Temperature float64  // 温度参数，默认 0.7
	TopP        float64  // Top-P 采样参数，默认 0.9
	TopK        int      // Top-K 采样参数，默认 40
	MaxTokens   int      // 最大生成 token 数，默认 2048
	StopStrings []string // 停止字符串
	StopTokens  []int32  // 停止 Token IDs
}

// NewBornClient 创建 born 客户端
func NewBornClient(config BornConfig) (*BornClient, error) {
	// 1. 加载 GGUF 模型
	backend, err := webgpu.New()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize WebGPU backend: %w", err)
	}
	model, err := llama.LoadGGUF(config.ModelPath, backend)
	if err != nil {
		return nil, fmt.Errorf("failed to load model from %s: %w", config.ModelPath, err)
	}

	// 2. 创建分词器
	tok, err := createTokenizer(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	// 3. 设置默认参数
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}
	if config.TopP == 0 {
		config.TopP = 0.9
	}
	if config.TopK == 0 {
		config.TopK = 40
	}
	if config.MaxTokens == 0 {
		config.MaxTokens = 2048
	}

	// 4. 设置针对特殊模型的结束符
	// Qwen 模型的结束符常常是 <|im_end|> 或者 151645
	if len(config.StopStrings) == 0 {
		config.StopStrings = []string{"<|im_end|>", "</s>"}
	}
	if len(config.StopTokens) == 0 {
		// 添加 Qwen 常见的结束符 id 151645, 151935，以及 LLaMA 等默认结束符
		config.StopTokens = []int32{151645, 151935, tok.EosToken()}
	}

	// 5. 创建生成器
	gen := generate.NewTextGenerator(model, tok, generate.SamplingConfig{
		Temperature: float32(config.Temperature),
		TopP:        float32(config.TopP),
		TopK:        config.TopK,
	})

	return &BornClient{
		model:     model,
		tokenizer: tok,
		generator: gen,
		config:    config,
	}, nil
}

// createTokenizer 根据配置创建分词器
func createTokenizer(config BornConfig) (tokenizer.Tokenizer, error) {
	switch config.TokenType {
	case "tiktoken":
		// 使用 TikToken 分词器，通常用于 GPT 系列模型
		return tokenizer.NewTikTokenForModel("gpt-4")
	case "bpe":
		// 使用 BPE 分词器，对于 born 框架可以使用 AutoLoad
		if config.VocabPath == "" {
			return nil, fmt.Errorf("tokenizer requires VocabPath for directory containing tokenizer.json")
		}
		return tokenizer.AutoLoad(config.VocabPath)
	case "huggingface":
		// 使用 HuggingFace 格式分词器
		if config.VocabPath == "" {
			return nil, fmt.Errorf("HuggingFace tokenizer requires VocabPath directory")
		}
		return tokenizer.LoadFromHuggingFace(config.VocabPath)
	default:
		// 如果提供了分词器路径，尝试自动加载
		if config.VocabPath != "" {
			return tokenizer.AutoLoad(config.VocabPath)
		}
		// 默认使用 TikToken
		return tokenizer.NewTikTokenForModel("gpt-4")
	}
}

// Chat 一次性聊天（非流式）
func (b *BornClient) Chat(ctx context.Context, msgs []MessageContext, system string) (string, error) {
	log.Println("Starting Chat with system prompt:", system)
	// 构建 prompt
	prompt := b.buildPrompt(msgs, system)

	// 非流式生成
	result, err := b.generator.Generate(prompt, generate.GenerateConfig{
		MaxTokens:   b.config.MaxTokens,
		Stream:      false,
		StopStrings: b.config.StopStrings,
		StopTokens:  b.config.StopTokens,
	})
	if err != nil {
		return "", fmt.Errorf("generation failed: %w", err)
	}

	return result, nil
}

// ChatStream 流式聊天
func (b *BornClient) ChatStream(ctx context.Context, msgs []MessageContext, system string, onToken func(string)) error {
	log.Println("Starting ChatStream with system prompt:", system)
	prompt := b.buildPrompt(msgs, system)

	// 流式生成
	stream, err := b.generator.GenerateStream(prompt, generate.GenerateConfig{
		MaxTokens:   b.config.MaxTokens,
		Stream:      true,
		StopStrings: b.config.StopStrings,
		StopTokens:  b.config.StopTokens,
	})
	// defer close(stream) // Generator is responsible for closing the channel
	if err != nil {
		return fmt.Errorf("stream generation failed: %w", err)
	}

	// 处理流式响应
	for chunk := range stream {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if chunk.Token != "" {
				onToken(chunk.Token)
			}
		}
	}

	return nil
}

// buildPrompt 构建 prompt
// 使用常见的聊天模板格式，兼容 LLaMA/Mistral 等模型
func (b *BornClient) buildPrompt(msgs []MessageContext, system string) string {
	var builder strings.Builder

	// 添加系统消息
	if system != "" {
		builder.WriteString(fmt.Sprintf("<|system|>\n%s\n", system))
	}

	// 添加历史消息
	for _, msg := range msgs {
		switch msg.Role {
		case "user":
			builder.WriteString(fmt.Sprintf("<|user|>\n%s\n", msg.Content))
		case "assistant":
			builder.WriteString(fmt.Sprintf("<|assistant|>\n%s\n", msg.Content))
		case "system":
			builder.WriteString(fmt.Sprintf("<|system|>\n%s\n", msg.Content))
		}
	}

	// 添加 assistant 提示，引导模型生成回复
	builder.WriteString("<|assistant|>\n")

	return builder.String()
}

// Close 释放资源
func (b *BornClient) Close() error {
	// born 的 loader.Model 可能不需要显式关闭
	// 但为了接口完整性保留此方法
	return nil
}

// GetConfig 获取当前配置
func (b *BornClient) GetConfig() BornConfig {
	return b.config
}

// GetModelInfo 获取模型信息
func (b *BornClient) GetModelInfo() string {
	return fmt.Sprintf("Model: %s, Tokenizer: %s", b.config.ModelPath, b.config.TokenType)
}

// /data/fanzengxu/project/minirag-go/models/qwen2.5-0.5b-instruct-q8_0.gguf
func main() {
	_, err := NewBornClient(BornConfig{
		ModelPath:   "/data/fanzengxu/project/minirag-go/models/qwen2.5-0.5b-instruct-q8_0.gguf",
		Temperature: 0.7,
		TopP:        0.9,
		TopK:        40,
		MaxTokens:   2048,
		// TokenType:   "bpe",
	})
	if err != nil {
		log.Fatalf("初始化 llm 客户端失败: %v", err)
	}
	_, err = purego.Dlopen("", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		panic(err)
	}
}
