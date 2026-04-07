from typing import List, Dict, Any
from pydantic import BaseModel, Field, model_validator


# Request format for /v1/completions endpoint.
class CompletionRequest(BaseModel):
    prompt: str | List[str]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/completions endpoint.
class CompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str


class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str


# Request format for /v1/chat/completions endpoint.
class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 100
    temperature: float = 1.0
    stop: List[str] | None = None
    n: int = 1


# Response format for /v1/chat/completions endpoint.
class ChatCompletionResponse(BaseModel):
    id: str
    choices: List[Dict[str, Any]]
    created: int
    model: str


class PromptContainer(BaseModel):
    # This matches the "prompt": {"prompt_token_ids": [...]} structure
    prompt_token_ids: List[int]

class GenerateRequest(BaseModel):
    # 1. Handle the nested prompt object
    prompt: PromptContainer | None = None
    
    # 2. Keep these for backward compatibility if needed
    text: str | List[str] | None = None
    input_ids: List[int] | List[List[int]] | None = None
    
    # 3. Pull top-level sampling params into the model
    n: int = 1
    max_tokens: int = 256
    temperature: float = 1.0
    logprobs: int = 0
    
    # 4. Keep the existing dictionary if other clients use it
    sampling_params: Dict[str, Any] | None = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_inputs(self) -> "GenerateRequest":
        # If we got the 'prompt' wrapper, move those IDs to 'input_ids'
        if self.prompt and self.input_ids is None:
            self.input_ids = self.prompt.prompt_token_ids
        
        # Ensure sampling_params dict is populated from top-level fields
        # This makes the rest of your generate() logic work as-is
        if self.sampling_params is not None:
            self.sampling_params.setdefault("n", self.n)
            self.sampling_params.setdefault("max_new_tokens", self.max_tokens)
            self.sampling_params.setdefault("temperature", self.temperature)
            
        return self


# Response format for /generate endpoint (SGLang compatible).
# For single completion (n=1): returns one GenerateResponse
# For multiple completions (n>1): returns List[GenerateResponse]
class GenerateResponse(BaseModel):
    text: str | List[str]  # Generated text(s)
    meta_info: Dict[
        str, Any
    ]  # Contains: output_token_logprobs, finish_reason, prompt_tokens, completion_tokens


class TokenLogprob(BaseModel):
    token_id: int
    logprob: float
    token: str | None = None  # Decoded token text (when return_text=True)


# Request format for /logprobs endpoint.
class LogprobsRequest(BaseModel):
    input_ids: List[int] | None = None  # Token IDs to compute logprobs for
    text: str | None = None  # Text to tokenize then compute logprobs (alternative to input_ids)
    return_text: bool = False  # Whether to include decoded token strings


# Response format for /logprobs endpoint.
class LogprobsResponse(BaseModel):
    logprobs: List[TokenLogprob]  # Per-token logprobs
    num_tokens: int  # Total number of tokens in the input
