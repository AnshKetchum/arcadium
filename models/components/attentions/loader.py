from pydantic import BaseModel
from models.components.attentions.grouped_query_attention import GroupedQueryAttention
from models.components.attentions.multi_head_attention import MultiHeadAttention
from models.components.attentions.multi_query_attention import MultiQueryAttention

class AttentionParameters(BaseModel):
    type: str
    num_query_heads: int
    num_kv_heads: int
    embedding_dimension: int 
    head_dimension: int

def load_attention(config: AttentionParameters):
    MULTI_HEAD = "multi-head" 
    MULTI_QUERY = "multi-query" 
    GROUP_QUERY = "group-query" 

    attention_config = config.model_dump()

    if attention_config["type"] == MULTI_HEAD:
        return MultiHeadAttention(**attention_config)
        
    if attention_config["type"] == MULTI_QUERY:
        return MultiQueryAttention(**attention_config)

    if attention_config["type"] == GROUP_QUERY:
        return GroupedQueryAttention(**attention_config)
    
    raise ModuleNotFoundError(f"Cannot find attention type. It can only be - {MULTI_HEAD}, {MULTI_QUERY}, or {GROUP_QUERY}")