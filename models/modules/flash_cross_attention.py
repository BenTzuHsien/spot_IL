import torch
import torch.nn as nn

torch.backends.cuda.enable_flash_sdp(True)          
torch.backends.cuda.enable_mem_efficient_sdp(True) 
torch.backends.cuda.enable_math_sdp(False)    

class FlashCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(FlashCrossAttention, self).__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            # batch_first = True,       
            # bias        = False
        )

    def forward(self, query, key_value):
        # query and key_value: (B, C, H, W)
        B, C, H, W = query.shape     

        # print(f"[Cross-Attention] query {query.shape}  key_value {key_value.shape}")            

        # (H*W, B, C)
        query_flat = query.view(B, C, -1).permute(2, 0, 1)
        key_value_flat = key_value.view(B, C, -1).permute(2, 0, 1)
        attn_output, attn_weight = self.mha(query_flat, key_value_flat, key_value_flat)
        # print(f"[Cross-Attention] attn_output {attn_output.shape} attn_weights {attn_weight.shape}")
       
       
        # (B, C, H, W)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)
        # print(f"[Cross-Attention] out {attn_output.shape}") 
        
        return attn_output, attn_weight