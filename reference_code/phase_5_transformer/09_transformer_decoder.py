# Source: AI_Learning_Cursor lines 19205-19392
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Complete Decoder Implementation
# Title: TRANSFORMER DECODER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
TRANSFORMER DECODER
===================
The decoding side of the Transformer (used in GPT, text generation).
"""

print("\n" + "=" * 60)
print("2. TRANSFORMER DECODER")
print("=" * 60)

"""
TRANSFORMER DECODER

Two types of attention in each block:
1. Masked Self-Attention: Attend to previous positions only (causal)
2. Cross-Attention (optional): Attend to encoder outputs

For GPT-style models (decoder-only):
- Only masked self-attention
- No cross-attention

For translation (encoder-decoder):
- Both masked self-attention and cross-attention
"""

class DecoderBlock(nn.Module):
    """
    Single Decoder Block.
    
    For decoder-only (GPT-style): cross_attention=False
    For encoder-decoder: cross_attention=True
    """
    
    def __init__(self, d_model, num_heads, d_ff=None, dropout=0.1, cross_attention=False):
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (optional, for encoder-decoder models)
        self.cross_attention = None
        if cross_attention:
            self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output=None, self_mask=None, cross_mask=None):
        """
        Args:
            x: Decoder input (batch, seq_len, d_model)
            encoder_output: Encoder output for cross-attention (batch, src_len, d_model)
            self_mask: Causal mask for self-attention
            cross_mask: Mask for cross-attention
        """
        # Masked self-attention
        self_attn = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(self_attn))
        
        # Cross-attention (if encoder-decoder model)
        if self.cross_attention is not None and encoder_output is not None:
            cross_attn = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
            x = self.norm2(x + self.dropout(cross_attn))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class TransformerDecoder(nn.Module):
    """
    Complete Transformer Decoder.
    
    For GPT-style (decoder-only): Set cross_attention=False
    For translation: Set cross_attention=True
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
        pad_idx=0,
        cross_attention=False
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Decoder blocks
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout, cross_attention)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Weight tying: share weights between input and output embeddings
        self.output_projection.weight = self.token_embedding.weight
        
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _create_causal_mask(self, seq_len, device):
        """Create causal (look-ahead) mask."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, x, encoder_output=None, src_mask=None):
        """
        Args:
            x: Token indices (batch, seq_len)
            encoder_output: From encoder, for cross-attention (optional)
            src_mask: Source padding mask for cross-attention
            
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        
        # Token + Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        for layer in self.layers:
            x = layer(x, encoder_output, causal_mask, src_mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits


# Test Decoder (GPT-style, decoder-only)
decoder = TransformerDecoder(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    num_layers=num_layers,
    cross_attention=False  # GPT-style
).to(device)

x = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
logits = decoder(x)

print(f"Decoder (GPT-style) configuration:")
print(f"  vocab_size: {vocab_size}")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  num_layers: {num_layers}")
print(f"\nInput shape: {x.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in decoder.parameters()):,}")
