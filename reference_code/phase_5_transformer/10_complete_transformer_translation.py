# Source: AI_Learning_Cursor lines 19400-19576
# Original transcript phase: 4 - THE TRANSFORMER ARCHITECTURE
# Nearest header: #### CODE: Complete Transformer Implementation
# Title: COMPLETE TRANSFORMER
#
# This file was extracted automatically by reference_code/_extract_from_transcript.py
# from the original Cursor session that produced this curriculum.
# It is intended as REFERENCE / SOLUTION code: build your own version FIRST,
# then diff your version against this one to find blind spots.

"""
COMPLETE TRANSFORMER
====================
Full encoder-decoder Transformer for sequence-to-sequence tasks.
"""

print("\n" + "=" * 60)
print("3. COMPLETE TRANSFORMER (ENCODER-DECODER)")
print("=" * 60)

class Transformer(nn.Module):
    """
    Complete Transformer for sequence-to-sequence tasks.
    
    Architecture:
    - Encoder: Processes source sequence
    - Decoder: Generates target sequence, attending to encoder output
    """
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_len=512,
        dropout=0.1,
        pad_idx=0
    ):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.d_model = d_model
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=src_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx
        )
        
        # Decoder with cross-attention
        self.decoder = TransformerDecoder(
            vocab_size=tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            d_ff=d_ff,
            max_len=max_len,
            dropout=dropout,
            pad_idx=pad_idx,
            cross_attention=True  # Enable cross-attention
        )
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Args:
            src: Source sequence (batch, src_len)
            tgt: Target sequence (batch, tgt_len)
            src_mask: Source padding mask
            tgt_mask: Target padding mask
            
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        # Encode source
        encoder_output = self.encoder(src, src_mask)
        
        # Decode with cross-attention to encoder
        logits = self.decoder(tgt, encoder_output, src_mask)
        
        return logits
    
    def encode(self, src, src_mask=None):
        """Encode source sequence."""
        return self.encoder(src, src_mask)
    
    def decode(self, tgt, encoder_output, src_mask=None):
        """Decode given encoder output."""
        return self.decoder(tgt, encoder_output, src_mask)


# Create full Transformer
src_vocab_size = 8000  # Source vocabulary (e.g., English)
tgt_vocab_size = 10000  # Target vocabulary (e.g., French)

transformer = Transformer(
    src_vocab_size=src_vocab_size,
    tgt_vocab_size=tgt_vocab_size,
    d_model=256,
    num_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,
    dropout=0.1
).to(device)

# Test forward pass
src = torch.randint(1, src_vocab_size, (batch_size, 20), device=device)
tgt = torch.randint(1, tgt_vocab_size, (batch_size, 25), device=device)

logits = transformer(src, tgt)

print(f"Full Transformer:")
print(f"  Source vocab: {src_vocab_size}")
print(f"  Target vocab: {tgt_vocab_size}")
print(f"  d_model: 256")
print(f"  Encoder layers: 4")
print(f"  Decoder layers: 4")
print(f"\nSource shape: {src.shape}")
print(f"Target shape: {tgt.shape}")
print(f"Output logits shape: {logits.shape}")
print(f"Total parameters: {sum(p.numel() for p in transformer.parameters()):,}")

# =================================
# 4. GENERATION (INFERENCE)
# =================================

print("\n" + "=" * 60)
print("4. TEXT GENERATION")
print("=" * 60)

def generate(model, src, max_len=50, start_token=1, end_token=2):
    """
    Generate output sequence autoregressively.
    
    Args:
        model: Trained Transformer
        src: Source sequence (1, src_len)
        max_len: Maximum generation length
        start_token: Start of sequence token
        end_token: End of sequence token
    """
    model.eval()
    device = src.device
    
    # Encode source
    encoder_output = model.encode(src)
    
    # Start with start token
    generated = torch.tensor([[start_token]], device=device)
    
    for _ in range(max_len):
        # Decode
        logits = model.decode(generated, encoder_output)
        
        # Get next token (greedy decoding)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        
        # Append to generated
        generated = torch.cat([generated, next_token], dim=1)
        
        # Stop if end token
        if next_token.item() == end_token:
            break
    
    return generated


# Test generation (with untrained model - just testing the mechanics)
src_test = torch.randint(1, src_vocab_size, (1, 10), device=device)
generated = generate(transformer, src_test, max_len=20)

print(f"Source: {src_test.squeeze().tolist()}")
print(f"Generated: {generated.squeeze().tolist()}")
print(f"\nNote: This is an untrained model, so output is random!")
print("After training, this would produce meaningful translations.")
