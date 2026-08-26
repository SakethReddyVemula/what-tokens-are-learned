import torch
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertForMaskedLM

def main():
    # 1. Setup matching your Medium model specs
    vocab_size = 10005
    batch_size = 64
    seq_length = 128
    mlm_probability = 0.15

    config = BertConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=384,
        initializer_range=0.02,
        intermediate_size=1024,
        layer_norm_eps=1e-12,
        max_position_embeddings=128,
        num_attention_heads=6,
        num_hidden_layers=6,
        vocab_size=vocab_size
    )

    # 2. Initialize Model with PyTorch's pristine random weights
    model = BertForMaskedLM(config)
    model.eval()

    # 3. Create a dummy batch mimicking DataCollator outputs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))

    # Labels: -100 for 85% of tokens, random target vocab ids for 15% masked
    labels = torch.full((batch_size, seq_length), -100)
    mask = torch.rand((batch_size, seq_length)) < mlm_probability
    labels[mask] = torch.randint(0, vocab_size, (int(mask.sum()),))

    # 4. Calculate loss using the completely untouched model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        initial_model_loss = outputs.loss.item()
        
        # Logits variance analysis
        logits_std = outputs.logits.std().item()

    print("=== MLM Loss Diagnostics ===")
    
    # 5. Mathematical Uniform Random Loss 
    # (If the model predicted exactly 1 / vocab_size evenly for every potential word)
    uniform_loss = -torch.log(torch.tensor(1.0 / vocab_size)).item()
    print(f"1) Mathematical Uniform Random Loss:  {uniform_loss:.4f}")
    
    print(f"2) Untrained BERT Model Initial Loss: {initial_model_loss:.4f}  (Logits StdDev: {logits_std:.4f})")

    # 6. Demonstrate what happens when weights/logits explode (The Post-LN Warmup Spike Issue)
    # We artificially inject a higher variance into the logits as if a violent gradient update occurred
    extreme_logits = torch.randn(batch_size * seq_length, vocab_size) * 10.0
    loss_fct = CrossEntropyLoss(ignore_index=-100)
    extreme_loss = loss_fct(extreme_logits, labels.view(-1)).item()
    
    print(f"3) Exploded/High-Variance Model Loss: {extreme_loss:.4f}  (Simulating aggressive initial Adam step)")

if __name__ == "__main__":
    main()

