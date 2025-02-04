import torch
from torch.optim import AdamW
from torch import nn
from test_text_generation import GenerationConfig, TextGenerator
from training.checkpointing import save_checkpoint
from visualization.metrics import plot_metrics

def train(model, train_loader, val_loader, config, tokenizer, pad_token_id=None):
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)  # Ignore padding tokens in loss computation

    # Initialize lists to track losses
    train_losses = []
    val_losses = []

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_train_loss = 0.0

        # Training
        total_batches = len(train_loader)
        optimizer.zero_grad()                        
        for index,batch in enumerate(train_loader):
            input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(config['device'])
            attention_mask = attention_mask.to(config['device'])
            target_ids = target_ids.to(config['device'])
            
            # Reshape attention mask to match model's expected shape
            if attention_mask.dim() == 2:  # [batch_size, seq_len]
                # Expand to [batch_size, num_heads, seq_len, seq_len]
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
                attention_mask = attention_mask.expand(-1, model.config['num_heads'], -1, -1)  # [batch_size, num_heads, 1, seq_len]
                attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)  # [batch_size, num_heads, seq_len, seq_len]
            
            outputs, mtp_outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)
            
            # Compute main loss
            main_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
            
            # Compute MTP loss
            mtp_loss = 0.0
            for k in range(mtp_outputs.size(1)):
                mtp_output_k = mtp_outputs[:, k, 0, :]  # Shape: [batch_size, vocab_size]
                target_k = target_ids[:, k]  # Shape: [batch_size]
                loss_k = criterion(mtp_output_k, target_k)
                mtp_loss += loss_k
            mtp_loss /= mtp_outputs.size(1)
                        
            total_loss = total_loss = main_loss + mtp_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)           
            optimizer.step()            
            optimizer.zero_grad()
            torch.cuda.empty_cache()            
            
            epoch_train_loss += total_loss.item()
            if index % 100 == 0:                
                print(f"Epoch: {epoch} Batch/Batches: {index}/{total_batches} Train Loss: {total_loss.item()}")

            if index % 500 == 0:                
                generator = TextGenerator(model, tokenizer)
                
                gen_config = GenerationConfig(
                    max_length=50,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                prompts = [
                    "The future of artificial intelligence "
                ]                
                for prompt in prompts:
                    generated_text = generator.generate(prompt, gen_config)
                    print(f"Generated text for prompt:\n{prompt}\n{generated_text}\n")
                    print("--------------------------------------------------")
                model.train()

        # Average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, target_ids = batch
                input_ids = input_ids.to(config['device'])
                attention_mask = attention_mask.to(config['device'])
                target_ids = target_ids.to(config['device'])
                
                # Reshape attention mask to match model's expected shape
                if attention_mask.dim() == 2:  # [batch_size, seq_len]
                    # Expand to [batch_size, num_heads, seq_len, seq_len]
                    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, seq_len]
                    attention_mask = attention_mask.expand(-1, model.config['num_heads'], -1, -1)  # [batch_size, num_heads, 1, seq_len]
                    attention_mask = attention_mask.expand(-1, -1, attention_mask.size(-1), -1)  # [batch_size, num_heads, seq_len, seq_len]

                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Compute main loss
                main_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                
                # Compute MTP loss
                mtp_loss = 0.0
                for k in range(mtp_outputs.size(1)):
                    mtp_output_k = mtp_outputs[:, k, 0, :]  # Shape: [batch_size, vocab_size]
                    target_k = target_ids[:, k]  # Shape: [batch_size]
                    loss_k = criterion(mtp_output_k, target_k)
                    mtp_loss += loss_k
                mtp_loss /= mtp_outputs.size(1)
                
                # Total loss
                total_loss = main_loss
                epoch_val_loss += total_loss.item()

        # Average validation loss for the epoch
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['num_epochs']}, "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        save_checkpoint(model, optimizer, epoch, config)        

    plot_metrics(train_losses, val_losses)