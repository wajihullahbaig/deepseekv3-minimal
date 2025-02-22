import torch
from torch.optim import AdamW
from torch import nn
from training.checkpointing import save_checkpoint
from visualization.metrics import plot_metrics
from transformers import get_linear_schedule_with_warmup

def train(model, train_loader, test_loader,val_loader, config):
    model.to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    # Calculate total training steps
    total_steps = config['num_epochs'] * len(train_loader)

    # Create a learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=config.get('pad_token_id'))

    train_losses = []
    val_losses = []
    test_losses = []

    for epoch in range(config['num_epochs']):
        model.train()
        epoch_train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()  # Clear gradients at the start of each batch

            input_ids, attention_mask, target_ids = batch
            input_ids = input_ids.to(config['device'])
            attention_mask = attention_mask.to(config['device']) if attention_mask is not None else None
            target_ids = target_ids.to(config['device'])

            # Adjust attention mask for multi-head attention
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, :, None].expand(-1, model.config['num_heads'], -1, -1)

            outputs, mtp_outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)

            # Compute main loss
            main_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))

            # Get seq_len from mtp_outputs
            depth, seq_len, hidden_dim = mtp_outputs.size(1), mtp_outputs.size(2), mtp_outputs.size(3)

            # Use seq_len to compute MTP loss
            mtp_loss = 0.0
            for k in range(depth):
                mtp_output_k = mtp_outputs[:, k, :, :]  # [batch_size, seq_len, hidden_dim]
                mtp_logits = model.output_head(mtp_output_k)  # [batch_size, seq_len, vocab_size]
                target_k = target_ids[:, k * seq_len : (k + 1) * seq_len]  # Slice targets for depth k
                mtp_loss += criterion(mtp_logits.view(-1, mtp_logits.size(-1)), target_k.view(-1))
            mtp_loss /= depth

            total_loss = main_loss + mtp_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Update the learning rate scheduler

            epoch_train_loss += total_loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}/{len(train_loader)}, Train Loss: {total_loss.item():.4f}")       
            
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
      
        # Testing
        model.eval()
        epoch_test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, target_ids = batch
                input_ids = input_ids.to(config['device'])
                attention_mask = attention_mask.to(config['device']) if attention_mask is not None else None
                target_ids = target_ids.to(config['device'])

                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, :, None].expand(-1, model.config['num_heads'], -1, -1)

                outputs, mtp_outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)

                main_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                # Get seq_len from mtp_outputs
                depth, seq_len, hidden_dim = mtp_outputs.size(1), mtp_outputs.size(2), mtp_outputs.size(3)

                # Use seq_len to compute MTP loss
                mtp_loss = 0.0
                for k in range(depth):
                    mtp_output_k = mtp_outputs[:, k, :, :]  # [batch_size, seq_len, hidden_dim]
                    mtp_logits = model.output_head(mtp_output_k)  # [batch_size, seq_len, vocab_size]
                    target_k = target_ids[:, k * seq_len : (k + 1) * seq_len]  # Slice targets for depth k
                    mtp_loss += criterion(mtp_logits.view(-1, mtp_logits.size(-1)), target_k.view(-1))
                mtp_loss /= depth

                total_loss = main_loss + mtp_loss
                epoch_test_loss += total_loss.item()

        avg_test_loss = epoch_test_loss / len(test_loader)
        test_losses.append(avg_test_loss)

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Test Loss: {avg_test_loss:.4f}")             
    

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, target_ids = batch
                input_ids = input_ids.to(config['device'])
                attention_mask = attention_mask.to(config['device']) if attention_mask is not None else None
                target_ids = target_ids.to(config['device'])

                if attention_mask.dim() == 2:
                    attention_mask = attention_mask[:, None, :, None].expand(-1, model.config['num_heads'], -1, -1)

                outputs, mtp_outputs = model(input_ids, attention_mask=attention_mask, target_ids=target_ids)

                main_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                # Get seq_len from mtp_outputs
                depth, seq_len, hidden_dim = mtp_outputs.size(1), mtp_outputs.size(2), mtp_outputs.size(3)

                # Use seq_len to compute MTP loss
                mtp_loss = 0.0
                for k in range(depth):
                    mtp_output_k = mtp_outputs[:, k, :, :]  # [batch_size, seq_len, hidden_dim]
                    mtp_logits = model.output_head(mtp_output_k)  # [batch_size, seq_len, vocab_size]
                    target_k = target_ids[:, k * seq_len : (k + 1) * seq_len]  # Slice targets for depth k
                    mtp_loss += criterion(mtp_logits.view(-1, mtp_logits.size(-1)), target_k.view(-1))
                mtp_loss /= depth

                total_loss = main_loss + mtp_loss
                epoch_val_loss += total_loss.item()

        avg_val_loss = epoch_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{config['num_epochs']}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, config)

    plot_metrics(train_losses, test_losses,val_losses)
