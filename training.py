import torch
from tqdm import tqdm

def train(ultrasound_encoder, dataloaders, optimizer, num_epochs, loss_fn_mir, loss_fn_pm, loss_fn_io, device, checkpoint_path):
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n[Training] Starting epoch {epoch + 1}/{num_epochs}")
        ultrasound_encoder.train()
        total_epoch_loss = 0.0
        total_epoch_loss_mir = 0.0
        total_epoch_loss_pm = 0.0
        total_epoch_loss_io = 0.0

        tqdm_total = min(len(dataloaders['mir']), len(dataloaders['pm']), len(dataloaders['io']))

        for batch_idx, (batch_mir, batch_pm, batch_io) in enumerate(tqdm(zip(dataloaders['mir'], dataloaders['pm'], dataloaders['io']),
                                                                          total=tqdm_total)):

            # MIR task
            masked_images, original_images = batch_mir
            masked_images, original_images = masked_images.to(device), original_images.to(device)
            reconstructions, mu, logvar = ultrasound_encoder(masked_images, task='MIR')

            reconstruction_loss = loss_fn_mir(reconstructions, original_images)
            kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(masked_images)
            loss_mir = reconstruction_loss + kl_divergence
            total_epoch_loss_mir += loss_mir.item()

            print(f"[Training][Batch {batch_idx+1}] MIR Loss: {loss_mir.item():.4f} (Reconstruction: {reconstruction_loss.item():.4f}, KL: {kl_divergence.item():.4f})")

            # PM task
            (image1, image2), labels_pm = batch_pm
            image1, image2, labels_pm = image1.to(device), image2.to(device), labels_pm.to(device)
            similarity_score = ultrasound_encoder(image1, task='PM', x_pair=image2)
            loss_pm = loss_fn_pm(similarity_score.squeeze(), labels_pm)
            total_epoch_loss_pm += loss_pm.item()

            print(f"[Training][Batch {batch_idx+1}] PM Loss: {loss_pm.item():.4f}")

            # IO task
            sequences, labels_io = batch_io
            sequences, labels_io = sequences.to(device), labels_io.to(device)
            outputs_io = ultrasound_encoder(sequences, task='IO')
            labels_io = labels_io.argmax(dim=1)
            loss_io = loss_fn_io(outputs_io, labels_io)
            total_epoch_loss_io += loss_io.item()

            print(f"[Training][Batch {batch_idx+1}] IO Loss: {loss_io.item():.4f}")

            # Total batch loss
            total_batch_loss = loss_mir + loss_pm + loss_io
            total_epoch_loss += total_batch_loss.item()

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            print(f"[Training][Batch {batch_idx+1}] Total Batch Loss: {total_batch_loss.item():.4f}")

        avg_epoch_loss = total_epoch_loss / tqdm_total
        avg_epoch_loss_mir = total_epoch_loss_mir / tqdm_total
        avg_epoch_loss_pm = total_epoch_loss_pm / tqdm_total
        avg_epoch_loss_io = total_epoch_loss_io / tqdm_total

        print(f"\n[Training] Epoch [{epoch+1}] Summary:")
        print(f"  Total Loss: {avg_epoch_loss:.4f}")
        print(f"  MIR Loss: {avg_epoch_loss_mir:.4f}")
        print(f"  PM Loss: {avg_epoch_loss_pm:.4f}")
        print(f"  IO Loss: {avg_epoch_loss_io:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': ultrasound_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, checkpoint_path)
            print(f"[Training] Checkpoint saved at epoch {epoch+1} with loss {best_loss:.4f}")
