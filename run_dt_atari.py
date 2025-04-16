import os
import sys
sys.path.append('./decision-transformer/atari') 
import sys
print(f"Using Python: {sys.executable}")
from mingpt.model_atari import GPT, GPTConfig
import numpy as np
import h5py
import torch
import argparse
from collections import deque
import random
import time
from torch.utils.data import Dataset, DataLoader
import cv2  # Make sure cv2 is imported at the top
import ale_py
import matplotlib.pyplot as plt
class MinariAtariDataset(Dataset):
    def __init__(self, game_name, max_len=30, max_ep_len=1000, scale=1000.0, frame_stack=4):
        self.max_len = max_len
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.frame_stack = frame_stack
        
        # Load data from the HDF5 file
        h5_path = f"project_data/atari/{game_name}/expert-v0/data/main_data.hdf5"
        self.data = []
        self.process_h5_file(h5_path)
        
        # Create indices for all trajectories
        indices = []
        for i, trajectory in enumerate(self.data):
            # For each timestep t in the trajectory, we create a sequence from t to t+max_len
            # Need frame_stack-1 initial frames for context
            for t in range(len(trajectory['stacked_observations']) - max_len + 1):
                indices.append((i, t))
        self.indices = indices
        
        # Get state dimension after stacking
        if len(self.data) > 0 and len(self.data[0]['stacked_observations']) > 0:
            self.state_dim = self.data[0]['stacked_observations'][0].shape[0]
        else:
            # Default Atari dimension with 4 frames of 84x84
            self.state_dim = 84 * 84 * 4

    def preprocess_frame_for_dataset(self, frame):
        try:
            # Reshape the frame if it's in binary encoding
            if len(frame.shape) == 1:
                # Reshape to correct dimensions (210, 160, 3) for Atari
                try:
                    frame = frame.reshape(210, 160, 3)
                except:
                    # If reshaping fails, frame might be different than expected
                    pass
            
            # Convert to grayscale if it's RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                # If it's already grayscale, ensure it's the right shape
                frame = frame.astype(np.float32)
            
            # Resize to 84x84 (standard for Atari)
            frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            frame = frame / 255.0
            
            return frame
        except ImportError:
            # Fallback if OpenCV is not available
            # Convert to grayscale if it's RGB
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
            
            # Simple resize using numpy (less accurate but works without cv2)
            if frame.shape[0] != 84 or frame.shape[1] != 84:
                h, w = frame.shape
                frame = frame.reshape(h//84, 84, w//84, 84).mean(axis=(0,2))
            
            # Normalize pixel values
            frame = frame / 255.0
            
            return frame

    def process_h5_file(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            # Get episode keys
            episode_keys = list(f.keys())
            
            for episode_key in episode_keys:
                episode_data = f[episode_key]
                
                # Extract relevant data
                observations = np.array(episode_data['observations'][1:])
                actions = np.array(episode_data['actions'])
                print(f"Actions: {actions} {len(actions)}")
                rewards = np.array(episode_data['rewards'])
                dones = np.logical_or(np.array(episode_data['terminations']), np.array(episode_data['truncations']))
                
                # Create stacked observations
                stacked_observations = []
                frame_buffer = deque(maxlen=self.frame_stack)
                
                # Pre-fill buffer with the first frame
                preprocessed_frame = self.preprocess_frame_for_dataset(observations[0])
                for _ in range(self.frame_stack):
                    frame_buffer.append(preprocessed_frame)
                
                # Process remaining frames
                for i in range(len(observations)):
                    # Process current frame
                    processed_frame = self.preprocess_frame_for_dataset(observations[i])
                    
                    # Remove oldest frame and add new one
                    if len(frame_buffer) == self.frame_stack:
                        frame_buffer.popleft()
                    frame_buffer.append(processed_frame)
                    
                    # Stack frames
                    frames = list(frame_buffer)
                    # Check to ensure all frames have the same shape
                    shapes = [frame.shape for frame in frames]
                    assert all(shape == (84, 84) for shape in shapes), f"Frame shapes don't match: {shapes}"
                    
                    # Stack and flatten
                    stacked_frame = np.stack(frames, axis=0).flatten()  # Shape: (4*84*84,)
                    stacked_observations.append(stacked_frame)
                
                stacked_observations = np.array(stacked_observations)
                
                # Create trajectory dictionary
                trajectory = {
                    'observations': observations,
                    'stacked_observations': stacked_observations,
                    'actions': actions,
                    'rewards': rewards,
                    'returns_to_go': np.zeros_like(rewards),  # Will calculate below
                    'timesteps': np.arange(len(rewards)),
                    'dones': dones
                }
                
                # Calculate returns-to-go (sum of future rewards)
                returns_to_go = np.zeros_like(rewards)
                episode_return = 0
                for i in reversed(range(len(rewards))):
                    episode_return = rewards[i] + episode_return
                    returns_to_go[i] = episode_return
                
                trajectory['returns_to_go'] = returns_to_go
                self.data.append(trajectory)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        traj_idx, start_idx = self.indices[idx]
        trajectory = self.data[traj_idx]
        
        # Get a sequence of stacked states, actions, rewards, etc.
        states = trajectory['stacked_observations'][start_idx:start_idx + self.max_len]
        actions = trajectory['actions'][start_idx:start_idx + self.max_len]
        returns_to_go = trajectory['returns_to_go'][start_idx:start_idx + self.max_len]
        timesteps = trajectory['timesteps'][start_idx:start_idx + self.max_len]
        
        # Pad sequences if needed
        tlen = states.shape[0]
        states_pad = np.zeros((self.max_len, states.shape[1]), dtype=states.dtype)
        actions_pad = np.zeros((self.max_len), dtype=actions.dtype)
        returns_to_go_pad = np.zeros((self.max_len), dtype=returns_to_go.dtype)
        timesteps_pad = np.zeros((self.max_len), dtype=timesteps.dtype)
        
        states_pad[:tlen] = states
        actions_pad[:tlen] = actions
        returns_to_go_pad[:tlen] = returns_to_go
        timesteps_pad[:tlen] = timesteps
        
        # Create attention mask (0 for padding, 1 for actual tokens)
        attention_mask = np.zeros(self.max_len, dtype=np.int64)
        attention_mask[:tlen] = 1
        
        # Scale returns
        returns_to_go_pad = returns_to_go_pad / self.scale
        
        return {
            'states': torch.from_numpy(states_pad).float(),
            'actions': torch.from_numpy(actions_pad).long(),
            'returns_to_go': torch.from_numpy(returns_to_go_pad).float(),
            'timesteps': torch.from_numpy(timesteps_pad).long(),
            'attention_mask': torch.from_numpy(attention_mask).long()
        }

# MinGPT model classes
class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class CausalSelfAttention(torch.nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = torch.nn.Linear(config.n_embd, config.n_embd)
        self.query = torch.nn.Linear(config.n_embd, config.n_embd)
        self.value = torch.nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = torch.nn.Dropout(config.attn_pdrop)
        self.resid_drop = torch.nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = torch.nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        
        # Apply the attention mask from the calling function
        if attention_mask is not None:
            # Reshape attention mask to match attention matrix dimensions
            att = att.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        
        # And apply the causal mask
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = torch.nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(torch.nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(config.n_embd)
        self.ln2 = torch.nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(config.n_embd, 4 * config.n_embd),
            torch.nn.GELU(),
            torch.nn.Linear(4 * config.n_embd, config.n_embd),
            torch.nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(torch.nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.transformer = torch.nn.ModuleDict(dict(
            wte = torch.nn.Linear(config.n_embd, config.n_embd),
            wpe = torch.nn.Embedding(config.block_size, config.n_embd),
            drop = torch.nn.Dropout(config.embd_pdrop),
            h = torch.nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = torch.nn.LayerNorm(config.n_embd),
        ))
        
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, torch.nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs_embeds, attention_mask=None):
        device = inputs_embeds.device
        b, t, _ = inputs_embeds.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # forward the GPT model itself
        x = self.transformer.wte(inputs_embeds) + self.transformer.wpe(pos)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)
        x = self.transformer.ln_f(x)
        
        return {'logits': x}

# Modified DecisionTransformer model for Atari
class DecisionTransformer(GPT):
    def __init__(self, state_dim, act_dim, hidden_size, max_length=None, max_ep_len=4000, 
                 action_tanh=True, **kwargs):
        config = GPTConfig(
            block_size=3*max_length,
            vocab_size=act_dim,
            n_layer=kwargs.get('n_layer', 6),
            n_head=kwargs.get('n_head', 8),
            n_embd=hidden_size,
            dropout=kwargs.get('dropout', 0.1),
            embd_pdrop=kwargs.get('embd_pdrop', 0.1), 
            attn_pdrop=kwargs.get('attn_pdrop', 0.1),
            resid_pdrop=kwargs.get('resid_pdrop', 0.1),
        )
        
        super().__init__(config)
        
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.max_ep_len = max_ep_len
        
        # State, action, and return embeddings
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(1, hidden_size)  # For Atari, action is a single integer
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        
        self.action_head = torch.nn.Linear(hidden_size, self.act_dim)
        
    def forward(self, states, actions, returns_to_go, timesteps, attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        
        # Embed each modality
        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions.unsqueeze(-1).float())
        returns_embeddings = self.embed_return(returns_to_go.unsqueeze(-1))
        time_embeddings = self.embed_timestep(timesteps)
        
        # Add time embeddings to each modality
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings
        
        # Stack modalities
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        
        # Create attention mask that works with stacked inputs
        if attention_mask is not None:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)
        else:
            stacked_attention_mask = None
        
        # Forward pass through the transformer
        transformer_outputs = super().forward(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        
        # Get action predictions
        x = transformer_outputs['logits']
        
        # Extract only the action prediction logits from the sequence
        action_preds = self.action_head(x[:, 1::3])  # (batch_size, seq_length, act_dim)
        
        return action_preds
    
    def get_action(self, states, actions, returns_to_go, timesteps, **kwargs):
        # Used for inference
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1)
        returns_to_go = returns_to_go.reshape(1, -1)
        timesteps = timesteps.reshape(1, -1)
        
        if states.shape[1] < self.max_length:
            # Pad all inputs if needed
            padding_length = self.max_length - states.shape[1]
            attention_mask = torch.cat([torch.ones(1, states.shape[1]), torch.zeros(1, padding_length)], dim=1).to(states.device)
            
            states = torch.cat(
                [states, torch.zeros(1, padding_length, self.state_dim, device=states.device)], dim=1)
            actions = torch.cat(
                [actions, torch.zeros(1, padding_length, device=actions.device)], dim=1)
            returns_to_go = torch.cat(
                [returns_to_go, torch.zeros(1, padding_length, device=returns_to_go.device)], dim=1)
            timesteps = torch.cat(
                [timesteps, torch.zeros(1, padding_length, device=timesteps.device)], dim=1)
        else:
            attention_mask = torch.ones(1, states.shape[1], device=states.device)
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]
            attention_mask = attention_mask[:, -self.max_length:]
        
        action_preds = self.forward(
            states, actions, returns_to_go, timesteps, attention_mask=attention_mask,
        )
        
        # Get the action prediction for the last state
        action_preds = action_preds[0, -1]
        
        # Convert to action (for Atari, use argmax since actions are discrete)
        action = torch.argmax(action_preds).item()
        
        return action

def train_dt_atari(
    game_name='breakout',
    sequence_length=30,
    learning_rate=1e-4,
    weight_decay=1e-4,
    num_epochs=10,
    batch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=123,
    embed_dim=128,
    n_layer=6,
    n_head=8,
    dropout=0.1,
    save_dir='dt_models',
    eval_interval=1,  # Evaluate every epoch
):
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading {game_name} dataset...")
    dataset = MinariAtariDataset(
        game_name=game_name,
        max_len=sequence_length,
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Get observation and action dimensions
    state_dim = dataset.state_dim
    act_dim = 18  # Standard for Atari (check if this is correct for your games)
    
    print(f"State dim: {state_dim}, Action dim: {act_dim}")
    
    # Create Decision Transformer model
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden_size=embed_dim,
        max_length=sequence_length,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
    ).to(device)
    
    # Initialize training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    
    # Define loss function (CrossEntropyLoss for discrete actions)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Lists to store metrics
    all_epoch_losses = []
    batch_losses = []  # Track individual batch losses for more detailed plotting
    eval_returns = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        start_time = time.time()
        
        for batch_idx, batch in enumerate(train_loader):
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            returns_to_go = batch['returns_to_go'].to(device)
            timesteps = batch['timesteps'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            action_preds = model(states, actions, returns_to_go, timesteps, attention_mask)
            
            # Calculate loss only on actual sequences (not padding)
            action_preds = action_preds.reshape(-1, act_dim)
            target_actions = actions.reshape(-1)
            
            # Create a mask to identify non-padding positions
            mask = attention_mask.reshape(-1).bool()
            
            # Apply mask and calculate loss
            loss = loss_fn(action_preds[mask], target_actions[mask])
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            
            batch_loss = loss.item()
            epoch_losses.append(batch_loss)
            batch_losses.append(batch_loss)  # Track all batch losses
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {batch_loss:.4f}")
        
        # End of epoch statistics
        mean_loss = np.mean(epoch_losses)
        all_epoch_losses.append(mean_loss)
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed. Mean loss: {mean_loss:.4f}, Time: {elapsed_time:.2f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(save_dir, f"{game_name}_dt_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': mean_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Evaluate model every eval_interval epochs
        if (epoch + 1) % eval_interval == 0:
            print(f"Evaluating model at epoch {epoch+1}...")
            model.eval()
            
            try:
                avg_return = evaluate_dt_atari(
                    model=model,
                    game_name=game_name,
                    num_eval_episodes=5,  # Reduced from 10 to 5 for faster evaluation
                    target_return=100,
                    device=device
                )
                eval_returns.append((epoch+1, avg_return))
                print(f"Evaluation at epoch {epoch+1}: Average return = {avg_return:.2f}")
                
                # Save evaluation results
                with open(os.path.join(save_dir, f"{game_name}_eval_returns.txt"), 'a') as f:
                    f.write(f"Epoch {epoch+1}: {avg_return:.2f}\n")
                    
            except Exception as e:
                print(f"Evaluation failed: {e}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, f"{game_name}_dt_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save training losses
    loss_path = os.path.join(save_dir, f"{game_name}_training_losses.txt")
    with open(loss_path, 'w') as f:
        for epoch, loss in enumerate(all_epoch_losses):
            f.write(f"Epoch {epoch+1}: {loss:.6f}\n")
    print(f"Training losses saved to {loss_path}")
    
    # Plot and save loss curves
    
    # 1. Epoch-wise loss curve
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), all_epoch_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss by Epoch for {game_name}')
    plt.grid(True)
    
    # 2. Batch-wise loss curve (more detailed)
    plt.subplot(1, 2, 2)
    plt.plot(batch_losses, 'r-', alpha=0.7)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss by Batch for {game_name}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{game_name}_loss_curves.png"))
    print(f"Loss curves saved to {save_dir}/{game_name}_loss_curves.png")
    
    # Plot evaluation returns if we have any
    if eval_returns:
        epochs, returns = zip(*eval_returns)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, returns, 'g-o', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Average Return')
        plt.title(f'Evaluation Returns for {game_name}')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, f"{game_name}_eval_returns.png"))
        print(f"Evaluation returns plot saved to {save_dir}/{game_name}_eval_returns.png")
    
    return model, all_epoch_losses, eval_returns


class SimpleFrameStack:
    def __init__(self, env, num_stack=4):
        self.env = env
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(num_stack, 84, 84), 
            dtype=np.float32
        )
        self.action_space = env.action_space
        
    def reset(self):
        obs, info = self.env.reset()
        processed = preprocess_frame(obs)
        for _ in range(self.num_stack):
            self.frames.append(processed)
        return self._get_obs(), info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(preprocess_frame(obs))
        return self._get_obs(), reward, terminated, truncated, info
        
    def _get_obs(self):
        return np.array(self.frames)
        
    def render(self):
        return self.env.render()
        
    def close(self):
        return self.env.close()
# Evaluation function
def evaluate_dt_atari(model, game_name, num_eval_episodes=10, target_return=100, device='cuda', 
                      render=True, render_freq=1, save_video=False, video_dir='videos'):
  
    # Set up video recording if requested
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    
    # Use standard Atari environment
    env_name = f"ALE/{game_name[0].upper() + game_name[1:]}-v5"
    
    try:
        print(f"Creating environment: {env_name}")
        # Set render_mode based on whether we're rendering or not
        render_mode = "human" if render else "rgb_array"
        env = gym.make(env_name, render_mode=render_mode)
        print(f"Successfully created environment: {env_name}")
    except Exception as e:
        print(f"Failed to create environment: {e}")
        # Try alternate naming
        alt_env_name = f"{game_name[0].upper() + game_name[1:]}-v5"
        try:
            render_mode = "human" if render else "rgb_array"
            env = gym.make(alt_env_name, render_mode=render_mode)
            print(f"Successfully created environment: {alt_env_name}")
        except Exception as e2:
            print(f"Failed again: {e2}")
            return 0  # Return 0 as fallback
    
    print("Starting evaluation...")
    
    # Set model to evaluation mode
    model.eval()
    model.to(device)
    
    returns = []
    for ep in range(num_eval_episodes):
        state, info = env.reset()
        done = False
        truncated = False
        
        # For video recording
        if save_video:
            video_frames = []
            video_path = os.path.join(video_dir, f"{game_name}_episode_{ep+1}.mp4")
        
        # Track lives for games like Breakout
        if 'lives' in info:
            initial_lives = info['lives']
            current_lives = initial_lives
            print(f"Game has lives tracking: {initial_lives} initial lives")
        else:
            initial_lives = None
            current_lives = None
            print("Game does not have lives tracking")
        
        # Initialize frame buffer for stacking
        frame_buffer = deque(maxlen=4)
        
        # Preprocess and fill buffer with initial frame
        processed_frame = preprocess_frame(state)
        for _ in range(4):
            frame_buffer.append(processed_frame)
        
        # Initialize sequence with zeros
        states = torch.zeros((1, model.max_length, model.state_dim), device=device)
        actions = torch.zeros((1, model.max_length), dtype=torch.long, device=device)
        returns_to_go = torch.zeros((1, model.max_length), device=device)
        timesteps = torch.arange(start=0, end=model.max_length, step=1).unsqueeze(0).to(device)
        
        # Set initial target return (scaled same as in training)
        returns_scale = 1000.0  # Should match scale in training
        target_return_tensor = torch.tensor([target_return], device=device)
        
        episode_return = 0
        t = 0
        
        print(f"Episode {ep+1}, Target return: {target_return}")
        
        while not (done or truncated) and t < 10000:  # Max episode length cap
            # Render based on frequency setting
            if render and t % render_freq == 0:
                frame = env.render()
                if save_video and frame is not None:
                    video_frames.append(frame)
            
            # Stack and flatten current frames
            stacked_frames = np.stack(list(frame_buffer), axis=0).flatten()
            state_tensor = torch.tensor(stacked_frames, dtype=torch.float).unsqueeze(0).to(device)
            
            # Update history
            if t < model.max_length:
                states[0, t] = state_tensor
                timesteps[0, t] = min(t, model.max_ep_len-1)
                returns_to_go[0, t] = target_return_tensor - episode_return
            else:
                # Shift sequences and update last position
                states = torch.roll(states, -1, dims=1)
                actions = torch.roll(actions, -1, dims=1)
                returns_to_go = torch.roll(returns_to_go, -1, dims=1)
                timesteps = torch.roll(timesteps, -1, dims=1)
                
                states[0, -1] = state_tensor
                timesteps[0, -1] = min(t, model.max_ep_len-1)
                returns_to_go[0, -1] = target_return_tensor - episode_return
            
            # Get action from model
            with torch.no_grad():
                action = model.get_action(
                    states,
                    actions,
                    returns_to_go / returns_scale,
                    timesteps,
                    t=t,
                )

            # Convert to tensor if it's an int
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, device=device)

            # Update action history
            if t < model.max_length:
                actions[0, t] = action
            else:
                actions[0, -1] = action
            
            # Execute action
            action_int = action.item() if isinstance(action, torch.Tensor) else int(action)
            next_state, reward, done, truncated, info = env.step(action_int)
            episode_return += reward
            
            # Check if we lost a life - important for games like Breakout
            if initial_lives is not None and 'lives' in info:
                if info['lives'] < current_lives:
                    current_lives = info['lives']
                    print(f"Lost a life! Remaining lives: {current_lives}")
                    
                    # If we've lost all lives or hit the configured limit,
                    # we should end the episode
                    if current_lives == 0 or (initial_lives - current_lives >= 5):
                        print(f"Lost 5 lives or all lives, ending episode")
                        break  # This will end the episode after losing 5 lives
            
            # Update frame buffer with new processed frame
            processed_frame = preprocess_frame(next_state)
            frame_buffer.append(processed_frame)
            
            # Move to next state
            t += 1
            
            # Print progress
            if t % 100 == 0:
                print(f"Episode {ep+1}, Step {t}, Return so far: {episode_return}")
        
        returns.append(episode_return)
        print(f"Episode {ep+1}/{num_eval_episodes}: Return = {episode_return}")
        
        # Save video if requested
        if save_video and video_frames:
            try:
                height, width, layers = video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
                
                for frame in video_frames:
                    video.write(frame)
                video.release()
                print(f"Video saved to {video_path}")
            except Exception as e:
                print(f"Failed to save video: {e}")
    
    env.close()
    
    if returns:
        avg_return = sum(returns) / len(returns)
        print(f"Average return over {num_eval_episodes} episodes: {avg_return:.2f}")
        return avg_return
    else:
        print("No complete returns recorded.")
        return 0
def preprocess_frame(frame):
    """
    Preprocess frame using same method as in training dataset.
    """
    import cv2
    import numpy as np
    
    # Reshape if needed
    if len(frame.shape) == 1:
        try:
            frame = frame.reshape(210, 160, 3)
        except:
            pass
    
    # Convert to grayscale if RGB
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Resize to 84x84
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    
    # Normalize to [0,1]
    frame = frame / 255.0
    
    return frame

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', type=str, default='breakout', help='Atari game to train on')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (-1 for CPU)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument('--eval', action='store_true', help='Whether to evaluate model')
    parser.add_argument('--model_path', type=str, default=None, help='Path to load model for evaluation')
    parser.add_argument('--target_return', type=float, default=10.0, help='Target return for evaluation')
    args = parser.parse_args()

    # Set device
    device = 'cpu' if args.gpu < 0 else f'cuda:{args.gpu}'
    print(f"Using device: {device}")
    
    if args.eval and args.model_path:
        # Load model for evaluation
        try:
            # Determine state_dim and act_dim for the model
            eval_dataset = MinariAtariDataset(game_name=args.game, max_len=30)
            state_dim = eval_dataset.state_dim
            act_dim = 18  # Standard for Atari
            
            # Create model with same architecture
            model = DecisionTransformer(
                state_dim=state_dim,
                act_dim=act_dim,
                hidden_size=128,
                max_length=30,
                n_layer=6,
                n_head=8,
            ).to(device)
            
            # Load trained weights
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
            
            # Evaluate model
            evaluate_dt_atari(model, args.game, target_return=args.target_return, device=device)
        except Exception as e:
            print(f"Error loading or evaluating model: {e}")
    else:
        # Train model
        train_dt_atari(
            game_name=args.game,
            device=device,
            num_epochs=args.epochs,
            seed=args.seed,
        )