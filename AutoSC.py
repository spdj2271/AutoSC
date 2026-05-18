import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from dataset import SequenceDataset
from utils import seed_everything, load_embedding, cluster_metric
from assignment_sequence_construction import AssignmentSequenceConstructor


class Config:
    seed: int = 1  # Random seed.

    dataset: str = ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-101", "ImageNet-Dogs",
                    "TinyImageNet", "DTD", "UCF-101", "ImageNet"][3]  # Dataset name.
    data_root: str = "./data_HFCLIP"  # Root directory of pre-extracted image embeddings.

    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # Device used for training and inference.

    len_sequences: int = 128  # Length of each assignment sequence.
    batch_size: int = 128  # Batch size for Transformer training.
    sequence_fill_last: bool = True  # Whether to complete the last short sequence by random sampling.

    embed_dim: int = 128  # Embedding dimension of the Transformer.
    n_head: int = 8  # Number of attention heads.
    n_layer: int = 6  # Number of Transformer decoder layers.

    lr: float = 1e-3  # Learning rate.
    num_epochs: int = 500  # Number of training epochs.

    n_trials: int = 16  # Number of trials for assignment sequence construction.
    anchor_size: int = 16  # Number of samples assigned at each construction step.
    perplexity: int = 200  # Perplexity for affinity graph construction.

    sinkhorn_reg: float = 0.2  # Entropic regularization strength used in Sinkhorn normalization.
    sinkhorn_iter: int = 4  # Number of Sinkhorn iterations.

    n_clusters: int = None  # Number of clusters, inferred from the dataset.
    BOS: int = None  # Beginning-of-sequence token, set to n_clusters.
    vocab_size: int = None  # Number of assignment tokens plus the BOS token.

    def __str__(self):
        return "\n".join(f"{k}: {getattr(self, k)}" for k in vars(self.__class__) if not k.startswith("__"))


class AutoSCTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=256, n_head=8, n_layer=6, vocab_size=12):
        """
        Autoregressive Transformer for assignment generation.

        input_dim:
            Dimension of input features.

        embed_dim:
            Hidden dimension of the Transformer.

        vocab_size:
            Number of assignment tokens.
            It equals n_clusters + 1, where the extra token is BOS.
        """
        super().__init__()

        # Project input features into the Transformer embedding space.
        #
        # Input:
        #   X: [B, T, input_dim]
        #
        # Output:
        #   memory: [B, T, embed_dim]
        #
        # Here, T is the assignment sequence length.
        self.visual_proj = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

        # Cluster assignments are treated as discrete tokens.
        #
        # For K clusters:
        #   cluster tokens: 0, 1, ..., K-1
        #   BOS token: K
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        # Transformer decoder models the conditional assignment distribution:
        #   p(y_t | y_<t, x_<=t)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        # Predict the next assignment token from the decoder output.
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, memory, tgt_idx, targets=None):
        """
        Forward pass of the autoregressive Transformer.

        memory:
            Projected input feature sequence.
            Shape: [B, T_mem, embed_dim]

        tgt_idx:
            Input assignment token sequence.
            During training, this is [BOS, y_1, ..., y_{T-1}].
            Shape: [B, T_tgt]

        targets:
            Ground-truth assignment sequence [y_1, ..., y_T].
            Shape: [B, T_tgt - 1]
        """
        T_tgt = tgt_idx.size(1)
        T_mem = memory.size(1)

        # Causal mask for assignment tokens.
        # It prevents the decoder from seeing future assignment tokens.
        tgt_mask = torch.triu(torch.ones(T_tgt, T_tgt, device=tgt_idx.device), diagonal=1).bool()

        # Causal mask for visual memory.
        # The t-th assignment prediction should not attend to future samples.
        memory_mask = torch.triu(torch.ones(T_tgt, T_mem, device=tgt_idx.device), diagonal=1).bool()

        # Convert assignment token indices to embeddings.
        tgt = self.token_emb(tgt_idx)

        # Decode assignment tokens conditioned on projected sample features.
        out = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        # Predict assignment logits for each position.
        logits = self.head(out)

        # Training loss.
        #
        # logits[:, :-1] corresponds to predictions after BOS and previous tokens.
        # targets corresponds to the desired assignments.
        if targets is not None:
            logits_for_loss = logits[:, :-1].reshape(-1, logits.size(-1))
            targets_for_loss = targets.reshape(-1)
            loss = F.cross_entropy(logits_for_loss, targets_for_loss)
        else:
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, memory, bos_token):
        """
        Generate one assignment token for each input sample.

        In evaluation, each sample is processed with a single BOS token.
        The model predicts its cluster assignment directly.
        """
        B = memory.shape[0]

        # Use BOS as the initial token.
        idx = torch.full((B, 1), bos_token, device=memory.device, dtype=torch.long)

        logits, _ = self.forward(memory, idx)

        # Return the most likely assignment token.
        return logits.argmax(dim=-1).squeeze(1)


def train_transformer(cfg, train_loader, test_loader):
    # Build the autoregressive Transformer for assignment generation.
    model = AutoSCTransformer(input_dim=cfg.input_dim, embed_dim=cfg.embed_dim, n_head=cfg.n_head,
                              n_layer=cfg.n_layer, vocab_size=cfg.vocab_size).to(cfg.device)

    # Optimizer for Transformer training.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    # Linearly decay the learning rate to zero during training.
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                                  total_iters=cfg.num_epochs * len(train_loader))

    # Train the Transformer on the constructed assignment sequences.
    for epoch in range(cfg.num_epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"{cfg.dataset} | Epoch {epoch}", ncols=150)
        epoch_loss = 0.0

        for ite, (x_batch, target_batch) in enumerate(pbar):
            x_batch = x_batch.to(cfg.device)
            target_batch = target_batch.to(cfg.device)

            # Project sample features into the Transformer memory space.
            memory = model.visual_proj(x_batch)

            # Use BOS as the first input token.
            # Input tokens:  [BOS, y_1, ..., y_{T-1}]
            # Target tokens: [y_1, y_2, ..., y_T]
            bos = torch.full((target_batch.size(0), 1), cfg.BOS, device=cfg.device, dtype=torch.long)
            target_seq = torch.cat([bos, target_batch], dim=1)

            # Predict the assignment sequence autoregressively.
            _, loss = model(memory=memory, tgt_idx=target_seq, targets=target_batch)

            # Standard optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record and display training loss.
            epoch_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}", "avg_loss": f"{epoch_loss / (ite + 1):.4f}"})

        # Evaluate the trained Transformer.
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_batch, labels_batch in test_loader:
                x_batch = x_batch.view(-1, cfg.input_dim).to(cfg.device)
                labels_batch = labels_batch.view(-1).to(cfg.device)
                # During evaluation, each sample is assigned independently with a BOS token.
                memory = model.visual_proj(x_batch).unsqueeze(1)
                preds = model.generate(memory=memory, bos_token=cfg.BOS)
                all_preds.append(preds.cpu())
                all_labels.append(labels_batch.cpu())
        # Compute clustering metrics.
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc, nmi, ari = cluster_metric(all_labels, all_preds)
        print(f"Epoch={epoch}, Loss={epoch_loss:.4f}, ACC={acc:.2f}, NMI={nmi:.2f}, ARI={ari:.2f}")


def main():
    # Initialize configuration.
    cfg = Config()

    # Set random seed for reproducibility.
    seed_everything(cfg.seed)

    # Load pre-extracted image embeddings.
    # If the dataset files are unavailable, fall back to a synthetic two-moons dataset.
    try:
        X, y = load_embedding(cfg.dataset, data_root=cfg.data_root)
    except Exception as e:
        print(f"Failed to load dataset: {cfg.dataset}. Error: {repr(e)}. Using sklearn two-moons dataset instead.")
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=1000, random_state=cfg.seed)
        X = X.astype(np.float32)
        cfg.dataset = "TwoMoons"

    # Infer the number of clusters and define the assignment vocabulary.
    cfg.input_dim = X.shape[1]
    n_clusters = len(np.unique(y))
    cfg.n_clusters = n_clusters
    cfg.BOS = n_clusters
    cfg.vocab_size = n_clusters + 1
    print(cfg)

    print("\n[Stage 1] Constructing the supervisory assignment sequence.")
    constructor = AssignmentSequenceConstructor(n_clusters=cfg.n_clusters, batch_size=cfg.batch_size,
                                                n_trials=cfg.n_trials, perplexity=cfg.perplexity,
                                                sinkhorn_reg=cfg.sinkhorn_reg, sinkhorn_iter=cfg.sinkhorn_iter)
    assignments, order = constructor.fit(X, y)

    # Report the quality of the constructed assignments.
    acc, nmi, ari = cluster_metric(y, assignments)
    print(f"Constructed assignment quality: ACC={acc:.3f}, NMI={nmi:.3f}, ARI={ari:.3f}")

    X, y, assignments = X[order], y[order], assignments[order]
    train_dataset = SequenceDataset(X=X, assignments=assignments, len_sequence=cfg.len_sequences,
                                    fill_last=cfg.sequence_fill_last, seed=cfg.seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=0)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=cfg.batch_size)

    print("\n[Stage 2] Training the autoregressive Transformer.")
    train_transformer(cfg=cfg, train_loader=train_loader, test_loader=test_loader)


if __name__ == "__main__":
    main()
