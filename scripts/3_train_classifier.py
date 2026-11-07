# ============================================================================
# FILE: 3_train_classifier_fixed.py
# COMPLETE FIXED DISTRIBUTED TRAINING
# ============================================================================

import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DISTRIBUTED SETUP FUNCTIONS
# ============================================================================

def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        # Single GPU or CPU mode
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

# ============================================================================
# DATASET CLASS
# ============================================================================

class DistributedFallacyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256, cache=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cached_encodings = None
        
        if cache:
            print(f"Caching {len(texts)} samples...")
            self.cached_encodings = []
            for text in tqdm(texts):
                encoding = self.tokenizer(
                    str(text),
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                self.cached_encodings.append(encoding)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        if self.cached_encodings:
            encoding = self.cached_encodings[idx]
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

# ============================================================================
# DISTRIBUTED TRAINER CLASS
# ============================================================================

class DistributedFallacyTrainer:
    def __init__(self, model_name='distilbert-base-uncased'):
        self.model_name = model_name
        self.rank, self.local_rank, self.world_size = setup_distributed()
        self.device = torch.device(f'cuda:{self.local_rank}' if torch.cuda.is_available() else 'cpu')
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        
        # Store data
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
        self.num_labels = None
        self.label_map = None
        self.id_to_label = None
        
        if is_main_process(self.rank):
            print(f"\n{'='*60}")
            print(f"DISTRIBUTED TRAINING CONFIGURATION")
            print(f"{'='*60}")
            print(f"World Size: {self.world_size} GPUs")
            print(f"Local Rank: {self.local_rank}")
            print(f"Device: {self.device}")
            print(f"Mixed Precision: Enabled")
            if torch.cuda.is_available():
                print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def load_data(self, train_file, val_file, test_file, label_map_path=None):
        """Load and prepare data"""
        
        if is_main_process(self.rank):
            print(f"\n📂 Loading data...")
        
        # Load CSV files
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
        test_df = pd.read_csv(test_file)
        
        # Load or create label map
        if label_map_path and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
        else:
            unique_labels = sorted(train_df['label'].unique())
            self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        self.num_labels = len(self.label_map)
        
        # Convert labels
        self.train_labels = train_df['label'].map(self.label_map).fillna(0).astype(int).values
        self.val_labels = val_df['label'].map(self.label_map).fillna(0).astype(int).values
        self.test_labels = test_df['label'].map(self.label_map).fillna(0).astype(int).values
        
        train_texts = train_df['text'].astype(str).tolist()
        val_texts = val_df['text'].astype(str).tolist()
        test_texts = test_df['text'].astype(str).tolist()
        
        if is_main_process(self.rank):
            print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
            print(f"Classes: {self.num_labels}")
        
        return train_texts, self.train_labels, val_texts, self.val_labels, test_texts, self.test_labels
    
    def initialize_model(self):
        """Initialize model with distributed support"""
        
        if is_main_process(self.rank):
            print(f"\n🤖 Initializing model on {self.world_size} GPUs...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True
        )
        model = model.to(self.device)
        
        if self.world_size > 1:
            self.model = DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)
        else:
            self.model = model
        
        if is_main_process(self.rank):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_dataloaders(self, train_texts, train_labels, val_texts, val_labels, 
                          test_texts, test_labels, batch_size=16):
        """Create distributed dataloaders"""
        
        train_dataset = DistributedFallacyDataset(train_texts, train_labels, self.tokenizer, cache=True)
        val_dataset = DistributedFallacyDataset(val_texts, val_labels, self.tokenizer, cache=False)
        test_dataset = DistributedFallacyDataset(test_texts, test_labels, self.tokenizer, cache=False)
        
        if self.world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, 
                                               rank=self.rank, shuffle=True, seed=42)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size,
                                            rank=self.rank, shuffle=False)
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size,
                                             rank=self.rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
            test_sampler = None
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, sampler=val_sampler,
                                shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, sampler=test_sampler,
                                 shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader, test_loader, train_sampler
    
    def setup_optimization(self, train_loader, train_labels, total_epochs=10, lr=2e-5):
        """Setup optimizer, scheduler, and loss - FIXED: now receives train_labels"""
        
        # Class weights for imbalance
        unique_classes = np.unique(train_labels)
        class_weights = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        
        # Scheduler
        total_steps = len(train_loader) * total_epochs
        self.scheduler = OneCycleLR(self.optimizer, max_lr=lr, total_steps=total_steps,
                                    pct_start=0.1, anneal_strategy='cos')
        
        if is_main_process(self.rank):
            print(f"\n📊 Optimization setup:")
            print(f"   Optimizer: AdamW, LR: {lr}")
            print(f"   Scheduler: OneCycleLR")
            print(f"   Mixed Precision: Enabled")
            print(f"   Class weights: {class_weights[:3].tolist()}...")
    
    def train_epoch(self, train_loader, epoch, train_sampler=None):
        """Train for one epoch"""
        
        if train_sampler:
            train_sampler.set_epoch(epoch)
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/10") if is_main_process(self.rank) else train_loader
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if is_main_process(self.rank):
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        
        # Synchronize loss across GPUs
        if self.world_size > 1:
            loss_tensor = torch.tensor(avg_loss).to(self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_loader, desc="Validating", disable=not is_main_process(self.rank)):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = self.model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        # Gather from all GPUs
        if self.world_size > 1:
            preds_tensor = torch.tensor(all_preds).to(self.device)
            labels_tensor = torch.tensor(all_labels).to(self.device)
            
            gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(self.world_size)]
            gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(gathered_preds, preds_tensor)
            dist.all_gather(gathered_labels, labels_tensor)
            
            all_preds = torch.cat(gathered_preds).cpu().numpy()
            all_labels = torch.cat(gathered_labels).cpu().numpy()
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return acc, f1
    
    def train(self, train_loader, val_loader, train_sampler, total_epochs=10):
        """Full training loop"""
        
        if is_main_process(self.rank):
            print(f"\n🚀 Starting distributed training...")
            print("="*60)
        
        best_val_f1 = 0
        
        for epoch in range(total_epochs):
            avg_loss = self.train_epoch(train_loader, epoch, train_sampler)
            val_acc, val_f1 = self.validate(val_loader)
            
            if val_f1 > best_val_f1 and is_main_process(self.rank):
                best_val_f1 = val_f1
                if self.world_size > 1:
                    self.model.module.save_pretrained("/content/models/classifier")
                else:
                    self.model.save_pretrained("/content/models/classifier")
                self.tokenizer.save_pretrained("/content/models/classifier")
                with open("/content/models/classifier/label_map.json", 'w') as f:
                    json.dump(self.label_map, f, indent=2)
                print(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
            
            if is_main_process(self.rank):
                print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
                print("-"*40)
        
        return best_val_f1

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training function"""
    
    DATA_PROCESSED = "/content/data/processed"
    MODELS_DIR = "/content/models/classifier"
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Initialize trainer
    trainer = DistributedFallacyTrainer()
    
    # Find data files
    train_file = f"{DATA_PROCESSED}/train_clean.csv"
    if not os.path.exists(train_file):
        train_file = f"{DATA_PROCESSED}/train.csv"
    
    val_file = f"{DATA_PROCESSED}/val_clean.csv"
    if not os.path.exists(val_file):
        val_file = f"{DATA_PROCESSED}/val.csv"
    
    test_file = f"{DATA_PROCESSED}/test_clean.csv"
    if not os.path.exists(test_file):
        test_file = f"{DATA_PROCESSED}/test.csv"
    
    label_map_path = f"{DATA_PROCESSED}/label_map.json"
    
    # Load data
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = trainer.load_data(
        train_file, val_file, test_file, label_map_path
    )
    
    # Initialize model
    trainer.initialize_model()
    
    # Create dataloaders
    train_loader, val_loader, test_loader, train_sampler = trainer.create_dataloaders(
        train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, batch_size=16
    )
    
    # Setup optimization - FIXED: pass train_labels
    trainer.setup_optimization(train_loader, train_labels, total_epochs=10, lr=2e-5)
    
    # Train
    best_f1 = trainer.train(train_loader, val_loader, train_sampler, total_epochs=10)
    
    if is_main_process(trainer.rank):
        print(f"\n✅ Training complete!")
        print(f"Best Validation F1: {best_f1:.4f}")
        
        # Final test evaluation
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        
        # Load best model
        model = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
        model = model.to(trainer.device)
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(trainer.device)
                attention_mask = batch['attention_mask'].to(trainer.device)
                labels = batch['labels'].cpu().numpy()
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"\n📊 Test Results:")
        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Macro F1: {test_f1:.4f}")
        
        target_names = [trainer.id_to_label[i] for i in range(trainer.num_labels)]
        print("\n📋 Per-class Performance:")
        print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))
    
    cleanup_distributed()

if __name__ == "__main__":
    main()