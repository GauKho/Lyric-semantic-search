from data_loader_multiplerankingloss import (
    load_and_prepare_data, create_data_splits, create_training_pairs, 
    save_examples, save_splits, load_splits, CONFIG
)
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import torch, logging, wandb, hashlib, os, glob, json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DetailedWandBCallback:
    """Enhanced W&B callback for logging detailed training metrics"""
    
    def __init__(self, val_df=None, model=None, log_detailed_metrics=True, embedding_samples=100, patience=5, min_delta=0.001):
        self.best_score = 0
        self.step_count = 0
        self.val_df = val_df
        self.model = model
        self.log_detailed_metrics = log_detailed_metrics
        self.embedding_samples = embedding_samples
        self.metric_history = []

        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_score_for_stopping = -float('inf')
        self.early_stopped = False
        self.early_stop_epoch = None
        self.early_stop_step = None
        
    def __call__(self, score, epoch, steps):
        """Enhanced callback function for comprehensive W&B logging during training"""
        self.step_count += 1

        # Early stopping check
        if score > self.best_score_for_stopping + self.min_delta:
            self.best_score_for_stopping = score
            self.wait = 0
        else:
            self.wait += 1
            
        # Check if we should stop early
        if self.wait >= self.patience and not self.early_stopped:
            self.early_stopped = True
            self.early_stop_epoch = epoch
            self.early_stop_step = steps
            logger.info(f"Early stopping triggered at epoch {epoch}, step {steps}")
            logger.info(f"Best score: {self.best_score_for_stopping:.6f}, Current score: {score:.6f}")
            logger.info(f"No improvement for {self.patience} evaluations")
        
        # Basic training metrics
        basic_metrics = {
            "train/eval_score": score,
            "train/epoch": epoch,
            "train/steps": steps,
            "train/step_count": self.step_count,
            "train/learning_progress": steps / max(steps, 1)  # Progress ratio
        }

        print(f"TRAINING PROGRESS - Step {self.step_count}")
        print(f"Epoch: {epoch}")
        print(f"Steps: {steps}")
        print(f"Current Score: {score:.6f}")
        print(f"Best Score: {self.best_score:.6f}")
        print(f"Early Stopping Counter: {self.wait}/{self.patience}")
        
        # Track best score
        if score > self.best_score:
            self.best_score = score
            basic_metrics["train/best_score"] = self.best_score
            basic_metrics["train/improvement"] = score - self.best_score if self.best_score > 0 else 0
            
        # Log basic metrics first
        wandb.log(basic_metrics, step=self.step_count)

        # Create and log score progression chart
        self._log_score_chart(score, epoch, steps)
        
        # Log detailed metrics if enabled and we have validation data
        if self.log_detailed_metrics and self.val_df is not None and len(self.val_df) > 0:
            try:
                # Get the current model from the training context
                # Note: This is a workaround since we can't directly access the model in callback
                if hasattr(self, '_current_model'):
                    detailed_metrics = self._compute_detailed_metrics(self._current_model)
                    wandb.log(detailed_metrics, step=self.step_count)
                    
            except Exception as e:
                logger.warning(f"Could not log detailed metrics: {e}")
        
        # Store metric history
        self.metric_history.append({
            'step': self.step_count,
            'epoch': epoch,
            'score': score,
            'steps': steps,
            'best_score': self.best_score,
            'patience_counter': self.wait,
            'early_stopped': self.early_stopped
        })
        
        print(f"[W&B LOG] Step {self.step_count} | Epoch {epoch} | Score: {score:.4f} | Best: {self.best_score:.4f} | Patience: {self.wait}/{self.patience}")
        
        # Return early stopping signal (if your training loop supports it)
        return not self.early_stopped
    def _log_score_chart(self, score, epoch, steps):
        """Log score progression chart to W&B"""
        try:
            # Create data for the chart
            chart_data = []
            for i, record in enumerate(self.metric_history):
                chart_data.append([record['epoch'], record['score'], record['step']])
            
            # Add current point
            chart_data.append([epoch, score, steps])
            
            # Create W&B table for the chart
            table = wandb.Table(
                columns=["Epoch", "Score", "Step"],
                data=chart_data
            )
            
            # Log the chart
            wandb.log({
                "charts/score_vs_epoch": wandb.plot.line(
                    table, "Epoch", "Score", 
                    title="Training Score vs Epoch"
                ),
                "charts/score_vs_step": wandb.plot.line(
                    table, "Step", "Score", 
                    title="Training Score vs Step"
                )
            }, step=self.step_count)
            
        except Exception as e:
            logger.warning(f"Could not create score chart: {e}")
    
    def should_stop_early(self):
        """Check if training should stop early"""
        return self.early_stopped
    
    def get_early_stopping_info(self):
        """Get early stopping information"""
        return {
            "early_stopped": self.early_stopped,
            "early_stop_epoch": self.early_stop_epoch,
            "early_stop_step": self.early_stop_step,
            "best_score": self.best_score_for_stopping,
            "patience_used": self.wait
        }
    
    def set_current_model(self, model):
        """Set the current model for detailed evaluation"""
        self._current_model = model
        
    def _compute_detailed_metrics(self, model):
        """Compute detailed metrics for validation set"""
        metrics = {}
        
        try:
            # Sample validation data to avoid memory issues
            sample_size = min(self.embedding_samples, len(self.val_df))
            val_sample = self.val_df.sample(n=sample_size, random_state=42)
            
            # Compute embedding similarities
            similarities = []
            for _, row in val_sample.iterrows():
                try:
                    lyric = row["Lyric"]
                    title = row["Title"]
                    artist = row["Artist"]
                    album = row["Album"] if pd.notna(row["Album"]) else ""
                    
                    # Create metadata string
                    if album and album.lower() != "unknown album":
                        metadata = f"{title} by {artist} from {album}"
                    else:
                        metadata = f"{title} by {artist}"
                    
                    # Compute embeddings
                    emb_lyric = model.encode([lyric])
                    emb_meta = model.encode([metadata])
                    
                    # Compute similarity
                    similarity = cosine_similarity(emb_lyric, emb_meta)[0][0]
                    similarities.append(similarity)
                    
                except Exception as e:
                    continue
            
            if similarities:
                metrics.update({
                    "train/val_similarity_mean": np.mean(similarities),
                    "train/val_similarity_std": np.std(similarities),
                    "train/val_similarity_median": np.median(similarities),
                    "train/val_similarity_min": np.min(similarities),
                    "train/val_similarity_max": np.max(similarities),
                    "train/val_similarity_q25": np.percentile(similarities, 25),
                    "train/val_similarity_q75": np.percentile(similarities, 75)
                })
                
        except Exception as e:
            logger.warning(f"Error computing detailed metrics: {e}")
            
        return metrics
    
    def get_training_summary(self):
        """Get a summary of training progress for final logging"""
        if not self.metric_history:
            return {}
            
        df = pd.DataFrame(self.metric_history)
        return {
            "training_summary/total_steps": len(self.metric_history),
            "training_summary/final_score": df['score'].iloc[-1],
            "training_summary/best_score": df['score'].max(),
            "training_summary/score_improvement": df['score'].iloc[-1] - df['score'].iloc[0],
            "training_summary/epochs_completed": df['epoch'].max(),
        }

class EnhancedInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """Enhanced IR evaluator that logs detailed metrics to W&B"""
    
    def __init__(self, *args, log_to_wandb=True, prefix="", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_to_wandb = log_to_wandb
        self.prefix = prefix
        
    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        """Enhanced evaluation with detailed W&B logging"""
        
        # Get detailed metrics from parent class
        detailed_metrics = super().__call__(model, output_path, epoch, steps)
        
        # Log to W&B if enabled
        if self.log_to_wandb and wandb.run is not None:
            wandb_metrics = {}
            
            if isinstance(detailed_metrics, dict):
                for key, value in detailed_metrics.items():
                    # Add prefix to metric names
                    metric_name = f"{self.prefix}/{key}" if self.prefix else key
                    wandb_metrics[metric_name] = value
                    
                    # Also log with train prefix for tracking during training
                    if self.prefix == "val":
                        wandb_metrics[f"train/{key}"] = value
                        
            else:
                # Single score
                metric_name = f"{self.prefix}/eval_score" if self.prefix else "eval_score"
                wandb_metrics[metric_name] = detailed_metrics
                
                if self.prefix == "val":
                    wandb_metrics["train/eval_score"] = detailed_metrics
            
            # Add epoch and steps info
            if epoch >= 0:
                wandb_metrics[f"{self.prefix}/epoch"] = epoch
            if steps >= 0:
                wandb_metrics[f"{self.prefix}/steps"] = steps
                
            wandb.log(wandb_metrics)
            
            # Print detailed metrics
            if isinstance(detailed_metrics, dict):
                logger.info(f"[{self.prefix.upper()}] Detailed metrics:")
                for key, value in detailed_metrics.items():
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"[{self.prefix.upper()}] Score: {detailed_metrics:.4f}")
        
        return detailed_metrics

def build_enhanced_ir_evaluator(df, split_name="val", max_queries=1500, use_full_data=True, log_to_wandb=True):
    queries = {}
    corpus = {}
    relevant_docs = defaultdict(set)
    
    # Use full dataset or sample
    if use_full_data:
        eval_df = df
        logger.info(f"Using full {split_name} dataset: {len(df)} samples")
    else:
        sample_size = min(max_queries, len(df))
        eval_df = df.sample(n=sample_size)
        logger.info(f"Using {split_name} sample: {len(eval_df)} samples")
    
    for idx, (_, row) in enumerate(eval_df.iterrows()):
        query_id = f"q{idx}"
        doc_id = f"d{idx}"
        
        title = row["Title"].strip()
        album = row["Album"].strip() if pd.notna(row["Album"]) else ""
        artist = row["Artist"].strip()
        
        # Create metadata string as QUERY
        if album and album.lower() != "unknown album":
            metadata_query = f"the song '{title}' by {artist} from the album '{album}'"
        else:
            metadata_query = f"{title} by {artist}"
            
        # Use lyric as DOCUMENT to retrieve
        lyric_document = row["Lyric"].strip()
        
        queries[query_id] = metadata_query  # What user searches
        corpus[doc_id] = lyric_document     # What we retrieve
        relevant_docs[query_id].add(doc_id)
    
    evaluator = EnhancedInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        name=f"lyrics-ir-{split_name}",
        mrr_at_k=[1, 3, 5, 10],
        map_at_k=[1, 3, 5, 10],
        ndcg_at_k=[1, 3, 5, 10],
        precision_recall_at_k=[1, 3, 5, 10],
        log_to_wandb=log_to_wandb,
        prefix=split_name
    )
    
    return evaluator


def evaluate_on_split(model, df, split_name="test", max_samples=3000):
    """Evaluate model on a specific split and log to W&B with detailed metrics"""
    logger.info(f"Evaluating on {split_name} split...")
    
    # Build enhanced evaluator for this split
    evaluator = build_enhanced_ir_evaluator(df, split_name, max_samples)
    
    # Run evaluation
    metrics = evaluator(model)
    
    # Additional custom metrics
    additional_metrics = compute_embedding_similarity_metrics(model, df, split_name, n_samples=min(500, len(df)))
    
    return metrics

def compute_embedding_similarity_metrics(model, df, split_name="test", n_samples=1000, use_full_data=True):
    """Compute embedding similarity metrics for a split with enhanced logging"""
    if use_full_data:
        eval_df = df
        logger.info(f"Using full {split_name} dataset: {len(df)} samples")
    else:
        sample_size = min(1500, max(len(df)))
        eval_df = df.sample(n=sample_size)
        logger.info(f"Using {split_name} sample: {len(eval_df)} samples")
    
    similarities = []
    for _, row in eval_df.iterrows():
        try:
            lyric = row["Lyric"]
            title = row["Title"]
            artist = row["Artist"]
            album = row["Album"] if pd.notna(row["Album"]) else ""
            
            # Create metadata string
            if album and album.lower() != "unknown album":
                metadata = f"{title} by {artist} from {album}"
            else:
                metadata = f"{title} by {artist}"
            
            # Compute embeddings
            emb_lyric = model.encode([lyric])
            emb_meta = model.encode([metadata])
            
            # Compute similarity
            similarity = cosine_similarity(emb_lyric, emb_meta)[0][0]
            similarities.append(similarity)
            
        except Exception as e:
            logger.warning(f"Error computing similarity: {e}")
            continue
    
    if similarities:
        metrics = {
            f"{split_name}/embedding_similarity_mean": np.mean(similarities),
            f"{split_name}/embedding_similarity_std": np.std(similarities),
            f"{split_name}/embedding_similarity_median": np.median(similarities),
            f"{split_name}/embedding_similarity_min": np.min(similarities),
            f"{split_name}/embedding_similarity_max": np.max(similarities),
            f"{split_name}/embedding_similarity_q25": np.percentile(similarities, 25),
            f"{split_name}/embedding_similarity_q75": np.percentile(similarities, 75),
            f"{split_name}/embedding_similarity_samples": len(similarities)
        }
        
        # Log distribution histogram
        if wandb.run is not None:
            wandb.log({
                f"{split_name}/similarity_histogram": wandb.Histogram(similarities),
                **metrics
            })
        
        logger.info(f"[{split_name.upper()}] Embedding similarity metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
    
    return similarities

def train_model(train_examples, val_df, test_df, output_path="lyrics_sbert_model", 
                run_name="lyric-sbert-run", project_name="lyric-semantic-search", early_stopping_patience=5):
    
    # Initialize W&B with streamlined config
    wandb_config = {
        "num_train_samples": len(train_examples),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "early_stopping_patience": early_stopping_patience,
        **{k: v for k, v in CONFIG.items() if k in ["base_model", "epochs", "learning_rate", 
                                                   "batch_size_cpu", "negative_ratio", 
                                                   "warmup_ratio", "weight_decay"]}
    }
    wandb.init(project=project_name, name=run_name, config=wandb_config)
    
    # Setup model and device
    model = SentenceTransformer(CONFIG["base_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model._first_module().auto_model.to("cuda")
    batch_size = CONFIG["batch_size_cuda"] if device.type == "cuda" else CONFIG["batch_size_cpu"]
    
    logger.info(f"Using device: {device}, Batch size: {batch_size}")
    
    # Prepare training components
    dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    loss = losses.MultipleNegativesRankingLoss(model)
    val_evaluator = build_enhanced_ir_evaluator(val_df, "val")
    
    # Initialize custom callback with error handling
    wandb_callback = None
    try:
        wandb_callback = DetailedWandBCallback(val_df, log_detailed_metrics=True, patience=early_stopping_patience)
        if not hasattr(wandb_callback, 'should_save'):
            logger.warning("DetailedWandBCallback missing attributes - using default callback")
            wandb_callback = None
    except Exception as e:
        logger.warning(f"Custom callback failed: {e} - continuing without it")
    
    # Calculate steps and log initial setup
    num_train_steps = len(dataloader) * CONFIG["epochs"]
    warmup_steps = int(CONFIG["warmup_ratio"] * num_train_steps)
    evaluation_steps = CONFIG["evaluation_steps"]
    
    wandb.log({
        "config/training_steps": num_train_steps,
        "config/warmup_steps": warmup_steps,
        "config/device": str(device)
    })
    
    # Log initial performance
    logger.info("Computing initial metrics...")
    initial_scores = {
        "initial/val_score": evaluate_on_split(model, val_df, "val_initial", max_samples=3000),
        "initial/test_score": evaluate_on_split(model, test_df, "test_initial", max_samples=3000)
    }
    wandb.log({k: (v if isinstance(v, (int, float)) else 0) for k, v in initial_scores.items()})
    
    # Training with comprehensive error handling
    logger.info("Starting training...")
    
    fit_kwargs = {
        "train_objectives": [(dataloader, loss)],
        "evaluator": val_evaluator,
        "evaluation_steps": evaluation_steps,
        "epochs": CONFIG["epochs"],
        "warmup_steps": warmup_steps,
        "optimizer_params": {"lr": CONFIG["learning_rate"]},
        "weight_decay": CONFIG["weight_decay"],
        "scheduler": "WarmupLinear",
        "output_path": output_path,
        "show_progress_bar": True,
        "use_amp": True,
        "save_best_model": True
    }
    
    if wandb_callback:
        fit_kwargs["callback"] = wandb_callback
    
    # Execute training with early stopping detection
    try:
        model.fit(**fit_kwargs)
        training_completed = True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Check if early stopping was the cause
        if wandb_callback and hasattr(wandb_callback, 'should_stop_early') and wandb_callback.should_stop_early():
            logger.info("Training stopped due to early stopping")
            training_completed = True
        else:
            raise
    
    # Extract early stopping info safely
    if wandb_callback and hasattr(wandb_callback, 'get_early_stopping_info'):
        early_stop_info = wandb_callback.get_early_stopping_info()
    else:
        early_stop_info = {
            "early_stopped": False,
            "early_stop_epoch": CONFIG["epochs"],
            "early_stop_step": num_train_steps,
            "best_score": 0.0,
            "patience_used": 0
        }
    
    # Log early stopping results
    wandb.log({
        "early_stopping/triggered": early_stop_info["early_stopped"],
        "early_stopping/best_score": early_stop_info["best_score"],
        "early_stopping/patience_used": early_stop_info["patience_used"]
    })
    
    if early_stop_info["early_stopped"]:
        print(f"\nEARLY STOPPING: Epoch {early_stop_info['early_stop_epoch']}, "
              f"Best Score: {early_stop_info['best_score']:.6f}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    
    try:
        final_model = SentenceTransformer(output_path)
    except Exception as e:
        logger.warning(f"Using current model: {e}")
        final_model = model
    
    # Evaluate and log final metrics
    final_metrics = {
        "final/val_score": evaluate_on_split(final_model, val_df, "val_final"),
        "final/test_score": evaluate_on_split(final_model, test_df, "test_final"),
        "training/completed": training_completed,
        "training/early_stopped": early_stop_info["early_stopped"]
    }
    wandb.log(final_metrics)
    
    # Log training progression if available
    if wandb_callback and hasattr(wandb_callback, 'metric_history') and wandb_callback.metric_history:
        try:
            df = pd.DataFrame(wandb_callback.metric_history)
            table = wandb.Table(columns=df.columns.tolist(), data=df.values.tolist())
            wandb.log({
                "charts/training_progress": table,
                "charts/score_progression": wandb.plot.line(table, "step", "score", 
                                                          title="Training Score Progression")
            })
        except Exception as e:
            logger.warning(f"Chart creation failed: {e}")
    
    logger.info("Training completed!")
    return final_model

def hash_file(file_path):
    """Create hash from file content to check for changes"""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_file_hashes(folder):
    """Get hashes for all CSV files in folder"""
    return {
        os.path.basename(f): hash_file(f)
        for f in glob.glob(os.path.join(folder, "*.csv"))
    }

def update_and_train_if_changed(data_folder, hash_path="data_hashes.json", 
                               force_retrain=False):
    """Check for data changes and retrain if necessary"""
    logger.info("Checking for data changes...")
    
    current_hashes = get_file_hashes(data_folder)
    
    if os.path.exists(hash_path) and not force_retrain:
        with open(hash_path, "r") as f:
            old_hashes = json.load(f)
    else:
        old_hashes = {}
    
    # Check if data has changed
    data_changed = (json.dumps(current_hashes, sort_keys=True) != 
                   json.dumps(old_hashes, sort_keys=True))
    
    if data_changed or force_retrain:
        if force_retrain:
            logger.info("Force retraining requested...")
        else:
            logger.info("Data changed. Reloading and retraining...")
        
        # Save current hashes
        with open(hash_path, "w") as f:
            json.dump(current_hashes, f, indent=2)
        
        # Load and prepare data
        logger.info("Loading data...")
        df = load_and_prepare_data(data_folder)
        
        if df.empty:
            logger.error("No data loaded. Exiting.")
            return
        
        # Create proper train/val/test splits
        logger.info("Creating data splits...")
        train_df, val_df, test_df = create_data_splits(df)
        
        # Save splits for inspection
        save_splits(train_df, val_df, test_df)
        
        # Create training pairs
        logger.info("Creating training pairs...")
        train_examples = create_training_pairs(train_df)
        
        if not train_examples:
            logger.error("No training examples created. Exiting.")
            return
        
        # Save training examples
        save_examples(train_examples, "training_data.pkl")
        
        logger.info(f"Total training pairs: {len(train_examples)}")
        logger.info(f"Validation samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Train model with enhanced monitoring
        model = train_model(train_examples, val_df, test_df)
        
        return model
    else:
        logger.info("No data changes detected. Skipping training.")
        return None

def test_model_performance(model, test_df, n_samples=100):
    """Test model performance with proper retrieval evaluation"""
    logger.info(f"Testing model performance with {n_samples} samples...")
    
    sample_df = test_df.sample(n=min(n_samples, len(test_df)), random_state=42)
    
    # Pre-encode all metadata for efficient retrieval
    all_metadata = []
    for _, row in test_df.iterrows():
        title, artist = row["Title"], row["Artist"]
        album = row["Album"] if pd.notna(row["Album"]) and row["Album"].lower() != "unknown album" else ""
        
        if album:
            metadata = f"{title} by {artist} from {album}"
        else:
            metadata = f"{title} by {artist}"
        all_metadata.append(metadata)
    
    logger.info("Encoding all metadata...")
    metadata_embeddings = model.encode(all_metadata, show_progress_bar=True)
    
    # Test retrieval performance
    similarities = []
    rank_positions = []
    
    for idx, (test_idx, row) in enumerate(tqdm(sample_df.iterrows(), desc="Testing retrieval", total=len(sample_df))):
        try:
            lyric = row["Lyric"]
            
            # Find correct metadata index in full dataset
            correct_idx = test_df.index.get_loc(test_idx)
            
            # Encode query lyric
            lyric_embedding = model.encode([lyric])
            
            # Compute similarities with all metadata
            sims = cosine_similarity(lyric_embedding, metadata_embeddings)[0]
            
            # Get ranking
            ranked_indices = np.argsort(sims)[::-1]  # Descending order
            correct_rank = np.where(ranked_indices == correct_idx)[0][0] + 1  # 1-indexed
            
            rank_positions.append(correct_rank)
            similarities.append(sims[correct_idx])
            
            # Progress logging
            if (idx + 1) % 20 == 0:
                current_mrr = np.mean([1.0 / rank for rank in rank_positions])
                print(f"Progress: {idx+1}/{len(sample_df)} | Current MRR: {current_mrr:.4f}")
                
        except Exception as e:
            logger.warning(f"Error in test prediction: {e}")
            continue
    
    # Calculate proper retrieval metrics
    if not rank_positions:
        logger.error("No successful predictions made")
        return 0, 0, 0
    
    mrr = np.mean([1.0 / rank for rank in rank_positions])
    top_1_accuracy = sum(1 for rank in rank_positions if rank == 1) / len(rank_positions)
    top_3_accuracy = sum(1 for rank in rank_positions if rank <= 3) / len(rank_positions)
    top_5_accuracy = sum(1 for rank in rank_positions if rank <= 5) / len(rank_positions)
    top_10_accuracy = sum(1 for rank in rank_positions if rank <= 10) / len(rank_positions)
    avg_similarity = np.mean(similarities)
    median_rank = np.median(rank_positions)
    
    # Comprehensive logging
    logger.info("=== RETRIEVAL TEST RESULTS ===")
    logger.info(f"Dataset size: {len(test_df)} songs")
    logger.info(f"Test samples: {len(rank_positions)}")
    logger.info(f"Top-1 Accuracy:  {top_1_accuracy:.4f}")
    logger.info(f"Top-3 Accuracy:  {top_3_accuracy:.4f}")
    logger.info(f"Top-5 Accuracy:  {top_5_accuracy:.4f}")
    logger.info(f"Top-10 Accuracy: {top_10_accuracy:.4f}")
    logger.info(f"Mean Reciprocal Rank: {mrr:.4f}")
    logger.info(f"Median Rank: {median_rank}")
    logger.info(f"Average Similarity: {avg_similarity:.4f}")

    print("\n" + "="*50)
    print("FINAL RETRIEVAL TEST RESULTS")
    print("="*50)
    print(f"Dataset Size:         {len(test_df):,} songs")
    print(f"Test Samples:         {len(rank_positions):,}")
    print(f"Top-1 Accuracy:       {top_1_accuracy:.6f}")
    print(f"Top-3 Accuracy:       {top_3_accuracy:.6f}")
    print(f"Top-5 Accuracy:       {top_5_accuracy:.6f}")
    print(f"Top-10 Accuracy:      {top_10_accuracy:.6f}")
    print(f"Mean Reciprocal Rank: {mrr:.6f}")
    print(f"Median Rank:          {median_rank:.1f}")
    print(f"Average Similarity:   {avg_similarity:.6f}")
    print("="*50)
    
    # Log to W&B if active
    if wandb.run is not None:
        test_metrics = {
            "test/dataset_size": len(test_df),
            "test/samples_tested": len(rank_positions),
            "test/top1_accuracy": top_1_accuracy,
            "test/top3_accuracy": top_3_accuracy,
            "test/top5_accuracy": top_5_accuracy,
            "test/top10_accuracy": top_10_accuracy,
            "test/mrr": mrr,
            "test/median_rank": median_rank,
            "test/avg_similarity": avg_similarity,
            "test/similarity_dist": wandb.Histogram(similarities),
            "test/rank_dist": wandb.Histogram(rank_positions)
        }
        wandb.log(test_metrics)
        
        # Create rank distribution chart
        rank_df = pd.DataFrame({
            'rank': rank_positions,
            'similarity': similarities
        })
        wandb.log({
            "test/rank_vs_similarity": wandb.plot.scatter(
                wandb.Table(dataframe=rank_df), 
                "rank", "similarity", title="Rank vs Similarity"
            )
        })
    
    return top_1_accuracy, mrr, avg_similarity

def main():

    file_path = "backend\\data\\csv"  # Update path as needed
    
    # Setup enhanced logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('training.log')  # File output
        ]
    )
    
    print(f"\n{'#'*100}")
    print("LYRIC SEMANTIC SEARCH MODEL TRAINING PIPELINE")
    print(f"{'#'*100}")
    print(f"Data folder: {file_path}")
    print(f"Log file: training.log")
    
    # Check if splits already exist
    print("\nChecking for existing data splits...")
    train_df, val_df, test_df = load_splits("./splits")
    
    if train_df is None:
        print("No existing splits found. Creating new splits...")
        model = update_and_train_if_changed(file_path, force_retrain=True)
        
        # Load the newly created splits
        train_df, val_df, test_df = load_splits("./splits")
        if train_df is None:
            print("Failed to create or load splits. Exiting.")
            return
        else:
            print("New splits created and loaded successfully")
    else:
        print("Existing splits found. Checking for data changes...")
        model = update_and_train_if_changed(file_path)
    
    # Test model performance if model exists
    model_path = "lyrics_sbert_model"
    if os.path.exists(model_path) and test_df is not None:
        print(f"\n{'='*80}")
        print("TESTING FINAL MODEL PERFORMANCE")
        print(f"{'='*80}")
        
        try:
            accuracy, similarity, mrr = test_model_performance(model_path, test_df, n_samples=200)
            
            print(f"\nFinal Performance Summary:")
            print(f"Test Accuracy: {accuracy:.6f}")
            print(f"Average Similarity: {similarity:.6f}")
            print(f"Mean Reciprocal Rank: {mrr:.6f}")
            
            # Additional comprehensive evaluation
            print("\nComputing comprehensive final metrics...") 
            model = SentenceTransformer(model_path)
            compute_embedding_similarity_metrics(model, test_df, "test_final_comprehensive")
            
        except Exception as e:
            print(f"Error during model testing: {e}")
            logger.error(f"Error during model testing: {e}")
    else:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
        if test_df is None:
            print("Test dataset not available")
    
    # Close W&B run
    if wandb.run is not None:
        print("\nClosing W&B run...")
        wandb.finish()
    
    print(f"\n{'#'*100}")
    print("ENHANCED PIPELINE COMPLETED!")
    print(f"{'#'*100}")
    
    # Print summary of what was accomplished
    print("\nSummary:")
    if train_df is not None:
        print(f"Data splits available: train({len(train_df)}), val({len(val_df)}), test({len(test_df)})")
    if os.path.exists(model_path):
        print(f"Trained model available at: {model_path}")
    if os.path.exists("training.log"):
        print(f"Training log saved to: training.log")
    
    print("\nCheck your W&B dashboard for detailed training metrics and visualizations!")

if __name__ == "__main__":
    main()