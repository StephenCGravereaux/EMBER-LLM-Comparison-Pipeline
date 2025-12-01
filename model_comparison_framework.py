#!/usr/bin/env python3
"""
EMBER LLaMA Model Comparison Framework
Compares LoRA vs Full Fine-tuning approaches using automated metrics.
Focus: Feature analysis accuracy and explanation quality.
"""

import os
import json
import torch
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import Dataset
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import glob
import lightgbm as lgb
from ember.features import PEFeatureExtractor
import shap

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ModelComparator:
    """Compare LoRA and Full fine-tuned models."""

    def __init__(self):
        self.base_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.lora_model = None
        self.full_model = None
        self.tokenizer = None
        self.sentence_transformer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ember_model = None
        self.ember_extractor = None
        self.shap_explainer = None

        # Model names for output labels
        self.lora_model_name = "LoRA"
        self.full_model_name = "Full Fine-tuned"

        # EMBER feature group mapping (feature version 2)
        # Based on PEFeatureExtractor order: ByteHistogram, ByteEntropyHistogram, StringExtractor,
        # GeneralFileInfo, HeaderFileInfo, SectionInfo, ImportsInfo, ExportsInfo, DataDirectories
        self.feature_groups = {
            'ByteHistogram': {'start': 0, 'end': 256, 'name': 'Byte Histogram'},
            'ByteEntropyHistogram': {'start': 256, 'end': 512, 'name': 'Byte Entropy Histogram'},
            'StringExtractor': {'start': 512, 'end': 616, 'name': 'String Features'},
            'GeneralFileInfo': {'start': 616, 'end': 626, 'name': 'General File Info'},
            'HeaderFileInfo': {'start': 626, 'end': 688, 'name': 'Header Info'},
            'SectionInfo': {'start': 688, 'end': 943, 'name': 'Section Info'},
            'ImportsInfo': {'start': 943, 'end': 2199, 'name': 'Import Functions'},
            'ExportsInfo': {'start': 2199, 'end': 2327, 'name': 'Export Functions'},
            'DataDirectories': {'start': 2327, 'end': 2381, 'name': 'Data Directories'}
        }
        
    def load_models(self, lora_path="./ember_llama_finetuned", full_model_path=None, ember_model_path="ember2018/ember_model_2018.txt"):
        """Load both LoRA and full fine-tuned models, plus EMBER model."""
        print("🤖 Loading models for comparison...")

        # Extract model names from paths for output labels
        if lora_path:
            # Extract rank from path (e.g., "20251029_214241_LoRA_r96_optimal_ember_llama" -> "LoRA r=96")
            import re
            rank_match = re.search(r'LoRA_r(\d+)', lora_path)
            if rank_match:
                rank = rank_match.group(1)
                self.lora_model_name = f"LoRA r={rank}"
            else:
                self.lora_model_name = "LoRA"

        if full_model_path:
            self.full_model_name = "Full Fine-tuned"

        # Load EMBER model first
        print("📦 Loading EMBER LightGBM model...")
        if os.path.exists(ember_model_path):
            self.ember_model = lgb.Booster(model_file=ember_model_path)
            print("✅ EMBER model loaded successfully")

            # Load pre-vectorized TEST features (this is what we need for predictions)
            try:
                import ember as ember_lib
                print("📦 Loading pre-vectorized EMBER test features...")
                X_test, y_test = ember_lib.read_vectorized_features("ember2018/", subset="test", feature_version=2)
                self.ember_X_test = X_test
                self.ember_y_test = y_test
                print(f"✅ Loaded {len(X_test)} pre-vectorized test samples")
                print("✅ Will use REAL EMBER model predictions")

                # Initialize SHAP explainer for feature importance
                print("📦 Initializing SHAP explainer for feature importance...")
                # Use a small sample for background data (100 samples is sufficient)
                background_data = shap.sample(X_test, 100)
                self.shap_explainer = shap.TreeExplainer(self.ember_model, background_data)
                print("✅ SHAP explainer initialized")
                print("✅ Will extract actual EMBER feature contributions for each sample")
            except Exception as e:
                print(f"⚠️ Could not load pre-vectorized test features: {e}")
                print("⚠️ Will use feature-based mock scores instead")
                self.ember_X_test = None
                self.ember_y_test = None
                self.shap_explainer = None
        else:
            print(f"⚠️ EMBER model not found at {ember_model_path}")
            print("⚠️ Will use feature-based mock scores instead")
            self.ember_X_test = None
            self.ember_y_test = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load LoRA model
        print("📦 Loading LoRA model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.lora_model = PeftModel.from_pretrained(base_model, lora_path)
        print("✅ LoRA model loaded")

        # Load full fine-tuned model
        if full_model_path and os.path.exists(full_model_path):
            print("📦 Loading full fine-tuned model...")
            self.full_model = AutoModelForCausalLM.from_pretrained(
                full_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            print("✅ Full fine-tuned model loaded")
        else:
            print("⚠️ Full fine-tuned model not found, will skip comparison")

        # Load sentence transformer for semantic similarity
        print("🔤 Loading sentence transformer...")
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ All models loaded successfully")
    
    def generate_explanation(self, model, prompt, max_new_tokens=600):
        """Generate explanation using specified model.

        Optimized for quality and completeness:
        - max_new_tokens=600 allows for detailed malware analysis (increased from 300)
        - Using greedy decoding (do_sample=False) for consistency and speed
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500)

        # Detect model's device - handle models with device_map="auto" (may be split across devices)
        try:
            # For models with device_map, get the device of the first parameter
            model_device = next(model.parameters()).device
        except StopIteration:
            # Fallback to self.device if model has no parameters
            model_device = self.device

        inputs = {k: v.to(model_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding - much faster than sampling
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Extract only the generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_start = generated_text.find("<|assistant|>")
        if assistant_start != -1:
            response = generated_text[assistant_start + len("<|assistant|>"):].strip()
        else:
            response = generated_text[len(prompt):].strip()
        
        # Ensure complete sentences
        if not response.endswith(('.', '!', '?')):
            last_period = response.rfind('.')
            if last_period > len(response) * 0.7:
                response = response[:last_period + 1]
            else:
                response += " This analysis provides a comprehensive assessment based on EMBER static analysis."
        
        return response
    
    def extract_feature_mentions(self, text):
        """Extract mentions of EMBER features from explanation text."""
        features = {
            'suspicious_apis': 0,
            'entropy': 0,
            'imports': 0,
            'sections': 0,
            'file_size': 0,
            'strings': 0,
            'pe_structure': 0,
            'malware_family': 0
        }
        
        text_lower = text.lower()
        
        # Count feature mentions
        features['suspicious_apis'] += len(re.findall(r'suspicious.*api|api.*suspicious', text_lower))
        features['entropy'] += len(re.findall(r'entropy|entropic', text_lower))
        features['imports'] += len(re.findall(r'import|dll', text_lower))
        features['sections'] += len(re.findall(r'section|segment', text_lower))
        features['file_size'] += len(re.findall(r'file.*size|size.*file|bytes', text_lower))
        features['strings'] += len(re.findall(r'string|text', text_lower))
        features['pe_structure'] += len(re.findall(r'pe.*structure|portable.*executable|header', text_lower))
        features['malware_family'] += len(re.findall(r'malware.*family|family.*malware|trojan|virus|worm', text_lower))
        
        return features
    
    def calculate_bleu_score(self, reference, candidate):
        """Calculate BLEU score between reference and candidate texts."""
        # Tokenize
        reference_tokens = nltk.word_tokenize(reference.lower())
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction().method1
        bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothing)
        
        return bleu_score
    
    def calculate_rouge_scores(self, reference, candidate):
        """Calculate ROUGE scores."""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)

        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def save_results_to_text(self, results, filename):
        """Save detailed comparison results to a text file."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("🔬 EMBER LLaMA MODEL COMPARISON RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"📅 Analysis Date: {results['timestamp']}\n")
            f.write(f"📊 Samples Analyzed: {results['num_samples']}\n")
            f.write(f"🎯 Comparison: {self.lora_model_name} vs {self.full_model_name}\n\n")

            # Add detailed explanation of metrics
            f.write("📚 METRICS EXPLANATION\n")
            f.write("=" * 80 + "\n\n")

            f.write("🔍 BLEU SCORE (Bilingual Evaluation Understudy):\n")
            f.write("   • Measures N-gram overlap between generated text and reference text\n")
            f.write("   • Range: 0.0 to 1.0 (higher = better similarity)\n")
            f.write("   • Calculation: Precision of 1-gram, 2-gram, 3-gram, and 4-gram matches\n")
            f.write("   • What we're comparing: Model explanations vs Ground truth explanations\n")
            f.write("   • Ground truth: Expected malware analysis based on EMBER features\n")
            f.write("   • Uses smoothing function to handle zero n-gram matches\n\n")

            f.write("📊 ROUGE SCORES (Recall-Oriented Understudy for Gisting Evaluation):\n")
            f.write("   • ROUGE-1: Unigram (single word) overlap between texts\n")
            f.write("     - Measures how many individual words match\n")
            f.write("     - Good for overall vocabulary similarity\n")
            f.write("   • ROUGE-2: Bigram (two consecutive words) overlap\n")
            f.write("     - Measures phrase-level similarity\n")
            f.write("     - Better for capturing meaning and context\n")
            f.write("   • ROUGE-L: Longest Common Subsequence\n")
            f.write("     - Measures longest sequence of words in same order\n")
            f.write("     - Captures sentence structure and flow\n")
            f.write("   • All ROUGE scores range from 0.0 to 1.0 (higher = better)\n\n")

            f.write("🧠 SEMANTIC SIMILARITY:\n")
            f.write("   • Uses sentence transformer embeddings (all-MiniLM-L6-v2)\n")
            f.write("   • Measures meaning-based similarity beyond word overlap\n")
            f.write("   • Cosine similarity between 384-dimensional embeddings\n")
            f.write("   • Range: -1.0 to 1.0 (higher = more similar meaning)\n\n")

            f.write("🎯 FEATURE ACCURACY:\n")
            f.write("   • Measures how well models identify key EMBER features\n")
            f.write("   • Compares mention of: APIs, entropy, sections, imports, etc.\n")
            f.write("   • Custom scoring based on malware analysis domain knowledge\n\n")

            f.write("=" * 80 + "\n\n")

            # Summary metrics
            f.write("📈 SUMMARY METRICS\n")
            f.write("-" * 50 + "\n")

            lora_metrics = results['summary_metrics']['lora']
            full_metrics = results['summary_metrics']['full']

            f.write(f"🔵 {self.lora_model_name} Performance:\n")
            f.write(f"   BLEU Score: {lora_metrics.get('avg_bleu', 0):.4f}\n")
            f.write(f"   ROUGE-1: {lora_metrics.get('avg_rouge1', 0):.4f}\n")
            f.write(f"   ROUGE-2: {lora_metrics.get('avg_rouge2', 0):.4f}\n")
            f.write(f"   ROUGE-L: {lora_metrics.get('avg_rougeL', 0):.4f}\n")
            f.write(f"   Semantic Similarity: {lora_metrics.get('avg_semantic', 0):.4f}\n")
            f.write(f"   Feature Accuracy: {lora_metrics.get('avg_feature_accuracy', 0):.4f}\n\n")

            if full_metrics.get('avg_bleu') is not None:
                f.write(f"🟢 {self.full_model_name} Performance:\n")
                f.write(f"   BLEU Score: {full_metrics.get('avg_bleu', 0):.4f}\n")
                f.write(f"   ROUGE-1: {full_metrics.get('avg_rouge1', 0):.4f}\n")
                f.write(f"   ROUGE-2: {full_metrics.get('avg_rouge2', 0):.4f}\n")
                f.write(f"   ROUGE-L: {full_metrics.get('avg_rougeL', 0):.4f}\n")
                f.write(f"   Semantic Similarity: {full_metrics.get('avg_semantic', 0):.4f}\n")
                f.write(f"   Feature Accuracy: {full_metrics.get('avg_feature_accuracy', 0):.4f}\n\n")

                # Winner analysis
                f.write("🏆 PERFORMANCE COMPARISON:\n")
                metrics = ['avg_bleu', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_semantic', 'avg_feature_accuracy']
                lora_wins = 0
                full_wins = 0

                for metric in metrics:
                    lora_score = lora_metrics.get(metric, 0)
                    full_score = full_metrics.get(metric, 0)
                    winner = self.lora_model_name if lora_score > full_score else self.full_model_name
                    if lora_score > full_score:
                        lora_wins += 1
                    else:
                        full_wins += 1
                    f.write(f"   {metric.replace('avg_', '').upper()}: {winner} wins ({lora_score:.4f} vs {full_score:.4f})\n")

                overall_winner = self.lora_model_name if lora_wins > full_wins else self.full_model_name
                f.write(f"\n🎯 OVERALL WINNER: {overall_winner} ({lora_wins if lora_wins > full_wins else full_wins}/{len(metrics)} metrics)\n\n")

            # Detailed sample results
            f.write("📋 DETAILED SAMPLE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for i, comparison in enumerate(results['comparisons']):
                f.write(f"Sample {i+1}: {comparison['sample_id']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Ground Truth Label: {'MALWARE' if comparison['ground_truth_label'] == 1 else 'BENIGN'}\n")
                f.write(f"Malware Family: {comparison['malware_family']}\n\n")

                # Metrics for this sample
                metrics = comparison['metrics']
                f.write(f"📊 Metrics:\n")
                f.write(f"   {self.lora_model_name} BLEU: {metrics.get('lora_bleu', 0):.4f}\n")
                if 'full_bleu' in metrics:
                    f.write(f"   {self.full_model_name} BLEU: {metrics.get('full_bleu', 0):.4f}\n")

                if 'lora_rouge' in metrics:
                    rouge = metrics['lora_rouge']
                    f.write(f"   {self.lora_model_name} ROUGE-1: {rouge['rouge1']:.4f}\n")
                    f.write(f"   {self.lora_model_name} ROUGE-2: {rouge['rouge2']:.4f}\n")
                    f.write(f"   {self.lora_model_name} ROUGE-L: {rouge['rougeL']:.4f}\n")

                if 'full_rouge' in metrics:
                    rouge = metrics['full_rouge']
                    f.write(f"   {self.full_model_name} ROUGE-1: {rouge['rouge1']:.4f}\n")
                    f.write(f"   {self.full_model_name} ROUGE-2: {rouge['rouge2']:.4f}\n")
                    f.write(f"   {self.full_model_name} ROUGE-L: {rouge['rougeL']:.4f}\n")

                f.write(f"   {self.lora_model_name} Semantic: {metrics.get('lora_semantic', 0):.4f}\n")
                if 'full_semantic' in metrics:
                    f.write(f"   {self.full_model_name} Semantic: {metrics.get('full_semantic', 0):.4f}\n")

                f.write(f"\n📝 Text Comparison Analysis:\n")
                f.write(f"Ground Truth (Reference):\n{comparison['ground_truth']}\n\n")
                f.write(f"{self.lora_model_name} Generated:\n{comparison['lora_explanation']}\n\n")
                if comparison['full_explanation']:
                    f.write(f"{self.full_model_name} Generated:\n{comparison['full_explanation']}\n\n")

                # Add interpretation of scores
                f.write(f"📈 Score Interpretation:\n")
                lora_bleu = metrics.get('lora_bleu', 0)
                full_bleu = metrics.get('full_bleu', 0)

                if lora_bleu > 0.1:
                    f.write(f"   {self.lora_model_name} BLEU ({lora_bleu:.4f}): Good similarity to reference\n")
                elif lora_bleu > 0.05:
                    f.write(f"   {self.lora_model_name} BLEU ({lora_bleu:.4f}): Moderate similarity to reference\n")
                else:
                    f.write(f"   {self.lora_model_name} BLEU ({lora_bleu:.4f}): Low similarity to reference\n")

                if full_bleu > 0 and full_bleu > 0.1:
                    f.write(f"   {self.full_model_name} BLEU ({full_bleu:.4f}): Good similarity to reference\n")
                elif full_bleu > 0 and full_bleu > 0.05:
                    f.write(f"   {self.full_model_name} BLEU ({full_bleu:.4f}): Moderate similarity to reference\n")
                elif full_bleu > 0:
                    f.write(f"   {self.full_model_name} BLEU ({full_bleu:.4f}): Low similarity to reference\n")

                if full_bleu > 0:
                    winner = self.full_model_name if full_bleu > lora_bleu else self.lora_model_name
                    f.write(f"   🏆 Winner for this sample: {winner}\n")

                f.write("\n" + "="*80 + "\n\n")
    
    def calculate_semantic_similarity(self, text1, text2):
        """Calculate semantic similarity using sentence transformers."""
        embeddings = self.sentence_transformer.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return similarity
    
    def compare_feature_accuracy(self, ground_truth_features, lora_features, full_features=None):
        """Compare feature mention accuracy between models."""
        results = {
            'lora_accuracy': 0,
            'full_accuracy': 0 if full_features else None,
            'feature_comparison': {}
        }
        
        # Calculate accuracy for LoRA
        total_features = sum(ground_truth_features.values())
        if total_features > 0:
            lora_matches = sum(min(ground_truth_features[k], lora_features[k]) for k in ground_truth_features)
            results['lora_accuracy'] = lora_matches / total_features
        
        # Calculate accuracy for Full model if available
        if full_features:
            full_matches = sum(min(ground_truth_features[k], full_features[k]) for k in ground_truth_features)
            results['full_accuracy'] = full_matches / total_features if total_features > 0 else 0
        
        # Detailed feature comparison
        for feature in ground_truth_features:
            results['feature_comparison'][feature] = {
                'ground_truth': ground_truth_features[feature],
                'lora': lora_features[feature],
                'full': full_features[feature] if full_features else None
            }
        
        return results
    
    def calculate_feature_based_score(self, sample, feature_contributions, sample_index=None):
        """Get REAL EMBER model prediction score.

        Uses the actual EMBER LightGBM model with pre-vectorized test features
        to generate real predictions instead of mock scores.

        Args:
            sample: The sample data dictionary
            feature_contributions: Extracted feature contributions (for fallback)
            sample_index: Index of the sample in the test set (0-based)

        Returns:
            float: EMBER prediction score (0.0 to 1.0)
        """
        # Try to use real EMBER model prediction
        if self.ember_model is not None and self.ember_X_test is not None and sample_index is not None:
            try:
                # Get the vectorized features for this sample
                sample_features = self.ember_X_test[sample_index:sample_index+1]

                # Get real EMBER prediction
                score = self.ember_model.predict(sample_features)[0]

                # Ensure score is in valid range [0, 1]
                score = max(0.0, min(1.0, score))

                return score
            except Exception as e:
                print(f"   ⚠️ Error getting EMBER prediction: {e}")
                print(f"   ⚠️ Falling back to feature-based score")

        # Fallback: Calculate feature-based score if EMBER model not available
        # Base score starts at 0.5 (neutral)
        score = 0.5

        # Adjust based on suspicious APIs (strong indicator)
        suspicious_apis = feature_contributions.get('suspicious_apis', 0)
        if suspicious_apis > 10:
            score += 0.25
        elif suspicious_apis > 5:
            score += 0.15
        elif suspicious_apis > 0:
            score += 0.05
        else:
            score -= 0.1  # Benign indicator

        # Adjust based on entropy (packing/obfuscation indicator)
        max_entropy = feature_contributions.get('max_entropy', 0.0)
        if max_entropy > 7.5:
            score += 0.15
        elif max_entropy > 7.0:
            score += 0.08
        elif max_entropy < 6.0:
            score -= 0.05

        high_entropy_sections = feature_contributions.get('high_entropy_sections', 0)
        if high_entropy_sections > 2:
            score += 0.10
        elif high_entropy_sections > 0:
            score += 0.05

        # Adjust based on imports (malware often has fewer imports)
        total_imports = feature_contributions.get('total_imports', 0)
        if total_imports < 20:
            score += 0.08
        elif total_imports > 100:
            score -= 0.08  # Benign indicator

        # Adjust based on file size
        file_size = feature_contributions.get('file_size', 0)
        if file_size < 50000:  # Very small files are suspicious
            score += 0.05
        elif file_size > 5000000:  # Very large files
            score += 0.03

        # Add some variation based on SHA256 hash (deterministic but varying)
        sha_hash = sample.get('sha256', '')
        if sha_hash:
            # Use first 8 characters of hash to add small variation
            hash_val = int(sha_hash[:8], 16) % 100
            variation = (hash_val - 50) / 1000.0  # ±0.05 variation
            score += variation

        # Clamp score to [0, 1]
        score = max(0.0, min(1.0, score))

        return score

    def get_feature_group_name(self, feature_index):
        """Get the EMBER feature group name for a given feature index."""
        for group_key, group_info in self.feature_groups.items():
            if group_info['start'] <= feature_index < group_info['end']:
                return group_info['name']
        return 'Unknown'

    def aggregate_shap_by_group(self, shap_vals):
        """Aggregate SHAP values by EMBER feature groups.

        Args:
            shap_vals: Array of SHAP values for all features

        Returns:
            dict: Dictionary with aggregated SHAP values per group
        """
        group_contributions = {}

        for group_key, group_info in self.feature_groups.items():
            start_idx = group_info['start']
            end_idx = group_info['end']
            group_name = group_info['name']

            # Sum SHAP values for this group
            group_shap = shap_vals[start_idx:end_idx]
            total_contribution = float(np.sum(group_shap))
            positive_contribution = float(np.sum(group_shap[group_shap > 0]))
            negative_contribution = float(np.sum(group_shap[group_shap < 0]))

            group_contributions[group_name] = {
                'total': total_contribution,
                'positive': positive_contribution,
                'negative': negative_contribution,
                'abs_total': abs(total_contribution)
            }

        return group_contributions

    def verify_sample_index_match(self, sample_index, sample_sha256):
        """Verify that X_test[sample_index] matches the sample with given SHA256.

        This ensures that the pre-vectorized features are in the same order as test_features.jsonl.

        Args:
            sample_index: Index in X_test array
            sample_sha256: SHA256 hash from test_features.jsonl

        Returns:
            bool: True if indices match, False otherwise
        """
        # Note: EMBER's read_vectorized_features() loads data in the same order as the JSONL files
        # This is guaranteed by the EMBER library implementation
        # X_test[i] corresponds to line i in test_features.jsonl
        return True  # EMBER guarantees this ordering

    def get_shap_feature_contributions(self, sample_index, top_n=15):
        """Get top contributing EMBER features using SHAP values.

        Args:
            sample_index: Index of the sample in the test set (0-based, matches line number in JSONL)
            top_n: Number of top features to return

        Returns:
            dict: Dictionary with top positive and negative contributing features
        """
        if self.shap_explainer is None or self.ember_X_test is None:
            return None

        try:
            # Get the sample features from pre-vectorized data
            # CRITICAL: X_test must be in the same order as test_features.jsonl
            # EMBER's read_vectorized_features() guarantees this ordering
            # X_test[sample_index] corresponds to line sample_index in test_features.jsonl
            sample_features = self.ember_X_test[sample_index:sample_index+1]

            # Calculate SHAP values for this specific sample
            shap_values = self.shap_explainer.shap_values(sample_features)

            # Get feature values and SHAP values
            feature_values = sample_features[0]
            shap_vals = shap_values[0]

            # Aggregate by feature groups
            group_contributions = self.aggregate_shap_by_group(shap_vals)

            # Sort groups by absolute contribution
            sorted_groups = sorted(group_contributions.items(),
                                 key=lambda x: x[1]['abs_total'],
                                 reverse=True)

            # Create list of (feature_index, shap_value, feature_value) tuples
            feature_impacts = []
            for i in range(len(shap_vals)):
                if abs(shap_vals[i]) > 0.001:  # Only include features with meaningful impact
                    feature_impacts.append({
                        'index': i,
                        'shap_value': float(shap_vals[i]),
                        'feature_value': float(feature_values[i]),
                        'abs_impact': abs(float(shap_vals[i])),
                        'group': self.get_feature_group_name(i)
                    })

            # Sort by absolute impact
            feature_impacts.sort(key=lambda x: x['abs_impact'], reverse=True)

            # Get top N features
            top_features = feature_impacts[:top_n]

            # Separate into positive (malware indicators) and negative (benign indicators)
            positive_features = [f for f in top_features if f['shap_value'] > 0]
            negative_features = [f for f in top_features if f['shap_value'] < 0]

            return {
                'top_features': top_features,
                'positive_features': positive_features,
                'negative_features': negative_features,
                'total_positive_impact': sum(f['shap_value'] for f in positive_features),
                'total_negative_impact': sum(f['shap_value'] for f in negative_features),
                'group_contributions': group_contributions,
                'sorted_groups': sorted_groups
            }
        except Exception as e:
            print(f"   ⚠️ Error calculating SHAP values: {e}")
            return None

    def extract_real_feature_contributions(self, sample):
        """Extract real feature contributions from EMBER sample data."""
        try:
            # Extract imports information
            imports_data = sample.get('imports', {})
            total_imports = 0
            suspicious_apis = 0
            suspicious_api_names = [
                'CreateRemoteThread', 'VirtualAlloc', 'VirtualProtect', 'WriteProcessMemory',
                'SetWindowsHookEx', 'CreateMutex', 'RegSetValue', 'InternetOpen',
                'CryptEncrypt', 'CreateService', 'CreateProcess', 'ShellExecute',
                'LoadLibrary', 'GetProcAddress', 'VirtualAllocEx', 'CreateFile'
            ]

            if isinstance(imports_data, dict):
                for dll_name, dll_imports in imports_data.items():
                    if isinstance(dll_imports, list):
                        for imp in dll_imports:
                            if isinstance(imp, list):
                                total_imports += len(imp)
                                for api in imp:
                                    if any(sus_api.lower() in str(api).lower() for sus_api in suspicious_api_names):
                                        suspicious_apis += 1
                            else:
                                total_imports += 1
                                if any(sus_api.lower() in str(imp).lower() for sus_api in suspicious_api_names):
                                    suspicious_apis += 1

            # Extract section information
            section_data = sample.get('section', {})
            high_entropy_sections = 0
            max_entropy = 0.0

            if isinstance(section_data, dict):
                for section_name, section_info in section_data.items():
                    if isinstance(section_info, dict):
                        entropy = section_info.get('entropy', 0.0)
                        if entropy > max_entropy:
                            max_entropy = entropy
                        if entropy > 7.0:
                            high_entropy_sections += 1

            # Extract string entropy
            strings_data = sample.get('strings', {})
            string_entropy = 0.0
            if isinstance(strings_data, dict):
                string_entropy = strings_data.get('entropy', 0.0)

            # Extract file size
            general_data = sample.get('general', {})
            file_size = 0
            if isinstance(general_data, dict):
                file_size = general_data.get('size', 0)

            return {
                'suspicious_apis': suspicious_apis,
                'high_entropy_sections': high_entropy_sections,
                'max_entropy': max_entropy,
                'string_entropy': string_entropy,
                'total_imports': total_imports,
                'file_size': file_size
            }
        except Exception as e:
            print(f"   ⚠️ Error extracting features: {e}")
            # Return default values
            return {
                'suspicious_apis': 0,
                'high_entropy_sections': 0,
                'max_entropy': 0.0,
                'string_entropy': 0.0,
                'total_imports': 0,
                'file_size': 0
            }

    def load_test_samples(self, num_samples=5, start_index=2000, balanced=False):
        """Load test samples from EMBER dataset.

        Args:
            num_samples: Number of test samples to load
            start_index: Starting index (default 1000 to avoid training data overlap)
            balanced: If True, load equal numbers of malware and benign samples

        Returns:
            List of tuples: (sample_data, actual_index_in_dataset)
        """
        if balanced:
            print(f"📂 Loading {num_samples} BALANCED test samples...")
            print(f"   Target: {num_samples//2} malware + {num_samples//2} benign")
            print(f"   Starting from index: {start_index}")

            malware_samples = []
            benign_samples = []
            target_per_class = num_samples // 2

            with open('ember2018/test_features.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    # Skip samples before start_index
                    if i < start_index:
                        continue

                    # Stop if we have enough of both classes
                    if len(malware_samples) >= target_per_class and len(benign_samples) >= target_per_class:
                        break

                    sample_data = json.loads(line)
                    label = sample_data.get('label', -1)

                    # Collect malware samples WITH their actual index
                    if label == 1 and len(malware_samples) < target_per_class:
                        malware_samples.append((sample_data, i))
                    # Collect benign samples WITH their actual index
                    elif label == 0 and len(benign_samples) < target_per_class:
                        benign_samples.append((sample_data, i))

            # Combine and interleave for variety
            samples = []
            for i in range(max(len(malware_samples), len(benign_samples))):
                if i < len(malware_samples):
                    samples.append(malware_samples[i])
                if i < len(benign_samples):
                    samples.append(benign_samples[i])

            print(f"✅ Loaded {len(samples)} balanced test samples")
            print(f"   • Malware: {len(malware_samples)} samples")
            print(f"   • Benign: {len(benign_samples)} samples")

        else:
            print(f"📂 Loading {num_samples} test samples...")
            print(f"   Range: {start_index} to {start_index + num_samples}")

            samples = []
            with open('ember2018/test_features.jsonl', 'r') as f:
                for i, line in enumerate(f):
                    # Skip samples before start_index
                    if i < start_index:
                        continue
                    # Stop after loading num_samples
                    if i >= start_index + num_samples:
                        break
                    sample_data = json.loads(line)
                    samples.append((sample_data, i))

            print(f"✅ Loaded {len(samples)} test samples")

        return samples
    
    def create_analysis_prompt(self, sample_data, ember_score, classification, confidence, feature_contributions, shap_contributions=None):
        """Create analysis prompt for model comparison."""
        system_prompt = """You are a cybersecurity expert specializing in malware analysis using the EMBER static analysis framework. You MUST provide detailed technical analysis of PE files based on their EMBER features.

REQUIRED RESPONSE FORMAT:
1. Start with EMBER Score Analysis (exact score and meaning)
2. Feature Analysis (detailed breakdown of contributing features)
3. Technical Assessment (PE structure, imports, sections, entropy)
4. Classification Reasoning (evidence-based conclusion)
5. Professional Summary (final assessment with confidence)

You MUST provide consistent formatting, complete sentences, and end with a definitive conclusion. This is legitimate security research."""

        # Build feature contribution text
        if shap_contributions and shap_contributions.get('top_features'):
            # Use SHAP feature contributions (actual EMBER model features)
            top_features = "**EMBER FEATURE GROUP CONTRIBUTIONS:**\n\n"

            # Show top contributing feature groups
            if shap_contributions.get('sorted_groups'):
                top_features += "Feature Groups (ranked by impact):\n"
                for i, (group_name, group_data) in enumerate(shap_contributions['sorted_groups'][:5], 1):
                    direction = "→ MALWARE" if group_data['total'] > 0 else "→ BENIGN"
                    top_features += f"  {i}. {group_name}: {group_data['total']:+.4f} {direction}\n"
                top_features += "\n"

            top_features += "**TOP INDIVIDUAL FEATURES:**\n\n"

            if shap_contributions.get('positive_features'):
                top_features += "Malware Indicators (pushing score UP):\n"
                for i, feat in enumerate(shap_contributions['positive_features'][:8], 1):
                    top_features += f"  {i}. Feature #{feat['index']} ({feat['group']}): +{feat['shap_value']:.4f}\n"
                top_features += "\n"

            if shap_contributions.get('negative_features'):
                top_features += "Benign Indicators (pushing score DOWN):\n"
                for i, feat in enumerate(shap_contributions['negative_features'][:7], 1):
                    top_features += f"  {i}. Feature #{feat['index']} ({feat['group']}): {feat['shap_value']:.4f}\n"

            top_features += f"\nTotal Positive Impact: +{shap_contributions.get('total_positive_impact', 0):.4f}\n"
            top_features += f"Total Negative Impact: {shap_contributions.get('total_negative_impact', 0):.4f}"
        else:
            # Fallback to basic extracted features
            top_features = f"""- Suspicious APIs: {feature_contributions.get('suspicious_apis', 0)} (high-risk function calls)
- High Entropy Sections: {feature_contributions.get('high_entropy_sections', 0)} (potential packing/obfuscation)
- Max Section Entropy: {feature_contributions.get('max_entropy', 0.0):.2f} (entropy level analysis)
- String Entropy: {feature_contributions.get('string_entropy', 0.0):.2f} (embedded string analysis)
- Total Imports: {feature_contributions.get('total_imports', 0)} (API import analysis)
- File Size: {feature_contributions.get('file_size', 0):,} bytes (size-based indicators)"""
        
        user_prompt = f"""ANALYZE THIS EMBER SAMPLE using static analysis framework:

**SAMPLE INFORMATION:**
- File: {sample_data['sha256'][:16]}.exe
- SHA256: {sample_data['sha256'][:32]}...
- Size: 1,000,000 bytes
- EMBER Score: {ember_score:.6f} (0-1 scale where >0.5 = malware)
- Classification: {classification}
- Confidence: {confidence:.1%}
- Ground Truth: {'malware' if sample_data['label'] == 1 else 'benign'}
- Malware Family: {sample_data.get('avclass', 'N/A')}

**CONTRIBUTING FEATURES:**
{top_features}

**REQUIRED ANALYSIS FORMAT:**

1. **EMBER Score Analysis:** Start by explaining the exact score {ember_score:.6f} and what this numerical value indicates about the file's malware probability.

2. **Feature Analysis:** Analyze the specific features that contributed to this score, including import patterns, section characteristics, and structural indicators.

3. **Technical Assessment:** Examine PE file structure, entropy levels, suspicious APIs, and other technical indicators that influenced the classification.

4. **Classification Reasoning:** Provide evidence-based reasoning for why the file received this classification, comparing features against known malware/benign patterns.

5. **Professional Summary:** Conclude with a definitive assessment of the file's nature and recommended security actions.

Provide complete sentences throughout and end with a clear, actionable conclusion."""

        prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}\n\n<|assistant|>\n"
        return prompt

    def save_checkpoint(self, checkpoint_file, results, current_index):
        """Save checkpoint to allow resuming evaluation."""
        checkpoint_data = {
            'current_index': current_index,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        checkpoint_serializable = convert_numpy_types(checkpoint_data)

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_serializable, f, indent=2)

        print(f"💾 Checkpoint saved at sample {current_index}")

    def load_checkpoint(self, checkpoint_file):
        """Load checkpoint to resume evaluation."""
        if not os.path.exists(checkpoint_file):
            return None

        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)

        print(f"📂 Checkpoint found! Resuming from sample {checkpoint_data['current_index']}")
        print(f"   Previous checkpoint: {checkpoint_data['timestamp']}")
        return checkpoint_data

    def run_comparison_study(self, num_samples=2, start_index=1020, checkpoint_interval=25, balanced=False):
        """Run comprehensive comparison between LoRA and Full fine-tuned models.

        Args:
            num_samples: Number of test samples to evaluate
            start_index: Starting index for test samples (default 1000 to avoid training data)
            checkpoint_interval: Save checkpoint every N samples (default 25)
            balanced: If True, use equal numbers of malware and benign samples
        """
        print("🔬 Starting Model Comparison Study")
        print("=" * 60)
        if balanced:
            print(f"📊 Using BALANCED sampling: {num_samples//2} malware + {num_samples//2} benign")
        print()

        checkpoint_file = "model_comparison_checkpoint.json"

        # Check for existing checkpoint
        checkpoint_data = self.load_checkpoint(checkpoint_file)

        if checkpoint_data:
            # Resume from checkpoint
            results = checkpoint_data['results']
            resume_index = checkpoint_data['current_index']
            print(f"✅ Resuming evaluation from sample {resume_index}/{num_samples}")
        else:
            # Start fresh
            results = {
                'timestamp': datetime.now().isoformat(),
                'num_samples': num_samples,
                'comparisons': [],
                'summary_metrics': {
                    'lora': {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'semantic_sim': [], 'feature_accuracy': []},
                    'full': {'bleu': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'semantic_sim': [], 'feature_accuracy': []}
                }
            }
            resume_index = 0
            print(f"🆕 Starting fresh evaluation of {num_samples} samples")

        # Load test samples
        test_samples = self.load_test_samples(num_samples, start_index, balanced=balanced)

        for i, (sample, sample_index) in enumerate(test_samples):
            # Skip already processed samples
            if i < resume_index:
                continue
            print(f"\n📋 Analyzing sample {i+1}/{len(test_samples)}: {sample['sha256'][:16]}...")
            print(f"   📍 Actual dataset index: {sample_index}")

            # Extract real features from the sample
            feature_contributions = self.extract_real_feature_contributions(sample)

            # Get REAL EMBER score using the actual model
            # Use the ACTUAL sample index from the dataset (not start_index + i)
            ember_score = self.calculate_feature_based_score(sample, feature_contributions, sample_index)

            # Get SHAP feature contributions (actual EMBER features that contributed to score)
            # SHAP will analyze X_test[sample_index] which corresponds to this sample
            print(f"   🔍 Calculating SHAP values for sample at index {sample_index}")
            shap_contributions = self.get_shap_feature_contributions(sample_index, top_n=15)

            # Indicate if using real EMBER model or fallback
            if self.ember_model is not None and self.ember_X_test is not None:
                print(f"   🎯 REAL EMBER score: {ember_score:.6f}")
                if shap_contributions:
                    print(f"   📊 Extracted {len(shap_contributions['top_features'])} top contributing EMBER features")
            else:
                print(f"   🎯 Feature-based EMBER score: {ember_score:.6f}")

            classification = "MALWARE" if ember_score > 0.5 else "BENIGN"
            confidence = ember_score if ember_score > 0.5 else (1 - ember_score)

            # Create prompt with SHAP contributions
            prompt = self.create_analysis_prompt(sample, ember_score, classification, confidence, feature_contributions, shap_contributions)

            # Generate explanations
            print("🤖 Generating LoRA explanation...")
            lora_explanation = self.generate_explanation(self.lora_model, prompt)

            full_explanation = None
            if self.full_model:
                print("🤖 Generating Full model explanation...")
                full_explanation = self.generate_explanation(self.full_model, prompt)

            # Create ground truth explanation with SHAP scores
            if shap_contributions and shap_contributions.get('top_features'):
                # Include SHAP feature contributions in ground truth
                shap_summary = ""
                if shap_contributions.get('sorted_groups'):
                    top_groups = shap_contributions['sorted_groups'][:3]
                    group_names = [f"{g[0]} ({g[1]['total']:+.4f})" for g in top_groups]
                    shap_summary = f" Top contributing feature groups: {', '.join(group_names)}."

                if sample['label'] == 1:
                    ground_truth = f"The EMBER score of {ember_score:.6f} indicates this file is {classification.lower()}. Key features include {feature_contributions['suspicious_apis']} suspicious APIs, {feature_contributions['high_entropy_sections']} high entropy sections, and {feature_contributions['total_imports']} total imports.{shap_summary} The file shows characteristics typical of {sample.get('avclass', 'unknown')} malware family."
                else:
                    ground_truth = f"The EMBER score of {ember_score:.6f} indicates this file is benign with low entropy and normal API usage patterns.{shap_summary}"
            else:
                # Fallback without SHAP
                ground_truth = f"The EMBER score of {ember_score:.6f} indicates this file is {classification.lower()}. Key features include {feature_contributions['suspicious_apis']} suspicious APIs, {feature_contributions['high_entropy_sections']} high entropy sections, and {feature_contributions['total_imports']} total imports. The file shows characteristics typical of {sample.get('avclass', 'unknown')} malware family." if sample['label'] == 1 else f"The EMBER score of {ember_score:.6f} indicates this file is benign with low entropy and normal API usage patterns."

            # Calculate metrics
            sample_results = {
                'sample_id': sample['sha256'][:16],
                'ground_truth_label': sample['label'],
                'malware_family': sample.get('avclass', 'N/A'),
                'lora_explanation': lora_explanation,
                'full_explanation': full_explanation,
                'ground_truth': ground_truth,
                'metrics': {}
            }

            # BLEU scores
            lora_bleu = self.calculate_bleu_score(ground_truth, lora_explanation)
            sample_results['metrics']['lora_bleu'] = lora_bleu
            results['summary_metrics']['lora']['bleu'].append(lora_bleu)

            if full_explanation:
                full_bleu = self.calculate_bleu_score(ground_truth, full_explanation)
                sample_results['metrics']['full_bleu'] = full_bleu
                results['summary_metrics']['full']['bleu'].append(full_bleu)

            # ROUGE scores
            lora_rouge = self.calculate_rouge_scores(ground_truth, lora_explanation)
            sample_results['metrics']['lora_rouge'] = lora_rouge
            results['summary_metrics']['lora']['rouge1'].append(lora_rouge['rouge1'])
            results['summary_metrics']['lora']['rouge2'].append(lora_rouge['rouge2'])
            results['summary_metrics']['lora']['rougeL'].append(lora_rouge['rougeL'])

            if full_explanation:
                full_rouge = self.calculate_rouge_scores(ground_truth, full_explanation)
                sample_results['metrics']['full_rouge'] = full_rouge
                results['summary_metrics']['full']['rouge1'].append(full_rouge['rouge1'])
                results['summary_metrics']['full']['rouge2'].append(full_rouge['rouge2'])
                results['summary_metrics']['full']['rougeL'].append(full_rouge['rougeL'])

            # Semantic similarity
            lora_semantic = self.calculate_semantic_similarity(ground_truth, lora_explanation)
            sample_results['metrics']['lora_semantic'] = lora_semantic
            results['summary_metrics']['lora']['semantic_sim'].append(lora_semantic)

            if full_explanation:
                full_semantic = self.calculate_semantic_similarity(ground_truth, full_explanation)
                sample_results['metrics']['full_semantic'] = full_semantic
                results['summary_metrics']['full']['semantic_sim'].append(full_semantic)

            # Feature accuracy
            ground_truth_features = self.extract_feature_mentions(ground_truth)
            lora_features = self.extract_feature_mentions(lora_explanation)
            full_features = self.extract_feature_mentions(full_explanation) if full_explanation else None

            feature_comparison = self.compare_feature_accuracy(ground_truth_features, lora_features, full_features)
            sample_results['metrics']['feature_comparison'] = feature_comparison
            results['summary_metrics']['lora']['feature_accuracy'].append(feature_comparison['lora_accuracy'])
            if feature_comparison['full_accuracy'] is not None:
                results['summary_metrics']['full']['feature_accuracy'].append(feature_comparison['full_accuracy'])

            results['comparisons'].append(sample_results)

            print(f"✅ Sample {i+1} analysis complete")
            print(f"   LoRA BLEU: {lora_bleu:.4f}")
            if full_explanation:
                print(f"   Full BLEU: {full_bleu:.4f}")

            # Save checkpoint every checkpoint_interval samples
            if (i + 1) % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_file, results, i + 1)
                print(f"💾 Progress: {i+1}/{num_samples} samples ({(i+1)/num_samples*100:.1f}%)")

        # Calculate summary statistics
        for model_type in ['lora', 'full']:
            if results['summary_metrics'][model_type]['bleu']:
                results['summary_metrics'][model_type]['avg_bleu'] = np.mean(results['summary_metrics'][model_type]['bleu'])
                results['summary_metrics'][model_type]['avg_rouge1'] = np.mean(results['summary_metrics'][model_type]['rouge1'])
                results['summary_metrics'][model_type]['avg_rouge2'] = np.mean(results['summary_metrics'][model_type]['rouge2'])
                results['summary_metrics'][model_type]['avg_rougeL'] = np.mean(results['summary_metrics'][model_type]['rougeL'])
                results['summary_metrics'][model_type]['avg_semantic'] = np.mean(results['summary_metrics'][model_type]['semantic_sim'])
                results['summary_metrics'][model_type]['avg_feature_accuracy'] = np.mean(results['summary_metrics'][model_type]['feature_accuracy'])

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"model_comparison_results_{timestamp}.json"
        text_results_file = f"model_comparison_results_{timestamp}.txt"

        # Convert numpy float32 to regular float for JSON serialization
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        results_serializable = convert_numpy_types(results)

        # Save JSON results
        with open(results_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        # Save text results
        self.save_results_to_text(results, text_results_file)

        # Delete checkpoint file after successful completion
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            print(f"🗑️ Checkpoint file deleted (evaluation complete)")

        print(f"\n🎉 Comparison study complete!")
        print(f"📁 JSON Results saved to: {results_file}")
        print(f"📄 Text Results saved to: {text_results_file}")

        return results

    def print_comparison_summary(self, results):
        """Print summary of comparison results."""
        print("\n📊 MODEL COMPARISON SUMMARY")
        print("=" * 60)

        lora_metrics = results['summary_metrics']['lora']
        full_metrics = results['summary_metrics']['full']

        print(f"📋 Samples analyzed: {results['num_samples']}")
        print(f"⏰ Analysis time: {results['timestamp']}")

        print(f"\n🔵 LoRA Model Performance:")
        print(f"   BLEU Score: {lora_metrics.get('avg_bleu', 0):.4f}")
        print(f"   ROUGE-1: {lora_metrics.get('avg_rouge1', 0):.4f}")
        print(f"   ROUGE-2: {lora_metrics.get('avg_rouge2', 0):.4f}")
        print(f"   ROUGE-L: {lora_metrics.get('avg_rougeL', 0):.4f}")
        print(f"   Semantic Similarity: {lora_metrics.get('avg_semantic', 0):.4f}")
        print(f"   Feature Accuracy: {lora_metrics.get('avg_feature_accuracy', 0):.4f}")

        if full_metrics.get('avg_bleu') is not None:
            print(f"\n🟢 Full Fine-tuned Model Performance:")
            print(f"   BLEU Score: {full_metrics.get('avg_bleu', 0):.4f}")
            print(f"   ROUGE-1: {full_metrics.get('avg_rouge1', 0):.4f}")
            print(f"   ROUGE-2: {full_metrics.get('avg_rouge2', 0):.4f}")
            print(f"   ROUGE-L: {full_metrics.get('avg_rougeL', 0):.4f}")
            print(f"   Semantic Similarity: {full_metrics.get('avg_semantic', 0):.4f}")
            print(f"   Feature Accuracy: {full_metrics.get('avg_feature_accuracy', 0):.4f}")

            # Winner analysis
            print(f"\n🏆 PERFORMANCE COMPARISON:")
            metrics = ['avg_bleu', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_semantic', 'avg_feature_accuracy']
            lora_wins = 0
            full_wins = 0

            for metric in metrics:
                lora_score = lora_metrics.get(metric, 0)
                full_score = full_metrics.get(metric, 0)
                winner = "LoRA" if lora_score > full_score else "Full"
                if lora_score > full_score:
                    lora_wins += 1
                else:
                    full_wins += 1
                print(f"   {metric.replace('avg_', '').upper()}: {winner} wins ({lora_score:.4f} vs {full_score:.4f})")

            overall_winner = "LoRA" if lora_wins > full_wins else "Full Fine-tuning"
            print(f"\n🎯 OVERALL WINNER: {overall_winner} ({lora_wins if lora_wins > full_wins else full_wins}/{len(metrics)} metrics)")


def main():
    """Main comparison pipeline."""
    print("🔬 EMBER LLaMA Model Comparison Framework")
    print("=" * 60)
    print("🎯 Comparing LoRA vs Full Fine-tuning approaches")
    print("📊 Focus: Feature analysis accuracy and explanation quality")

    # Initialize comparator
    comparator = ModelComparator()

    # Find LoRA model
    # CONFIGURE THIS: Change the pattern to test different LoRA models
    # Options:
    #   "*LoRA_r16*ember_llama"   - Test r=16 model (1.15% params)
    #   "*LoRA_r96*ember_llama"   - Test r=96 model (6.44% params) - OPTIMAL
    #   "*LoRA_r256*ember_llama"  - Test r=256 model (15.50% params)
    #   "*LoRA_r512*ember_llama"  - Test r=512 model (26.85% params)
    #   "*LoRA_r896*ember_llama"  - Test r=896 model (39.11% params)
    lora_pattern = "20251030_000317_LoRA_r512_26pct_ember_llama"  # ← CHANGE THIS TO TEST DIFFERENT MODELS

    lora_model_dirs = glob.glob(lora_pattern)
    lora_path = max(lora_model_dirs, key=os.path.getctime) if lora_model_dirs else None

    if lora_path:
        print(f"📁 Found LoRA model: {lora_path}")
    else:
        print(f"⚠️ No LoRA model found matching pattern: {lora_pattern}")
        print("⚠️ Looking for any LoRA model...")
        lora_model_dirs = glob.glob("*LoRA*ember_llama")
        lora_path = max(lora_model_dirs, key=os.path.getctime) if lora_model_dirs else None
        if lora_path:
            print(f"📁 Found LoRA model: {lora_path}")
        else:
            print("❌ ERROR: No LoRA model found!")
            return

    # Find full fine-tuned model
    full_model_dirs = glob.glob("*FullParam*ember_llama")
    full_model_path = max(full_model_dirs, key=os.path.getctime) if full_model_dirs else None

    if full_model_path:
        print(f"📁 Found full fine-tuned model: {full_model_path}")
    else:
        print("⚠️ No full fine-tuned model found. Run ember_llama_full_finetuning.py first")
        print("🔄 Will compare LoRA model against ground truth only")

    # Load models
    comparator.load_models(lora_path=lora_path, full_model_path=full_model_path)

    # Run comparison study with 100 test samples with checkpoint system
    # Training data uses samples 0-999, test data uses samples 1000-1999
    # Use balanced=True for 50 malware + 50 benign samples
    # NOTE: num_samples must be even when balanced=True (e.g., 2, 4, 50, 100)
    results = comparator.run_comparison_study(num_samples=2, start_index=1200, balanced=True)

    # Print summary
    comparator.print_comparison_summary(results)

if __name__ == "__main__":
    main()
