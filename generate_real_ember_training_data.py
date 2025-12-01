#!/usr/bin/env python3
"""
Generate Training Data from Real EMBER 2018 Dataset
This script creates training examples for LLaMA using actual malware and benign samples.
"""

import json
import os
import time
from datetime import datetime
from real_ember_analysis import load_real_ember_samples, create_feature_vector_from_real_ember
from ember_llama_inference import EMBERLLaMAInference

def generate_real_ember_training_data(num_samples=1000, start_index=0):
    """Generate training data from real EMBER samples.

    Args:
        num_samples: Number of samples to generate
        start_index: Starting index in the EMBER dataset (for train/test split)
    """

    print("🎯 REAL EMBER TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"📊 Target samples: {num_samples}")
    print(f"📊 Starting from index: {start_index}")
    print(f"📂 Source: EMBER 2018 dataset (real PE files)")
    
    # Load real EMBER samples
    print(f"\n📂 Loading REAL EMBER 2018 dataset samples...")
    real_samples = load_real_ember_samples(max_samples=num_samples, start_index=start_index)
    print(f"✅ Loaded {len(real_samples)} real samples")
    
    # Initialize EMBER pipeline for analysis
    print(f"\n🔍 Initializing EMBER analysis pipeline...")
    pipeline = EMBERLLaMAInference()
    print(f"✅ Pipeline ready")
    
    # Generate training examples
    training_examples = []
    malware_count = 0
    benign_count = 0
    
    print(f"\n🚀 Generating training examples from real data...")
    
    for i, sample in enumerate(real_samples, 1):
        print(f"\n📋 Processing sample {i}/{len(real_samples)}: {sample['sha256'][:16]}...")
        
        try:
            # Get ground truth
            label_str = "MALWARE" if sample['label'] == 1 else "BENIGN"
            avclass = sample.get('avclass', 'N/A')
            
            if sample['label'] == 1:
                malware_count += 1
            else:
                benign_count += 1
            
            print(f"🏷️ Ground Truth: {label_str}")
            if avclass and avclass != 'N/A':
                print(f"🦠 Malware Family: {avclass}")
            
            # Convert to feature vector
            feature_vector = create_feature_vector_from_real_ember(sample)
            
            # Get EMBER prediction
            ember_score = pipeline.ember_model.predict([feature_vector])[0]
            classification = "MALWARE" if ember_score > 0.5 else "BENIGN"
            confidence = max(ember_score, 1 - ember_score)
            
            print(f"📊 EMBER Score: {ember_score:.6f}")
            print(f"📊 Classification: {classification}")
            
            # Generate detailed explanation using real features
            explanation = generate_real_explanation(sample, ember_score, classification, confidence)
            
            # Create training example
            training_example = {
                'input': f"""ANALYZE REAL EMBER 2018 SAMPLE:
SHA256: {sample['sha256']}
EMBER Score: {ember_score:.6f}
Classification: {classification}
Ground Truth: {label_str}
Malware Family: {avclass}

Provide detailed technical analysis explaining this classification.""",
                
                'output': explanation,
                
                'metadata': {
                    'sha256': sample['sha256'],
                    'ember_score': ember_score,
                    'classification': classification,
                    'ground_truth': label_str,
                    'malware_family': avclass,
                    'confidence': confidence,
                    'sample_type': 'real_ember_2018'
                }
            }
            
            training_examples.append(training_example)
            print(f"✅ Training example generated ({len(explanation)} chars)")
            
        except Exception as e:
            print(f"❌ Error processing sample: {e}")
            continue
    
    # Save training data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"real_ember_llama_training_{timestamp}.jsonl"
    
    print(f"\n💾 Saving training data to {output_file}...")
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    # Generate summary
    print(f"\n🎉 REAL EMBER TRAINING DATA GENERATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Total training examples: {len(training_examples)}")
    print(f"🦠 Malware samples: {malware_count}")
    print(f"✅ Benign samples: {benign_count}")
    print(f"📄 Output file: {output_file}")
    print(f"💡 Ready for fine-tuning with real EMBER data!")
    
    return training_examples, output_file

def generate_real_explanation(sample, ember_score, classification, confidence):
    """Generate detailed explanation for real EMBER sample."""
    
    # Extract key features from real sample
    imports = sample['features']['imports_summary']
    sections = sample['features']['sections_summary'] 
    general = sample['features']['general']
    strings = sample['features']['strings']
    header = sample['features']['header_summary']
    
    explanation = f"""This real EMBER 2018 sample receives a score of {ember_score:.6f}, classifying it as {classification} with {confidence:.1%} confidence.

KEY FEATURE ANALYSIS:

🔗 IMPORT ANALYSIS:
- Total API imports: {imports['total_imports']}
- Suspicious APIs detected: {imports['suspicious_apis']}
- kernel32.dll imports: {imports['kernel32_imports']}
- ntdll.dll imports: {imports['ntdll_imports']}
- advapi32.dll imports: {imports['advapi32_imports']}

📁 SECTION ANALYSIS:
- Total sections: {sections['num_sections']}
- Executable sections: {sections['executable_sections']}
- High entropy sections (>7.0): {sections['high_entropy_sections']}
- Maximum entropy: {sections['max_entropy']:.3f}

📄 FILE CHARACTERISTICS:
- File size: {general['size']:,} bytes
- Virtual size: {general['vsize']:,} bytes
- Has digital signature: {general['has_signature']}
- Has resources: {general['has_resources']}

🔤 STRING ANALYSIS:
- Number of strings: {strings['numstrings']}
- Average string length: {strings['avlength']:.2f}
- String entropy: {strings['entropy']:.3f}
- Registry references: {strings['registry']}
- URL references: {strings['urls']}

CLASSIFICATION REASONING:
"""
    
    # Add specific reasoning based on features
    if imports['suspicious_apis'] > 0:
        explanation += f"⚠️ RISK INDICATOR: {imports['suspicious_apis']} suspicious API calls suggest potential malicious behavior.\n"
    
    if sections['high_entropy_sections'] > 0:
        explanation += f"⚠️ RISK INDICATOR: {sections['high_entropy_sections']} high-entropy sections may indicate packing or obfuscation.\n"
    
    if sections['max_entropy'] > 7.0:
        explanation += f"⚠️ RISK INDICATOR: Maximum entropy of {sections['max_entropy']:.3f} suggests compressed or encrypted content.\n"
    
    if general['size'] > 1000000:
        explanation += f"⚠️ SIZE INDICATOR: Large file size ({general['size']:,} bytes) requires additional scrutiny.\n"
    
    if strings['entropy'] > 6.0:
        explanation += f"⚠️ STRING INDICATOR: High string entropy ({strings['entropy']:.3f}) may indicate obfuscation.\n"
    
    # Add benign indicators
    if imports['suspicious_apis'] == 0:
        explanation += f"✅ BENIGN INDICATOR: No suspicious API calls detected.\n"
    
    if sections['high_entropy_sections'] == 0:
        explanation += f"✅ BENIGN INDICATOR: No high-entropy sections found.\n"
    
    if general['has_signature']:
        explanation += f"✅ BENIGN INDICATOR: File has digital signature.\n"
    
    explanation += f"\nThis analysis demonstrates how EMBER's 2,381 features from real PE files contribute to malware detection decisions."
    
    return explanation

if __name__ == "__main__":
    # Generate training data from samples 0-999 (first 1000 samples)
    # Test data will use samples 1000-1999 (next 1000 samples)
    generate_real_ember_training_data(num_samples=1000, start_index=0)
