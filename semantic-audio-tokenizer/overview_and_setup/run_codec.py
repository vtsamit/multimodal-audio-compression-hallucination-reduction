#!/usr/bin/env python3
"""
run_codec.py - SemantiCodec with AAD+DAST Frame Selection

This script:
1. Reads importance scores from /content/importance_scores/ (computed by Colab)
2. Extracts only selected frames from original audio
3. Encodes selected frames with SemantiCodec
4. Saves compressed audio to data/replaced/

Note: AAD+DAST computation is done in Colab CELL 5-6, not here!
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
import librosa
import tempfile

# Add path to SemantiCodec
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from Semantic_codec import SemanticCodec
    CODEC_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: SemantiCodec not available, using fallback mode")
    CODEC_AVAILABLE = False

# ============================================================================
# CONFIGURATION
# ============================================================================

ORIGINAL_DIR = Path("data/original")
REPLACED_DIR = Path("data/replaced")
SCORES_DIR = Path("/content/importance_scores")

# Check if AAD+DAST mode is enabled
USE_FRAME_SELECTION = SCORES_DIR.exists() and any(SCORES_DIR.glob('*.npy'))

print("=" * 70)
print("  🎵 SemantiCodec Processing")
if USE_FRAME_SELECTION:
    num_scores = len(list(SCORES_DIR.glob('*.npy')))
    print(f"  🎯 Mode: AAD+DAST (Frame Selection ENABLED)")
    print(f"  📊 Importance scores: {num_scores} files")
    print(f"  📁 Scores directory: {SCORES_DIR}")
else:
    print(f"  ℹ️  Mode: Standard (No Frame Selection)")
print("=" * 70)

# ============================================================================
# FRAME SELECTION FUNCTION (KEY CHANGE)
# ============================================================================

def extract_selected_frames(audio: np.ndarray, sr: int, filename: str) -> np.ndarray:
    """
    Extract only selected frames based on importance scores from Colab
    
    This is the KEY function that implements AAD+DAST frame selection.
    It reads the pre-computed importance scores and extracts the selected frames.
    
    Args:
        audio: Full audio array
        sr: Sample rate (should be 16000)
        filename: Audio filename (e.g., 'sample_0001.wav')
    
    Returns:
        selected_audio: Audio with only selected frames concatenated
    """
    
    if not USE_FRAME_SELECTION:
        return audio
    
    # Load importance scores computed by Colab
    score_file = SCORES_DIR / filename.replace('.wav', '.npy')
    
    if not score_file.exists():
        print(f"  ⚠️ No scores for {filename}, using full audio")
        return audio
    
    # Load selection metadata
    try:
        score_data = np.load(score_file, allow_pickle=True).item()
        selected_indices = score_data['selected_indices']
        num_frames = score_data['num_frames']
        keep_ratio = score_data.get('keep_ratio', 0.7)
        alpha = score_data.get('alpha', 0.5)
    except Exception as e:
        print(f"  ⚠️ Error loading scores for {filename}: {e}")
        return audio
    
    # Calculate samples per frame
    samples_per_frame = len(audio) // num_frames
    
    if samples_per_frame == 0:
        print(f"  ⚠️ Warning: samples_per_frame is 0 for {filename}")
        return audio
    
    # Extract selected frames
    selected_frames = []
    for idx in selected_indices:
        start = idx * samples_per_frame
        end = min(start + samples_per_frame, len(audio))
        selected_frames.append(audio[start:end])
    
    # Concatenate all selected frames
    selected_audio = np.concatenate(selected_frames)
    
    # Log compression info
    compression_pct = (1 - len(selected_indices)/num_frames) * 100
    print(f"  ✓ {filename}: {len(selected_indices)}/{num_frames} frames")
    print(f"    Compression: {compression_pct:.0f}%, Alpha: {alpha}, Keep: {keep_ratio:.0%}")
    
    return selected_audio

# ============================================================================
# CODEC PROCESSING
# ============================================================================

def process_with_semanticodec(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Encode and decode audio with SemantiCodec
    
    Args:
        audio: Audio array (already frame-selected if AAD+DAST mode)
        sr: Sample rate
    
    Returns:
        decoded_audio: Reconstructed audio
    """
    
    if not CODEC_AVAILABLE:
        # Fallback: return as-is
        return audio
    
    try:
        # Initialize codec
        codec = SemanticCodec(
            token_rate=100,
            vocab_size=16384
        )
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
        
        # Encode
        codes = codec.encode(audio_tensor)
        
        # Decode
        reconstructed = codec.decode(codes)
        
        # Convert back to numpy
        if isinstance(reconstructed, torch.Tensor):
            decoded_audio = reconstructed.squeeze().cpu().numpy()
        else:
            decoded_audio = reconstructed
        
        return decoded_audio
        
    except Exception as e:
        print(f"  ⚠️ Error in SemantiCodec: {e}")
        return audio

# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def main():
    """
    Main processing loop
    
    For each audio file:
    1. Load audio
    2. Extract selected frames (if AAD+DAST mode)
    3. Encode with SemantiCodec
    4. Save compressed audio
    """
    
    # Check input directory
    if not ORIGINAL_DIR.exists():
        print(f"❌ Error: Input directory not found: {ORIGINAL_DIR}")
        print(f"   Expected: {ORIGINAL_DIR.resolve()}")
        return
    
    # Setup output directory
    REPLACED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get audio files
    wav_files = sorted(ORIGINAL_DIR.glob('*.wav'))
    
    if not wav_files:
        print(f"❌ Error: No .wav files found in {ORIGINAL_DIR}")
        return
    
    print(f"\n📁 Found {len(wav_files)} audio files")
    print(f"📁 Input:  {ORIGINAL_DIR.resolve()}")
    print(f"📁 Output: {REPLACED_DIR.resolve()}")
    print()
    
    # Process each file
    success_count = 0
    
    for idx, input_path in enumerate(wav_files, 1):
        try:
            print(f"[{idx}/{len(wav_files)}] Processing: {input_path.name}")
            
            # 1. Load audio
            audio, sr = librosa.load(str(input_path), sr=16000, mono=True)
            original_length = len(audio)
            
            # 2. Extract selected frames (if AAD+DAST mode)
            if USE_FRAME_SELECTION:
                audio = extract_selected_frames(audio, sr, input_path.name)
                print(f"    Audio length: {original_length} → {len(audio)} samples")
            else:
                print(f"    Audio length: {len(audio)} samples")
            
            # 3. Encode with SemantiCodec
            processed_audio = process_with_semanticodec(audio, sr)
            
            # 4. Save output
            output_path = REPLACED_DIR / input_path.name
            sf.write(str(output_path), processed_audio, sr)
            
            print(f"    ✅ Saved: {output_path.name}")
            success_count += 1
            
        except Exception as e:
            print(f"    ❌ Error processing {input_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print()
    print("=" * 70)
    print(f"✅ Processing complete!")
    print(f"   Success: {success_count}/{len(wav_files)} files")
    print(f"   Output:  {REPLACED_DIR.resolve()}")
    print("=" * 70)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
