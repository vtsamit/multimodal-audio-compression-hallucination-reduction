"""
Batch processing script for AAD+DAST SemantiCodec
Processes audio files with frame selection based on importance scores
"""

import os
import tempfile
from pathlib import Path
import torch
import soundfile as sf
import gc


# Import the novel codec
from attention_token_allocation import AADDASTSemanticCodec, clear_gpu_memory


# ============================================================================
# CONFIGURATION
# ============================================================================

ORIGINAL_DIR = Path("data/original")
REPLACED_DIR = Path("data/replaced")

# Codec settings
TOKEN_RATE = 100
VOCAB_SIZE = 16384

# AAD+DAST settings
ALPHA = 0.5         # Fusion parameter: 0.0=AAD only, 0.5=balanced, 1.0=attention only
KEEP_RATIO = 0.7    # Keep 70% of frames, drop 30%


# ============================================================================
# LALM INITIALIZATION
# ============================================================================

def load_lalm():
    """
    Load LALM with proper configuration for AAD+DAST (Colab Pro optimized)
    Uses 8-bit quantization
    """
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    
    print("\n🧠 Loading Qwen2-Audio-7B with 8-bit quantization...")
    
    model_name = "Qwen/Qwen2-Audio-7B-Instruct"
    
    # Simple 8-bit config (no CPU offload needed on Colab Pro)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    # Load model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model.eval()
    
    print(f"✅ LALM loaded: {model_name}")
    print(f"✅ Attention mode: eager (supports AAD+DAST)")
    print(f"📊 GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    return model, processor




def unload_lalm(model, processor):
    """Unload LALM to free GPU memory"""
    print("\n🗑️ Unloading LALM...")
    del model
    del processor
    clear_gpu_memory()
    print(f"📊 GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_audio(input_path: Path, output_path: Path, 
                  lalm_model, lalm_processor,
                  question: str = "What sounds are in this audio?",
                  alpha: float = ALPHA,
                  keep_ratio: float = KEEP_RATIO):
    """
    Process single audio file with AAD+DAST codec
    
    Pipeline:
    1. Compute importance with LALM
    2. Select important frames
    3. Unload LALM
    4. Load SemantiCodec
    5. Encode selected frames
    6. Decode and save
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*70}")
    
    # Initialize codec (doesn't load SemantiCodec yet)
    codec = AADDASTSemanticCodec(
        aad_model=lalm_model,
        aad_processor=lalm_processor,
        alpha=alpha,
        keep_ratio=keep_ratio
    )
    
    try:
        # Phase 1-2: Compute importance and select frames (uses LALM)
        result = codec.encode(str(input_path), question)
        
        # Phase 3: Encode selected frames (will load SemantiCodec)
        # Note: In this single-file version, LALM is still loaded
        # For batch processing, LALM would be unloaded here
        result = codec.encode_selected_frames(result)
        
        # Decode to audio
        waveform = codec.decode(result)
        
        # Save output
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform
        
        sf.write(str(output_path), waveform_np[0, 0], 16000)
        
        # Print statistics
        if isinstance(result, dict):
            print(f"\n✅ Success!")
            print(f"  → Compression: {result['compression_ratio']:.1%} frames kept")
            print(f"  → Original frames: {result['num_frames']}")
            print(f"  → Selected frames: {len(result['selected_indices'])}")
            print(f"  → Final tokens: {result['tokens'].shape[1]}")
            print(f"  → Saved: {output_path}")
        
    except Exception as e:
        print(f"  ✗ Failed: {input_path.name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()


def process_video(input_path: Path, output_path: Path,
                  lalm_model, lalm_processor,
                  question: str = "What sounds are in this audio?",
                  alpha: float = ALPHA,
                  keep_ratio: float = KEEP_RATIO):
    """
    Process video file: extract audio, compress with AAD+DAST, mux back
    """
    import ffmpeg
    
    print(f"\n{'='*70}")
    print(f"Processing video: {input_path.name}")
    print(f"{'='*70}")
    
    REPLACED_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        extracted_audio = tmpdir / "extracted.wav"
        processed_audio = tmpdir / "processed.wav"

        # Extract audio
        print("  → Extracting audio...")
        (
            ffmpeg
            .input(str(input_path))
            .output(str(extracted_audio), ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )

        # Process with AAD+DAST
        print("  → Processing with AAD+DAST...")
        codec = AADDASTSemanticCodec(
            aad_model=lalm_model,
            aad_processor=lalm_processor,
            alpha=alpha,
            keep_ratio=keep_ratio
        )
        
        result = codec.encode(str(extracted_audio), question)
        result = codec.encode_selected_frames(result)
        waveform = codec.decode(result)
        
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.cpu().numpy()
        else:
            waveform_np = waveform
        
        sf.write(str(processed_audio), waveform_np[0, 0], 16000)
        
        # Print stats
        if isinstance(result, dict):
            print(f"  → Compression: {result['compression_ratio']:.1%} frames kept")

        # Mux video + processed audio
        print("  → Muxing video and audio...")
        video_stream = ffmpeg.input(str(input_path))
        new_audio_stream = ffmpeg.input(str(processed_audio))

        (
            ffmpeg
            .output(
                video_stream.video,
                new_audio_stream.audio,
                str(output_path),
                vcodec="copy",
                acodec="aac",
                shortest=None
            )
            .overwrite_output()
            .run(quiet=True)
        )

    print(f"  ✅ Saved: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Main processing loop
    Processes all audio/video files in ORIGINAL_DIR
    """
    if not ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Input folder not found: {ORIGINAL_DIR.resolve()}")

    REPLACED_DIR.mkdir(parents=True, exist_ok=True)

    # Load LALM once for all files
    lalm_model, lalm_processor = load_lalm()

    try:
        # Find files
        mp4s, wavs = [], []
        for p in ORIGINAL_DIR.iterdir():
            if p.suffix.lower() == ".mp4":
                mp4s.append(p)
            elif p.suffix.lower() == ".wav":
                wavs.append(p)
            else:
                print(f"Skipping: {p.name} (not wav/mp4)")

        mp4s = sorted(mp4s)
        wavs = sorted(wavs)

        if not mp4s and not wavs:
            print(f"No .mp4/wav files found in {ORIGINAL_DIR.resolve()}")
            return

        print(f"\nFound {len(wavs)} wav files and {len(mp4s)} mp4 files")
        print(f"Settings:")
        print(f"  - Alpha (AAD+DAST fusion): {ALPHA}")
        print(f"  - Keep ratio: {KEEP_RATIO}")
        print(f"  - Token rate: {TOKEN_RATE}")
        print("="*70)

        # Process WAV files
        for src in wavs:
            dst = REPLACED_DIR / src.name
            try:
                process_audio(
                    src, dst, 
                    lalm_model, lalm_processor,
                    question="What sounds are in this audio?",
                    alpha=ALPHA,
                    keep_ratio=KEEP_RATIO
                )
            except Exception as e:
                print(f"  ✗ Failed: {src.name} - {e}")

        # Process MP4 files
        for src in mp4s:
            dst = REPLACED_DIR / src.name
            try:
                process_video(
                    src, dst,
                    lalm_model, lalm_processor,
                    question="What sounds are in this audio?",
                    alpha=ALPHA,
                    keep_ratio=KEEP_RATIO
                )
            except Exception as e:
                print(f"  ✗ Failed: {src.name} - {e}")

        print("\n" + "="*70)
        print("Processing complete!")
        
    finally:
        # Always clean up LALM
        unload_lalm(lalm_model, lalm_processor)


if __name__ == "__main__":
    main()
