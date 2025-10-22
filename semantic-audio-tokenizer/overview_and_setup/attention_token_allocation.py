"""
AAD+DAST Framework for Audio Compression with Hallucination Reduction
Novel approach combining Audio-Aware Decoding (AAD) and Dynamic Adaptive Semantic Tokenization (DAST)
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
import warnings
import gc

warnings.filterwarnings("ignore")

from pesq import pesq
from pystoi import stoi


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"  [MEM] GPU allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


# ============================================================================
# QUALITY METRICS
# ============================================================================

class QualityMetrics:    
    def __init__(self, sr=16000):
        self.sr = sr
    
    def evaluate_quality(self, reference_audio, degraded_audio):      
        min_len = min(len(reference_audio), len(degraded_audio))
        ref = reference_audio[:min_len]
        deg = degraded_audio[:min_len]
        
        metrics = {}
        
        try:
            pesq_score = pesq(self.sr, ref, deg, 'wb')
            metrics['pesq'] = {
                'score': pesq_score,
                'interpretation': self.pesq_interpretation(pesq_score)
            }
        except Exception as e:
            print(f"PESQ failed: {e}")
            metrics['pesq'] = {'score': 0.0, 'interpretation': 'Failed'}

        try:
            stoi_score = stoi(ref, deg, self.sr)
            metrics['stoi'] = {
                'score': stoi_score,
                'interpretation': self.stoi_interpretation(stoi_score)
            }
        except Exception as e:
            print(f"STOI failed: {e}")
            metrics['stoi'] = {'score': 0.0, 'interpretation': 'Failed'}
        
        return metrics
    
    def pesq_interpretation(self, score):
        if score >= 4.0: return "Excellent quality"
        elif score >= 3.0: return "Good quality"
        elif score >= 2.0: return "Fair quality"
        else: return "Poor quality"
    
    def stoi_interpretation(self, score):
        if score >= 0.9: return "High intelligibility"
        elif score >= 0.7: return "Good intelligibility"
        elif score >= 0.5: return "Fair intelligibility"
        else: return "Poor intelligibility"


# ============================================================================
# AAD+DAST PROCESSOR
# ============================================================================

class AADDASTProcessor:
    """
    Compute importance scores using LALM (Audio-Aware Decoding + DAST fusion)
    Phase 1: Importance scoring with AAD (local) + Attention (global)
    Phase 2: Frame selection based on combined importance scores
    """
    
    def __init__(self, model, processor, alpha=0.5, keep_ratio=0.7):
        """
        Args:
            model: LALM (e.g., Qwen2-Audio)
            processor: Corresponding audio-text processor
            alpha: Balance between global (attention) and local (AAD)
                   α=1.0: Pure attention, α=0.0: Pure AAD, α=0.5: Balanced
            keep_ratio: Proportion of frames to keep (0.7 = 70%)
        """
        self.model = model
        self.processor = processor
        self.alpha = alpha
        self.keep_ratio = keep_ratio
        
    def compute_importance_scores(self, audio_path, question, num_frames):
        """
        Phase 1: Compute AAD+DAST importance scores
        
        Returns:
            importance_scores: [num_frames] normalized scores
            debug_info: Dictionary with intermediate values
        """
        print(f"[Phase 1] Computing importance for {num_frames} frames...")
        
        # Load raw audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_np = audio.astype(np.float32)
        silence_np = np.zeros_like(audio_np)
        
        # Prepare LALM inputs
        inputs_with = self._prepare_inputs(audio_np, question)
        inputs_without = self._prepare_inputs(silence_np, question)
        
        # Run LALM (2 forward passes for AAD)
        print("  → Running LALM with audio...")
        with torch.no_grad():
            outputs_with = self.model.generate(
                **inputs_with,
                max_new_tokens=10,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        print("  → Running LALM with silence...")
        with torch.no_grad():
            outputs_without = self.model.generate(
                **inputs_without,
                max_new_tokens=10,
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )
        
        # Compute local importance (AAD signal)
        print("  → Computing local importance (AAD)...")
        local_scores = self._compute_local_importance(
            outputs_with, outputs_without, num_frames
        )
        
        # Compute global importance (Attention aggregation)
        print("  → Computing global importance (Attention)...")
        global_scores = self._compute_global_importance(
            outputs_with, num_frames
        )
        
        # DAST fusion: combine local and global
        print(f"  → Fusing scores (α={self.alpha})...")
        importance_scores = self._fuse_scores(local_scores, global_scores)
        
        debug_info = {
            'local_scores': local_scores,
            'global_scores': global_scores,
            'alpha': self.alpha
        }
        
        return importance_scores, debug_info
    
    def _compute_local_importance(self, outputs_with, outputs_without, num_frames):
        """
        Local importance: AAD signal magnitude per frame
        Uses attention weights to map AAD signal back to audio frames
        """
        local_importance = np.zeros(num_frames)
        n_generated = len(outputs_with.scores)
        
        for t in range(n_generated):
            # Compute AAD signal for this token
            logits_with = outputs_with.scores[t][0].float().cpu()
            logits_without = outputs_without.scores[t][0].float().cpu()
            aad_magnitude = torch.abs(logits_with - logits_without).max().item()
            
            # Extract attention weights to audio frames
            if hasattr(outputs_with, 'attentions') and outputs_with.attentions:
                try:
                    attn = outputs_with.attentions[t][-1]  # Last layer
                    attn_weights = attn.mean(dim=1)[0, -1, :].cpu().numpy()
                    
                    # Map to audio frames (adjust indices based on model)
                    audio_start = 1  # Typically after BOS token
                    audio_end = audio_start + num_frames
                    
                    if audio_end <= len(attn_weights):
                        audio_attn = attn_weights[audio_start:audio_end]
                        audio_attn = audio_attn / (audio_attn.sum() + 1e-8)
                        
                        # Accumulate: importance[frame] += AAD_signal * attention[frame]
                        local_importance += aad_magnitude * audio_attn
                except Exception as e:
                    print(f"    Warning: Attention extraction failed for token {t}: {e}")
                    continue
        
        # Normalize to [0, 1]
        if local_importance.max() > 0:
            local_importance = local_importance / local_importance.max()
        else:
            local_importance = np.ones(num_frames) / num_frames
        
        return local_importance
    
    def _compute_global_importance(self, outputs_with, num_frames):
        """
        Global importance: Attention aggregation across all generated tokens
        Captures which frames the model attends to most
        """
        global_importance = np.zeros(num_frames)
        
        for t in range(len(outputs_with.attentions)):
            try:
                attn = outputs_with.attentions[t][-1]  # Last layer
                attn_weights = attn.mean(dim=1)[0, -1, :].cpu().numpy()
                
                # Map to audio frames
                audio_start = 1
                audio_end = audio_start + num_frames
                
                if audio_end <= len(attn_weights):
                    audio_attn = attn_weights[audio_start:audio_end]
                    global_importance += audio_attn
            except:
                continue
        
        # Normalize to [0, 1]
        if global_importance.max() > 0:
            global_importance = global_importance / global_importance.max()
        else:
            global_importance = np.ones(num_frames) / num_frames
        
        return global_importance
    
    def _fuse_scores(self, local_scores, global_scores):
        """
        DAST fusion formula: S = G·α + (L/ΣL)·(1-α)
        
        Args:
            local_scores: AAD-based importance [num_frames]
            global_scores: Attention-based importance [num_frames]
        
        Returns:
            Combined importance scores [num_frames]
        """
        # Normalize local scores
        local_norm = local_scores / (local_scores.sum() + 1e-8)
        
        # Combine: global * α + local * (1 - α)
        combined = global_scores * self.alpha + local_norm * (1 - self.alpha)
        
        # Softmax normalization for final scores
        combined_exp = np.exp(combined - combined.max())
        scores = combined_exp / combined_exp.sum()
        
        return scores
    
    def select_frames(self, importance_scores):
        """
        Phase 2: Select top-k% frames based on importance
        
        Returns:
            selected_indices: Frame indices to keep (sorted)
            selection_mask: Boolean mask [num_frames]
        """
        num_frames = len(importance_scores)
        target_keep = int(num_frames * self.keep_ratio)
        
        # Select top-k frames by importance
        top_indices = np.argsort(importance_scores)[-target_keep:]
        top_indices = np.sort(top_indices)  # Keep temporal order
        
        # Create boolean mask
        selection_mask = np.zeros(num_frames, dtype=bool)
        selection_mask[top_indices] = True
        
        return top_indices, selection_mask
    
    def _prepare_inputs(self, audio_np, question):
        """Prepare inputs for LALM"""
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio"},
                {"type": "text", "text": question},
            ],
        }]
        
        try:
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        except Exception:
            # Fallback for models without chat template
            audio_tok = getattr(self.processor, "audio_token", "<|AUDIO|>")
            prompt = f"{audio_tok}\n{question}"
            inputs = self.processor(
                text=prompt, audio=audio_np, sampling_rate=16000, return_tensors="pt"
            )
        
        # Move to model device
        device = next(self.model.parameters()).device
        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                inputs[k] = v.to(device)
        
        return inputs


# ============================================================================
# CODEC WRAPPER (WITH PROPER MEMORY MANAGEMENT)
# ============================================================================

class AADDASTSemanticCodec(nn.Module):
    """
    SemantiCodec with AAD+DAST guided frame selection
    
    Novel approach:
    1. Compute importance using LALM (AAD+DAST)
    2. Select top-k% most important frames
    3. Encode ONLY selected frames with SemantiCodec
    4. Result: Fewer tokens, reduced hallucination
    """
    
    def __init__(self, aad_model, aad_processor, token_rate=100, 
                 semantic_vocab_size=16384, alpha=0.5, keep_ratio=0.7):
        super().__init__()
        
        # Store codec parameters (don't load SemantiCodec yet!)
        self.token_rate = token_rate
        self.semantic_vocab_size = semantic_vocab_size
        
        # Initialize AAD+DAST processor (uses LALM only)
        self.aad_dast = AADDASTProcessor(aad_model, aad_processor, alpha, keep_ratio)
        self.quality_metrics = QualityMetrics()
        
        # SemantiCodec will be loaded later
        self.base_codec = None
    
    def encode(self, audio_path, question):
        """
        Phase 1-2: Compute importance and select frames
        (LALM is used here, SemantiCodec not loaded yet)
        
        Returns:
            Dictionary with selected frame indices and metadata
        """
        print(f"\n{'='*70}")
        print(f"[ENCODE] Audio: {Path(audio_path).name}")
        print(f"[ENCODE] Question: {question}")
        print(f"{'='*70}\n")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        num_frames = int(duration * self.token_rate)
        
        print(f"[Info] Audio duration: {duration:.2f}s")
        print(f"[Info] Expected frames: {num_frames}")
        
        # PHASE 1: Importance scoring (uses LALM)
        importance_scores, debug_info = self.aad_dast.compute_importance_scores(
            audio_path, question, num_frames
        )
        
        # PHASE 2: Frame selection
        print(f"\n[Phase 2] Selecting frames (keep_ratio={self.aad_dast.keep_ratio})...")
        selected_indices, selection_mask = self.aad_dast.select_frames(importance_scores)
        print(f"  → Selected {len(selected_indices)}/{num_frames} frames ({len(selected_indices)/num_frames:.1%})")
        
        # Return metadata (encoding happens later, after LALM is unloaded)
        return {
            'audio_path': audio_path,
            'selected_indices': selected_indices,
            'selection_mask': selection_mask,
            'importance_scores': importance_scores,
            'compression_ratio': len(selected_indices) / num_frames,
            'question': question,
            'num_frames': num_frames,
            'debug_info': debug_info
        }
    
    def encode_selected_frames(self, result_dict):
        """
        Phase 3: Load SemantiCodec and encode selected frames
        (Call this AFTER unloading LALM to avoid OOM)
        
        Args:
            result_dict: Output from encode() method
        
        Returns:
            Updated result_dict with 'tokens' field
        """
        print(f"\n[Phase 3] Loading SemantiCodec and encoding selected frames...")
        
        # NOW load SemantiCodec (LALM should be unloaded by now)
        from semanticodec import SemantiCodec as OriginalSemantiCodec
        
        print("  → Loading SemantiCodec...")
        self.base_codec = OriginalSemantiCodec(
            token_rate=self.token_rate,
            semantic_vocab_size=self.semantic_vocab_size
        )
        clear_gpu_memory()
        
        # Extract selected audio frames
        audio_path = result_dict['audio_path']
        selected_indices = result_dict['selected_indices']
        num_frames = result_dict['num_frames']
        
        print("  → Extracting selected frames...")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        selected_audio = self._extract_audio_frames(audio, sr, selected_indices, num_frames)
        
        # Save to temporary file
        temp_path = "temp_selected_audio.wav"
        sf.write(temp_path, selected_audio, sr)
        
        # Encode with SemantiCodec
        print("  → Encoding with SemantiCodec...")
        tokens = self.base_codec.encode(temp_path)
        
        # Clean up temp file
        Path(temp_path).unlink()
        
        # Add tokens to result
        result_dict['tokens'] = tokens
        
        print(f"  → Encoded to {tokens.shape[1]} tokens (shape: {tokens.shape})")
        print(f"  → Compression: {num_frames} frames → {tokens.shape[1]} tokens")
        
        return result_dict
    
    def _extract_audio_frames(self, audio, sr, selected_indices, total_frames):
        """
        Extract selected raw audio frames and concatenate them
        
        CRITICAL: We extract RAW AUDIO, not feature vectors!
        """
        samples_per_frame = len(audio) // total_frames
        selected_audio = []
        
        for idx in selected_indices:
            start = idx * samples_per_frame
            end = min(start + samples_per_frame, len(audio))
            selected_audio.append(audio[start:end])
        
        # Concatenate all selected segments
        return np.concatenate(selected_audio)
    
    def decode(self, encoded_data):
        """
        Decode tokens back to audio
        
        Args:
            encoded_data: Either dict with 'tokens' key or tokens directly
        
        Returns:
            Decoded audio waveform
        """
        if isinstance(encoded_data, dict):
            tokens = encoded_data.get('tokens')
        else:
            tokens = encoded_data
        
        # Load SemantiCodec if not already loaded
        if self.base_codec is None:
            from semanticodec import SemantiCodec as OriginalSemantiCodec
            self.base_codec = OriginalSemantiCodec(
                token_rate=self.token_rate,
                semantic_vocab_size=self.semantic_vocab_size
            )
        
        # Ensure tokens are on correct device
        target_device = next(self.base_codec.encoder.parameters()).device
        if tokens.device != target_device:
            tokens = tokens.to(target_device)
        
        return self.base_codec.decode(tokens)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_aad_dast_codec(model, processor, audio_path, question, 
                        alpha=0.5, keep_ratio=0.7):
    """
    Test function for AAD+DAST codec
    
    Args:
        model: LALM (Qwen2-Audio, etc.)
        processor: Corresponding processor
        audio_path: Path to test audio
        question: Question for hallucination context
        alpha: AAD+DAST fusion parameter (0.0-1.0)
        keep_ratio: Proportion of frames to keep (0.0-1.0)
    """
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
    
    # Initialize codec
    codec = AADDASTSemanticCodec(
        aad_model=model,
        aad_processor=processor,
        alpha=alpha,
        keep_ratio=keep_ratio
    )
    
    # Phase 1-2: Importance scoring and frame selection
    result = codec.encode(audio_path, question)
    
    # Unload LALM (in practice, this would be done by calling code)
    print("\n[NOTE] In production, LALM would be unloaded here")
    print("[NOTE] For testing, we keep it loaded")
    
    # Phase 3: Encode selected frames
    result = codec.encode_selected_frames(result)
    
    # Decode
    waveform = codec.decode(result)
    
    # Print results
    if isinstance(result, dict):
        print("\n=== AAD+DAST Results ===")
        print(f"Question: {question}")
        print(f"Compression: {result['compression_ratio']:.1%} frames kept")
        print(f"Original frames: {result['num_frames']}")
        print(f"Selected frames: {len(result['selected_indices'])}")
        print(f"Final tokens: {result['tokens'].shape[1]}")
        print(f"Alpha (fusion): {alpha}")
        print(f"Keep ratio: {keep_ratio}")
        
        return True
    else:
        print("Processing failed")
        return False


if __name__ == "__main__":
    print("AAD+DAST Framework")
    print("=" * 70)
    print("Novel approach combining:")
    print("  - Audio-Aware Decoding (AAD) for local importance")
    print("  - DAST-style attention fusion for global importance")
    print("  - Frame selection (not quantization) for hallucination reduction")
    print("=" * 70)
    print("\nTo use:")
    print("1. Load your LALM (e.g., Qwen2-Audio)")
    print("2. Create codec = AADDASTSemanticCodec(model, processor)")
    print("3. result = codec.encode(audio_path, question)")
    print("4. Unload LALM (del model, processor, clear_gpu_memory())")
    print("5. result = codec.encode_selected_frames(result)")
    print("6. waveform = codec.decode(result)")
    print("\nReady!")
