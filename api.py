#!/usr/bin/env python3
"""
Stable Audio API - Enhanced with Style Transfer capabilities
Designed to be called alongside existing websockets backend
"""

from flask import Flask, request, jsonify, send_file
import torch
import torchaudio
import io
import base64
import uuid
import os
import time
import re
import threading
import gc
from einops import rearrange
from huggingface_hub import login
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from contextlib import contextmanager

from riff_manager import RiffManager

from model_loader_enhanced import model_manager, load_model

import soundfile as sf

def save_audio(buffer, audio_tensor, sample_rate):
    """Save audio with soundfile backend (supports BytesIO)"""
    # Convert tensor to numpy and transpose for soundfile (expects [samples, channels])
    audio_np = audio_tensor.cpu().numpy().T  # Shape: (samples, channels)
    sf.write(buffer, audio_np, sample_rate, format='WAV', subtype='PCM_16')

# Replace your existing load_model() function with this:
def get_model(model_type="standard", finetune_repo=None, finetune_checkpoint=None, base_repo=None):
    """Get model using the enhanced model manager with caching"""
    return load_model(model_type, finetune_repo, finetune_checkpoint, base_repo)

app = Flask(__name__)

# Initialize riff manager globally (add this near the top with other globals)
riff_manager = RiffManager()

# Global model storage
model_cache = {}
model_lock = threading.Lock()

@contextmanager
def resource_cleanup():
    """Context manager to ensure proper cleanup of GPU resources."""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

# def load_model():
#     """Load model if not already loaded."""
#     with model_lock:
#         if 'model' not in model_cache:
#             print("🔄 Loading stable-audio-open-small model...")
            
#             # Authenticate with HF
#             hf_token = os.getenv('HF_TOKEN')
#             if hf_token:
#                 login(token=hf_token)
#                 print(f"✅ HF authenticated ({hf_token[:10]}...)")
#             else:
#                 raise ValueError("HF_TOKEN environment variable required")
            
#             # Load model
#             model, config = get_pretrained_model("stabilityai/stable-audio-open-small")
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             model = model.to(device)
#             if device == "cuda":
#                 model = model.half()
            
#             model_cache['model'] = model
#             model_cache['config'] = config
#             model_cache['device'] = device
#             print(f"✅ Model loaded on {device}")
#             print(f"   Sample rate: {config['sample_rate']}")
#             print(f"   Sample size: {config['sample_size']}")
#             print(f"   Diffusion objective: {getattr(model, 'diffusion_objective', 'unknown')}")
        
#         return model_cache['model'], model_cache['config'], model_cache['device']

def extract_bpm(prompt):
    """Extract BPM from prompt for future loop processing."""
    # Look for patterns like "120bpm", "90 bpm", "140 BPM"
    bpm_match = re.search(r'(\d+)\s*bpm', prompt.lower())
    if bpm_match:
        return int(bpm_match.group(1))
    return None

def process_input_audio(audio_file, target_sr):
    """Process uploaded audio file into tensor format."""
    try:
        # Load audio file
        if hasattr(audio_file, 'read'):
            # File-like object from Flask
            audio_bytes = audio_file.read()
            audio_buffer = io.BytesIO(audio_bytes)
            waveform, sample_rate = torchaudio.load(audio_buffer)
        else:
            # File path
            waveform, sample_rate = torchaudio.load(audio_file)
        
        # Convert to mono if stereo (take average of channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Ensure we have stereo output (duplicate mono to stereo)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        print(f"📁 Processed input audio: {waveform.shape} at {target_sr}Hz")
        return sample_rate, waveform
    
    except Exception as e:
        raise ValueError(f"Failed to process input audio: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        model, config, device = load_model()
        return jsonify({
            "status": "healthy",
            "model_loaded": True,
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "model_info": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"],
                "diffusion_objective": getattr(model, 'diffusion_objective', 'unknown')
            }
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(e)
        }), 500
    
@app.route('/models/checkpoints', methods=['POST'])
def list_checkpoints():
    """
    List available checkpoints from a Hugging Face repository
    
    JSON Body:
    {
        "finetune_repo": "thepatch/jerry_grunge"
    }
    
    Returns:
    {
        "success": true,
        "repo": "thepatch/jerry_grunge",
        "checkpoints": [
            "jerry_un-encoded_epoch=32-step=2000.ckpt",
            "jerry_un-encoded_epoch=28-step=1800.ckpt",
            ...
        ]
    }
    """
    try:
        from huggingface_hub import list_repo_files
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        finetune_repo = data.get('finetune_repo', '').strip()
        if not finetune_repo:
            return jsonify({"error": "finetune_repo is required"}), 400
        
        # List all files in the repo
        try:
            all_files = list_repo_files(repo_id=finetune_repo, repo_type="model")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Could not access repository: {str(e)}",
                "hint": "Check that the repository exists and is public"
            }), 404
        
        # Filter for .ckpt files
        checkpoints = [f for f in all_files if f.endswith('.ckpt')]
        
        if not checkpoints:
            return jsonify({
                "success": False,
                "error": "No .ckpt checkpoint files found in repository",
                "repo": finetune_repo
            }), 404
        
        # Sort checkpoints (optional - by name or try to parse epoch/step)
        checkpoints.sort()
        
        return jsonify({
            "success": True,
            "repo": finetune_repo,
            "checkpoints": checkpoints,
            "count": len(checkpoints)
        })
        
    except Exception as e:
        print(f"Checkpoint listing error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
@app.route('/models/switch', methods=['POST'])
def switch_model():
    """
    Switch between standard and finetune models
    
    JSON Body:
    {
        "model_type": "standard",  // or "finetune"
        "finetune_repo": "S3Sound/am_saos1",  // required if finetune
        "finetune_checkpoint": "am_saos1_e18_s4800.ckpt"  // required if finetune
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        model_type = data.get('model_type', 'standard')
        finetune_repo = data.get('finetune_repo')
        finetune_checkpoint = data.get('finetune_checkpoint')
        
        if model_type not in ['standard', 'finetune']:
            return jsonify({"error": "model_type must be 'standard' or 'finetune'"}), 400
        
        if model_type == 'finetune':
            if not finetune_repo or not finetune_checkpoint:
                return jsonify({
                    "error": "finetune_repo and finetune_checkpoint required for finetune models"
                }), 400
        
        # Load the requested model (this will cache it)
        print(f"Switching to {model_type} model...")
        model, config, device = get_model(model_type, finetune_repo, finetune_checkpoint)
        
        # Get cache status
        cache_info = model_manager.list_loaded_models()
        
        return jsonify({
            "success": True,
            "model_type": model_type,
            "device": device,
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"]
            },
            "cache_status": cache_info,
            "message": f"Successfully switched to {model_type} model"
        })
        
    except Exception as e:
        print(f"Model switch error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/models/status', methods=['GET'])
def models_status():
    """Get status of loaded models"""
    try:
        cache_info = model_manager.list_loaded_models()
        
        # Get detailed info about each loaded model
        detailed_info = {}
        for model_key in cache_info['loaded_models']:
            if model_key in model_manager.model_cache:
                model_data = model_manager.model_cache[model_key]
                detailed_info[model_key] = {
                    "type": model_data["type"],
                    "source": model_data["source"],
                    "device": model_data["device"],
                    "sample_rate": model_data["config"]["sample_rate"],
                    "sample_size": model_data["config"]["sample_size"]
                }
        
        return jsonify({
            "cache_status": cache_info,
            "model_details": detailed_info,
            "max_models": model_manager.max_models,
            "cuda_available": torch.cuda.is_available()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/models/prompts', methods=['GET'])
def get_model_prompts():
    try:
        cache = model_manager.list_loaded_models()
        usage_order = cache.get("usage_order", [])
        # You may need direct access; if so, read model_manager.model_cache instead of list_loaded_models
        model_cache = getattr(model_manager, "model_cache", {})

        # --- selection inputs ---
        key = request.args.get("key")
        repo = request.args.get("repo")
        checkpoint = request.args.get("checkpoint")
        prefer = (request.args.get("prefer") or "active").lower()  # active | finetune | recent

        # Helper to resolve a cache_key by repo+checkpoint substrings
        def find_key_by_repo_ckpt(r, ck):
            for k in reversed(usage_order):  # prefer most recent
                entry = model_cache.get(k, {})
                if (r and r in str(entry.get("repo", ""))) and (ck and ck in str(entry.get("checkpoint", ""))):
                    return k
            return None

        # 1) explicit key
        if key and key in model_cache:
            selected_key = key
        # 2) repo+checkpoint
        elif repo and checkpoint:
            selected_key = find_key_by_repo_ckpt(repo, checkpoint)
        else:
            selected_key = None
            # 3) prefer active
            if prefer == "active":
                active_key = getattr(model_manager, "active_model_key", None)
                if active_key in model_cache:
                    selected_key = active_key
            # 4) prefer finetune
            if not selected_key and prefer in ("finetune",):
                for k in reversed(usage_order):
                    if model_cache.get(k, {}).get("type") == "finetune":
                        selected_key = k
                        break
            # 5) fallback to most recent
            if not selected_key and usage_order:
                selected_key = usage_order[-1]

        if not selected_key or selected_key not in model_cache:
            return jsonify({"success": True, "prompts": None, "message": "no model selected"}), 200

        entry = model_cache[selected_key]
        prompts = entry.get("prompts")

        # Lazy load prompts.json if this is a finetune and prompts missing
        try:
            if entry.get("type") == "finetune" and not prompts:
                # You added this helper earlier; falls back to None on failure
                prompts = model_manager._fetch_prompts_for_repo(
                    repo_id=entry.get("repo"),
                    checkpoint=entry.get("checkpoint")
                )
                entry["prompts"] = prompts
        except Exception:
            pass  # non-fatal

        # Normalize payload so JUCE can rely on consistent shape
        if not prompts:
            prompts = {"version": 1, "dice": {"generic": [], "drums": [], "instrumental": []}}

        payload = {
            "success": True,
            "model_key": selected_key,
            "type": entry.get("type"),
            "source": entry.get("repo") or entry.get("source"),
            "checkpoint": entry.get("checkpoint"),
            "prompts": prompts,
        }
        return jsonify(payload), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/models/clear-cache', methods=['POST'])
def clear_model_cache():
    """Clear all cached models"""
    try:
        model_manager.clear_cache()
        return jsonify({
            "success": True,
            "message": "Model cache cleared successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
# Test endpoint for the finetune
@app.route('/test/finetune', methods=['GET'])
def test_finetune():
    """Test endpoint to verify finetune loading"""
    try:
        print("Testing finetune loading...")
        
        # Try to load the finetune
        model, config, device = get_model(
            model_type="finetune",
            finetune_repo="S3Sound/am_saos1",
            finetune_checkpoint="am_saos1_e18_s4800.ckpt"
        )
        
        cache_info = model_manager.list_loaded_models()
        
        return jsonify({
            "success": True,
            "message": "Finetune loaded successfully",
            "device": device,
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"]
            },
            "cache_status": cache_info
        })
        
    except Exception as e:
        print(f"Finetune test failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
    
def sampler_kwargs_for_objective(model, client_overrides=None):
    """
    Choose sampler kwargs based on the model's diffusion objective.
    client_overrides: optional dict from the request payload to let users override.
    """
    obj = getattr(model, "diffusion_objective", None)
    kw = {}

    # Let client override explicitly if they pass something like {"sampler_type": "..."}
    if client_overrides:
        kw.update(client_overrides)

    if obj in ("rf_denoiser", "rectified_flow"):
        # For rf_* objectives, pingpong is the default for the post-adversarial rf_denoiser.
        # For rectified_flow, safest is to omit sampler_type and let stable-audio-tools choose defaults.
        if obj == "rf_denoiser":
            kw.setdefault("sampler_type", "pingpong")
        else:
            # Ensure we DON'T force pingpong on rectified_flow
            kw.pop("sampler_type", None)
    elif obj == "v":
        # v-objective typically uses k-diffusion samplers chosen inside generate_diffusion_cond
        kw.pop("sampler_type", None)
    else:
        # Unknown—be conservative
        kw.pop("sampler_type", None)

    return kw

@app.route('/generate', methods=['POST'])
def generate_audio():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        # ---- Model selection ----
        model_type = data.get('model_type', 'standard')
        finetune_repo = data.get('finetune_repo')
        finetune_checkpoint = data.get('finetune_checkpoint')

        # ---- Prompt ----
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400

        # ---- Load model (returns model, config dict, device) ----
        model, config, device = get_model(model_type, finetune_repo, finetune_checkpoint)

        # ---- Per-model timing derived from config ----
        sample_rate = int(config.get("sample_rate", 44100))
        model_sample_size = int(config.get("sample_size", 524288))
        model_seconds_max = max(1, model_sample_size // sample_rate)  # guard

        # Client may request seconds_total; default to model max
        req_seconds_total = data.get("seconds_total")
        if req_seconds_total is None:
            seconds_total = model_seconds_max
        else:
            # clamp to model capability
            try:
                seconds_total = int(req_seconds_total)
            except Exception:
                return jsonify({"error": "seconds_total must be an integer number of seconds"}), 400
            if seconds_total < 1:
                return jsonify({"error": "seconds_total must be >= 1"}), 400
            if seconds_total > model_seconds_max:
                # clamp + inform
                seconds_total = model_seconds_max

        # ---- Steps default depends on diffusion objective ----
        diffusion_objective = getattr(model, "diffusion_objective", None)
        steps = data.get('steps')
        if steps is None:
            # sensible defaults by objective
            if diffusion_objective == "rectified_flow":
                steps = 50
            else:
                steps = 8
        # validate steps
        if not isinstance(steps, int) or steps < 1 or steps > 250:
            return jsonify({"error": "steps must be integer between 1-250"}), 400

        # ---- CFG scale (same as before) ----
        cfg_scale = data.get('cfg_scale', 1.0)
        if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
            return jsonify({"error": "cfg_scale must be number between 0-20"}), 400

        negative_prompt = data.get('negative_prompt')
        return_format = data.get('return_format', 'file')
        if return_format not in ['file', 'base64']:
            return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400

        # ---- Seed ----
        seed = data.get('seed', -1)
        if seed != -1:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)

        # ---- Conditioning assembled from request + model config ----
        conditioning = [{
            "prompt": prompt,
            "seconds_total": seconds_total
        }]

        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_total": seconds_total
            }]

        # ---- Sampler kwargs based on objective (and client override if provided) ----
        client_sampler_overrides = {}
        if "sampler_type" in data:
            client_sampler_overrides["sampler_type"] = data["sampler_type"]
        skw = sampler_kwargs_for_objective(model, client_sampler_overrides)

        print(f"Generating with {model_type} model:")
        print(f"   Prompt: {prompt}")
        print(f"   Objective: {diffusion_objective}")
        print(f"   seconds_total: {seconds_total} (max {model_seconds_max})")
        print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        print(f"   Negative: {negative_prompt or 'None'}")

        # ---- Generation (unchanged except we pass the per-model sizes) ----
        start_time = time.time()
        with resource_cleanup():
            if device == "cuda":
                torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                output = generate_diffusion_cond(
                    model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    sample_size=model_sample_size,  # from model config
                    device=device,
                    seed=seed,
                    **skw
                )
        generation_time = time.time() - start_time

        # ---- Post (same as before) ----
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
        output_int16 = output.mul(32767).to(torch.int16).cpu()

        detected_bpm = extract_bpm(prompt)
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
            "sample_rate": sample_rate,
            "duration_seconds": seconds_total,
            "generation_time": round(generation_time, 2),
            "realtime_factor": round(seconds_total / max(generation_time, 1e-6), 2),
            "detected_bpm": detected_bpm,
            "device": device
        }
        if device == "cuda":
            memory_used = torch.cuda.memory_allocated() / 1e9
            memory_peak = torch.cuda.max_memory_allocated() / 1e9
            metadata["gpu_memory_used"] = round(memory_used, 2)
            metadata["gpu_memory_peak"] = round(memory_peak, 2)
            torch.cuda.reset_peak_memory_stats()

        print(f"✅ Generated in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")

        if return_format == "file":
            buffer = io.BytesIO()
            save_audio(buffer, output_int16, sample_rate)
            buffer.seek(0)
            filename = f"stable_audio_{seed}_{int(time.time())}.wav"
            del output, output_int16
            return send_file(buffer, mimetype='audio/wav', as_attachment=True, download_name=filename)

        else:
            buffer = io.BytesIO()
            save_audio(buffer, output_int16, sample_rate)
            audio_b64 = base64.b64encode(buffer.getvalue()).decode()
            del output, output_int16
            return jsonify({"success": True, "audio_base64": audio_b64, "metadata": metadata})

    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        import traceback; traceback.print_exc()
        with resource_cleanup():
            pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/generate/style-transfer', methods=['POST'])
def generate_style_transfer():
    """
    Generate audio using style transfer from input audio.
    
    Form Data:
    - audio_file: Audio file (WAV, MP3, etc.)
    - prompt: Text prompt for style guidance (required)
    - negative_prompt: Negative text prompt (optional)
    - style_strength: Style transfer strength 0.1-1.0 (optional, default 0.8)
    - steps: Diffusion steps (optional, default 8)
    - cfg_scale: CFG scale (optional, default 6.0)
    - seed: Random seed (optional, -1 for random)
    - return_format: "file" or "base64" (optional, default "file")
    """
    try:
        # Check if audio file is provided
        if 'audio_file' not in request.files:
            return jsonify({"error": "audio_file is required"}), 400
        
        audio_file = request.files['audio_file']
        if audio_file.filename == '':
            return jsonify({"error": "No audio file selected"}), 400
        
        # Parse form parameters
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        negative_prompt = request.form.get('negative_prompt')
        style_strength = float(request.form.get('style_strength', 0.8))
        steps = int(request.form.get('steps', 8))
        cfg_scale = float(request.form.get('cfg_scale', 6.0))
        seed = int(request.form.get('seed', -1))
        return_format = request.form.get('return_format', 'file')
        
        # Validate parameters
        if not (0.1 <= style_strength <= 1.0):
            return jsonify({"error": "style_strength must be between 0.1-1.0"}), 400
        
        if not isinstance(steps, int) or steps < 1 or steps > 100:
            return jsonify({"error": "steps must be integer between 1-100"}), 400
        
        if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
            return jsonify({"error": "cfg_scale must be number between 0-20"}), 400
        
        if return_format not in ['file', 'base64']:
            return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400
        
        # Load model
        model, config, device = load_model()
        
        # Process input audio
        input_sr, input_audio = process_input_audio(audio_file, config["sample_rate"])
        
        # Set seed for reproducibility
        if seed != -1:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        else:
            # Generate random seed for logging
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # Prepare conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_total": 12  # Fixed duration for this model
        }]
        
        # Prepare negative conditioning if provided
        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_total": 12
            }]
        
        print(f"🎨 Style transfer generation:")
        print(f"   Input audio: {input_audio.shape}")
        print(f"   Prompt: {prompt}")
        print(f"   Negative: {negative_prompt or 'None'}")
        print(f"   Style strength: {style_strength}")
        print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        
        # Generate audio with style transfer
        start_time = time.time()
        
        with resource_cleanup():
            # Clear GPU cache before generation
            if device == "cuda":
                torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                output = generate_diffusion_cond(
                    model,
                    steps=steps,
                    cfg_scale=cfg_scale,
                    conditioning=conditioning,
                    negative_conditioning=negative_conditioning,
                    sample_size=config["sample_size"],
                    sampler_type="pingpong",
                    device=device,
                    seed=seed,
                    init_audio=(config["sample_rate"], input_audio),
                    init_noise_level=style_strength
                )
            
            generation_time = time.time() - start_time
            
            # Post-process audio
            output = rearrange(output, "b d n -> d (b n)")  # (2, N) stereo
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
            output_int16 = output.mul(32767).to(torch.int16).cpu()
            
            # Extract BPM for future use
            detected_bpm = extract_bpm(prompt)
            
            # Prepare response metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "style_strength": style_strength,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sample_rate": config["sample_rate"],
                "duration_seconds": 12,
                "generation_time": round(generation_time, 2),
                "realtime_factor": round(12 / generation_time, 2),
                "detected_bpm": detected_bpm,
                "device": device,
                "input_audio_shape": list(input_audio.shape)
            }
            
            # Add memory info if CUDA
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_peak = torch.cuda.max_memory_allocated() / 1e9
                metadata["gpu_memory_used"] = round(memory_used, 2)
                metadata["gpu_memory_peak"] = round(memory_peak, 2)
                # Reset peak stats for next generation
                torch.cuda.reset_peak_memory_stats()
            
            print(f"✅ Style transfer complete in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")
            
            if return_format == "file":
                # Return as WAV file download
                buffer = io.BytesIO()
                save_audio(buffer, output_int16, config["sample_rate"])
                buffer.seek(0)
                
                filename = f"style_transfer_{seed}_{int(time.time())}.wav"
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                del input_audio
                
                return send_file(
                    buffer,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            
            else:  # base64 format
                # Return as JSON with base64 audio
                buffer = io.BytesIO()
                save_audio(buffer, output_int16, config["sample_rate"])
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                del input_audio
                
                return jsonify({
                    "success": True,
                    "audio_base64": audio_b64,
                    "metadata": metadata
                })
        
    except Exception as e:
        print(f"❌ Style transfer error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup on error as well
        with resource_cleanup():
            pass
            
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/generate/loop', methods=['POST'])
def generate_loop():
    try:
        # Detect content type and parse accordingly
        content_type = request.headers.get('Content-Type', '').lower()
        
        if 'application/json' in content_type:
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON body required"}), 400
            
            # ---- NEW: Model selection parameters ----
            model_type = data.get('model_type', 'standard')
            finetune_repo = data.get('finetune_repo')
            finetune_checkpoint = data.get('finetune_checkpoint')
            
            # Parse JSON parameters
            prompt = data.get('prompt')
            loop_type = data.get('loop_type', 'auto')
            bars = data.get('bars')
            style_strength = float(data.get('style_strength', 0.8))
            steps = int(data.get('steps', 8))
            cfg_scale = float(data.get('cfg_scale', 6.0))
            seed = int(data.get('seed', -1))
            return_format = data.get('return_format', 'file')
            
            audio_file = None
            
        else:
            # Form data input
            # ---- NEW: Model selection from form ----
            model_type = request.form.get('model_type', 'standard')
            finetune_repo = request.form.get('finetune_repo')
            finetune_checkpoint = request.form.get('finetune_checkpoint')
            
            prompt = request.form.get('prompt')
            loop_type = request.form.get('loop_type', 'auto')
            bars = request.form.get('bars')
            style_strength = float(request.form.get('style_strength', 0.8))
            steps = int(request.form.get('steps', 8))
            cfg_scale = float(request.form.get('cfg_scale', 6.0))
            seed = int(request.form.get('seed', -1))
            return_format = request.form.get('return_format', 'file')
            
            audio_file = request.files.get('audio_file')
        
        # Check for input audio file (style transfer mode)
        input_audio = None
        if audio_file and audio_file.filename != '':
            # ---- NEW: Load model with parameters ----
            model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
            input_sr, input_audio = process_input_audio(audio_file, config["sample_rate"])
        
        # Extract BPM from prompt
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be specified in prompt (e.g., '120bpm')"}), 400
        
        # Calculate bars if not specified
        if bars:
            bars = int(bars)
        else:
            # ---- NEW: Get timing from model config ----
            if input_audio is None:
                model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
            
            sample_rate = int(config.get("sample_rate", 44100))
            model_sample_size = int(config.get("sample_size", 524288))
            max_duration = model_sample_size / sample_rate
            
            seconds_per_beat = 60.0 / detected_bpm
            seconds_per_bar = seconds_per_beat * 4
            max_loop_duration = max_duration - 1.0  # Leave buffer
            
            possible_bars = [8, 4, 2, 1]
            bars = 1
            
            for bar_count in possible_bars:
                loop_duration = seconds_per_bar * bar_count
                if loop_duration <= max_loop_duration:
                    bars = bar_count
                    break
            
            print(f"🎵 Auto-selected {bars} bars ({bars * seconds_per_bar:.2f}s) for {detected_bpm} BPM")
        
        # Validate parameters
        if bars not in [1, 2, 4, 8]:
            return jsonify({"error": "bars must be 1, 2, 4, or 8"}), 400
        
        # Pre-calculate loop timing
        seconds_per_beat = 60.0 / detected_bpm
        seconds_per_bar = seconds_per_beat * 4
        calculated_loop_duration = seconds_per_bar * bars
        
        # Warn if loop might be too long
        if calculated_loop_duration > max_duration:
            print(f"⚠️  Warning: {bars} bars at {detected_bpm}bpm = {calculated_loop_duration:.2f}s (may exceed generated audio)")
            if calculated_loop_duration > max_duration + 1.0:
                bars = max(1, bars // 2)
                calculated_loop_duration = seconds_per_bar * bars
                print(f"🔧 Auto-reduced to {bars} bars ({calculated_loop_duration:.2f}s)")
        
        # Enhance prompt based on loop_type
        enhanced_prompt = prompt
        negative_prompt = ""
        
        if loop_type == "drums":
            if "drum" not in prompt.lower():
                enhanced_prompt = f"{prompt} drum loop"
            negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
        elif loop_type == "instruments":
            if "drum" in prompt.lower():
                enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
            negative_prompt = "drums, percussion, kick, snare, hi-hat"
        
        print(f"🔄 Loop generation:")
        print(f"   BPM: {detected_bpm}, Bars: {bars}")
        print(f"   Type: {loop_type}")
        print(f"   Model: {model_type}")
        print(f"   Enhanced prompt: {enhanced_prompt}")
        print(f"   Negative: {negative_prompt}")
        print(f"   Input audio: {'Yes' if input_audio is not None else 'No'}")
        
        # ---- NEW: Load model if not already loaded ----
        if input_audio is None:
            model, config, device = load_model(model_type, finetune_repo, finetune_checkpoint)
        
        # Set seed
        if seed != -1:
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
        
        # ---- NEW: Use per-model timing ----
        sample_rate = int(config.get("sample_rate", 44100))
        model_sample_size = int(config.get("sample_size", 524288))
        seconds_total = max(1, model_sample_size // sample_rate)
        
        # Prepare conditioning
        conditioning = [{
            "prompt": enhanced_prompt,
            "seconds_total": seconds_total
        }]
        
        negative_conditioning = None
        if negative_prompt:
            negative_conditioning = [{
                "prompt": negative_prompt,
                "seconds_total": seconds_total
            }]
        
        # ---- NEW: Dynamic sampler selection ----
        client_sampler_overrides = {}
        skw = sampler_kwargs_for_objective(model, client_sampler_overrides)
        
        print(f"   Using sampler kwargs: {skw}")
        
        # Generate audio
        start_time = time.time()
        
        with resource_cleanup():
            if device == "cuda":
                torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                if input_audio is not None:
                    # Style transfer mode
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=model_sample_size,
                        device=device,
                        seed=seed,
                        init_audio=(sample_rate, input_audio),
                        init_noise_level=style_strength,
                        **skw  # Use dynamic sampler
                    )
                else:
                    # Text-to-audio mode
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=model_sample_size,
                        device=device,
                        seed=seed,
                        **skw  # Use dynamic sampler
                    )
            
            generation_time = time.time() - start_time
            
            # Post-process audio
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
            
            # Calculate loop slice
            loop_duration = calculated_loop_duration
            loop_samples = int(loop_duration * sample_rate)
            
            # Safety check
            available_samples = output.shape[1]
            available_duration = available_samples / sample_rate
            
            if loop_samples > available_samples:
                print(f"⚠️  Requested loop ({loop_duration:.2f}s) exceeds available audio ({available_duration:.2f}s)")
                print(f"   Using maximum available: {available_duration:.2f}s")
                loop_samples = available_samples
                loop_duration = available_duration
            
            # Extract loop
            loop_output = output[:, :loop_samples]
            loop_output_int16 = loop_output.mul(32767).to(torch.int16).cpu()
            
            # Prepare metadata
            metadata = {
                "prompt": enhanced_prompt,
                "original_prompt": prompt,
                "negative_prompt": negative_prompt,
                "loop_type": loop_type,
                "detected_bpm": detected_bpm,
                "bars": bars,
                "loop_duration_seconds": round(loop_duration, 2),
                "calculated_duration_seconds": round(calculated_loop_duration, 2),
                "available_audio_seconds": round(available_duration, 2),
                "seconds_per_bar": round(seconds_per_bar, 2),
                "style_transfer": input_audio is not None,
                "style_strength": style_strength if input_audio is not None else None,
                "model_type": model_type,  # NEW
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sample_rate": sample_rate,
                "generation_time": round(generation_time, 2),
                "device": device
            }
            
            print(f"✅ Loop generated: {loop_duration:.2f}s ({bars} bars at {detected_bpm}bpm)")
            
            if return_format == "file":
                buffer = io.BytesIO()
                save_audio(buffer, loop_output_int16, sample_rate)
                buffer.seek(0)
                
                filename = f"loop_{loop_type}_{detected_bpm}bpm_{bars}bars_{seed}.wav"
                
                # Cleanup
                del output
                del loop_output
                del loop_output_int16
                if input_audio is not None:
                    del input_audio
                
                return send_file(
                    buffer,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            
            else:  # base64 format
                buffer = io.BytesIO()
                save_audio(buffer, loop_output_int16, sample_rate)
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Cleanup
                del output
                del loop_output
                del loop_output_int16
                if input_audio is not None:
                    del input_audio
                
                return jsonify({
                    "success": True,
                    "audio_base64": audio_b64,
                    "metadata": metadata
                })
    
    except Exception as e:
        print(f"❌ Loop generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        with resource_cleanup():
            pass
            
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed model information."""
    try:
        model, config, device = load_model()
        return jsonify({
            "model_name": "stabilityai/stable-audio-open-small",
            "device": device,
            "config": {
                "sample_rate": config["sample_rate"],
                "sample_size": config["sample_size"],
                "max_duration_seconds": 12,
                "diffusion_objective": getattr(model, 'diffusion_objective', 'unknown'),
                "io_channels": getattr(model, 'io_channels', 'unknown')
            },
            "supported_endpoints": {
                "/generate": "Text-to-audio generation",
                "/generate/style-transfer": "Audio-to-audio style transfer",
                "/generate/loop": "BPM-aware loop generation (text or style transfer)"
            },
            "supported_parameters": {
                "prompt": {"type": "string", "required": True},
                "steps": {"type": "int", "default": 8, "range": "1-100"},
                "cfg_scale": {"type": "float", "default": 6.0, "range": "0-20"},
                "negative_prompt": {"type": "string", "required": False},
                "seed": {"type": "int", "default": -1, "note": "-1 for random"},
                "return_format": {"type": "string", "default": "file", "options": ["file", "base64"]},
                "style_strength": {"type": "float", "default": 0.8, "range": "0.1-1.0", "note": "For style transfer"},
                "loop_type": {"type": "string", "default": "auto", "options": ["drums", "instruments", "auto"]},
                "bars": {"type": "int", "default": "auto", "options": [1, 2, 4, 8]}
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/generate/loop-with-riff', methods=['POST'])
def generate_loop_with_riff():
    """
    Generate loop using your personal riff library as style transfer input
    
    Form Data:
    - prompt: Text prompt with BPM (required, e.g., "aggressive techno 140bpm")
    - key: Musical key (required, e.g., "gsharp", "f", "csharp")
    - loop_type: "drums" or "instruments" (optional, default "instruments")
    - bars: Number of bars (optional, auto-calculated)
    - style_strength: Style transfer strength (optional, default 0.8)
    - steps: Diffusion steps (optional, default 8)
    - cfg_scale: CFG scale (optional, default 1.0)
    - seed: Random seed (optional, -1 for random)
    - return_format: "file" or "base64" (optional, default "file")
    """
    try:
        # Parse form parameters
        prompt = request.form.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        key = request.form.get('key')
        if not key:
            return jsonify({"error": "key is required (e.g., 'gsharp', 'f', 'csharp')"}), 400
        
        loop_type = request.form.get('loop_type', 'instruments')
        bars = request.form.get('bars')
        style_strength = float(request.form.get('style_strength', 0.8))
        steps = int(request.form.get('steps', 8))
        cfg_scale = float(request.form.get('cfg_scale', 1.0))
        seed = int(request.form.get('seed', -1))
        return_format = request.form.get('return_format', 'file')
        
        # Extract BPM from prompt
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be specified in prompt (e.g., '120bpm')"}), 400
        
        print(f"🎸 Riff-based generation request:")
        print(f"   Key: {key}")
        print(f"   Target BPM: {detected_bpm}")
        print(f"   Prompt: {prompt}")
        print(f"   Loop type: {loop_type}")
        
        # Get riff from library
        riff_temp_path = riff_manager.get_riff_for_style_transfer(key, detected_bpm)
        if not riff_temp_path:
            available_keys = riff_manager.get_available_keys()
            return jsonify({
                "error": f"No riffs available for key '{key}'",
                "available_keys": available_keys
            }), 400
        
        try:
            # Load model
            model, config, device = load_model()
            
            # Process the riff audio
            input_sr, input_audio = process_input_audio_from_path(riff_temp_path, config["sample_rate"])
            
            # Calculate bars if not specified
            if bars:
                bars = int(bars)
            else:
                seconds_per_beat = 60.0 / detected_bpm
                seconds_per_bar = seconds_per_beat * 4
                max_loop_duration = 10.0
                
                possible_bars = [8, 4, 2, 1]
                bars = 1
                
                for bar_count in possible_bars:
                    loop_duration = seconds_per_bar * bar_count
                    if loop_duration <= max_loop_duration:
                        bars = bar_count
                        break
                
                print(f"🎵 Auto-selected {bars} bars for {detected_bpm} BPM")
            
            # Enhance prompt based on loop_type
            enhanced_prompt = prompt
            negative_prompt = ""
            
            if loop_type == "drums":
                if "drum" not in prompt.lower():
                    enhanced_prompt = f"{prompt} drum loop"
                negative_prompt = "melody, harmony, pitched instruments, vocals, singing"
            elif loop_type == "instruments":
                if "drum" in prompt.lower():
                    enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
                negative_prompt = "drums, percussion, kick, snare, hi-hat"
            
            # Set seed
            if seed != -1:
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed(seed)
            else:
                seed = torch.randint(0, 2**32 - 1, (1,)).item()
                torch.manual_seed(seed)
                if device == "cuda":
                    torch.cuda.manual_seed(seed)
            
            # Prepare conditioning
            conditioning = [{
                "prompt": enhanced_prompt,
                "seconds_total": 12
            }]
            
            negative_conditioning = None
            if negative_prompt:
                negative_conditioning = [{
                    "prompt": negative_prompt,
                    "seconds_total": 12
                }]
            
            print(f"🎨 Starting style transfer with {key} riff...")
            
            # Generate audio with style transfer
            start_time = time.time()
            
            with resource_cleanup():
                if device == "cuda":
                    torch.cuda.empty_cache()
                
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=config["sample_size"],
                        sampler_type="pingpong",
                        device=device,
                        seed=seed,
                        init_audio=(config["sample_rate"], input_audio),
                        init_noise_level=style_strength
                    )
                
                generation_time = time.time() - start_time
                
                # Post-process audio (same as existing endpoint)
                output = rearrange(output, "b d n -> d (b n)")
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
                
                # Calculate loop slice
                sample_rate = config["sample_rate"]
                seconds_per_beat = 60.0 / detected_bpm
                seconds_per_bar = seconds_per_beat * 4
                loop_duration = seconds_per_bar * bars
                loop_samples = int(loop_duration * sample_rate)
                
                # Safety check
                available_samples = output.shape[1]
                available_duration = available_samples / sample_rate
                
                if loop_samples > available_samples:
                    loop_samples = available_samples
                    loop_duration = available_duration
                
                # Extract loop
                loop_output = output[:, :loop_samples]
                loop_output_int16 = loop_output.mul(32767).to(torch.int16).cpu()
                
                # Prepare metadata
                metadata = {
                    "prompt": enhanced_prompt,
                    "original_prompt": prompt,
                    "key": key,
                    "negative_prompt": negative_prompt,
                    "loop_type": loop_type,
                    "detected_bpm": detected_bpm,
                    "bars": bars,
                    "loop_duration_seconds": round(loop_duration, 2),
                    "style_transfer": True,
                    "style_strength": style_strength,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "seed": seed,
                    "sample_rate": sample_rate,
                    "generation_time": round(generation_time, 2),
                    "device": device,
                    "source": "personal_riff_library"
                }
                
                print(f"✅ Riff-based loop generated: {loop_duration:.2f}s ({bars} bars at {detected_bpm}bpm)")
                
                if return_format == "file":
                    buffer = io.BytesIO()
                    save_audio(buffer, loop_output_int16, sample_rate)
                    buffer.seek(0)
                    
                    filename = f"riff_{key}_{loop_type}_{detected_bpm}bpm_{bars}bars_{seed}.wav"
                    
                    # Cleanup
                    del output
                    del loop_output
                    del loop_output_int16
                    del input_audio
                    
                    return send_file(
                        buffer,
                        mimetype='audio/wav',
                        as_attachment=True,
                        download_name=filename
                    )
                
                else:  # base64 format
                    buffer = io.BytesIO()
                    save_audio(buffer, loop_output_int16, sample_rate)
                    audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    # Cleanup
                    del output
                    del loop_output
                    del loop_output_int16
                    del input_audio
                    
                    return jsonify({
                        "success": True,
                        "audio_base64": audio_b64,
                        "metadata": metadata
                    })
        
        finally:
            # Always clean up the temp riff file
            if os.path.exists(riff_temp_path):
                os.unlink(riff_temp_path)
                print(f"🧹 Cleaned up temp riff file")
    
    except Exception as e:
        print(f"❌ Riff generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/riffs/available', methods=['GET'])
def get_available_riffs():
    """Get information about available riffs"""
    try:
        keys = riff_manager.get_available_keys()
        riff_info = {}
        
        for key in keys:
            riffs = riff_manager.get_riffs_for_key(key)
            riff_info[key] = [
                {
                    "filename": riff["filename"],
                    "original_bpm": riff["original_bpm"],
                    "description": riff["description"]
                }
                for riff in riffs
            ]
        
        return jsonify({
            "available_keys": keys,
            "total_keys": len(keys),
            "riff_details": riff_info
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Helper function for loading audio from file path (add this too)
def process_input_audio_from_path(file_path, target_sr):
    """Process audio file from path into tensor format."""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Ensure stereo output
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        
        print(f"📁 Processed riff audio: {waveform.shape} at {target_sr}Hz")
        return sample_rate, waveform
    
    except Exception as e:
        raise ValueError(f"Failed to process riff audio: {str(e)}")
    


@app.route('/cleanup', methods=['POST'])
def manual_cleanup():
    """Manually trigger GPU cleanup."""
    try:
        with resource_cleanup():
            pass
        return jsonify({"message": "GPU cleanup completed successfully"})
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Pre-load model on startup
    print("🚀 Starting Enhanced Stable Audio API...")
    try:
        load_model()
        print("✅ Model pre-loaded successfully")
    except Exception as e:
        print(f"❌ Failed to pre-load model: {e}")
        print("Will attempt to load on first request...")
    
    # Run server
    app.run(host='0.0.0.0', port=8005, debug=False)