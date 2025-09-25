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

def load_model():
    """Load model if not already loaded."""
    with model_lock:
        if 'model' not in model_cache:
            print("🔄 Loading stable-audio-open-small model...")
            
            # Authenticate with HF
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                login(token=hf_token)
                print(f"✅ HF authenticated ({hf_token[:10]}...)")
            else:
                raise ValueError("HF_TOKEN environment variable required")
            
            # Load model
            model, config = get_pretrained_model("stabilityai/stable-audio-open-small")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            if device == "cuda":
                model = model.half()
            
            model_cache['model'] = model
            model_cache['config'] = config
            model_cache['device'] = device
            print(f"✅ Model loaded on {device}")
            print(f"   Sample rate: {config['sample_rate']}")
            print(f"   Sample size: {config['sample_size']}")
            print(f"   Diffusion objective: {getattr(model, 'diffusion_objective', 'unknown')}")
        
        return model_cache['model'], model_cache['config'], model_cache['device']

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

@app.route('/generate', methods=['POST'])
def generate_audio():
    """
    Generate audio from text prompt.
    
    JSON Body:
    {
        "prompt": "lo-fi hip-hop beat with pianos 90bpm",  // required
        "steps": 8,                                        // optional, default 8
        "cfg_scale": 1.0,                                  // optional, default 1.0
        "negative_prompt": "distorted harsh noise",        // optional
        "seed": 12345,                                     // optional, -1 for random
        "return_format": "file"                            // "file" or "base64", default "file"
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400
        
        # Parse and validate parameters
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({"error": "prompt is required"}), 400
        
        steps = data.get('steps', 8)  # Default to 8 like HF example
        cfg_scale = data.get('cfg_scale', 1.0)
        negative_prompt = data.get('negative_prompt')
        seed = data.get('seed', -1)
        return_format = data.get('return_format', 'file')
        
        # Validate parameters
        if not isinstance(steps, int) or steps < 1 or steps > 250:
            return jsonify({"error": "steps must be integer between 1-100"}), 400
        
        if not isinstance(cfg_scale, (int, float)) or cfg_scale < 0 or cfg_scale > 20:
            return jsonify({"error": "cfg_scale must be number between 0-20"}), 400
        
        if return_format not in ['file', 'base64']:
            return jsonify({"error": "return_format must be 'file' or 'base64'"}), 400
        
        # Load model
        model, config, device = load_model()
        
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
        
        print(f"🎵 Generating audio:")
        print(f"   Prompt: {prompt}")
        print(f"   Negative: {negative_prompt or 'None'}")
        print(f"   Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        
        # Generate audio with resource cleanup
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
                    seed=seed
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
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "sample_rate": config["sample_rate"],
                "duration_seconds": 12,
                "generation_time": round(generation_time, 2),
                "realtime_factor": round(12 / generation_time, 2),
                "detected_bpm": detected_bpm,
                "device": device
            }
            
            # Add memory info if CUDA
            if device == "cuda":
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_peak = torch.cuda.max_memory_allocated() / 1e9
                metadata["gpu_memory_used"] = round(memory_used, 2)
                metadata["gpu_memory_peak"] = round(memory_peak, 2)
                # Reset peak stats for next generation
                torch.cuda.reset_peak_memory_stats()
            
            print(f"✅ Generated in {generation_time:.2f}s ({metadata['realtime_factor']:.1f}x RT)")
            
            if return_format == "file":
                # Return as WAV file download
                buffer = io.BytesIO()
                torchaudio.save(buffer, output_int16, config["sample_rate"], format="wav")
                buffer.seek(0)
                
                filename = f"stable_audio_{seed}_{int(time.time())}.wav"
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                
                return send_file(
                    buffer,
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=filename
                )
            
            else:  # base64 format
                # Return as JSON with base64 audio
                buffer = io.BytesIO()
                torchaudio.save(buffer, output_int16, config["sample_rate"], format="wav")
                audio_b64 = base64.b64encode(buffer.getvalue()).decode()
                
                # Explicitly clean up tensors before returning
                del output
                del output_int16
                
                return jsonify({
                    "success": True,
                    "audio_base64": audio_b64,
                    "metadata": metadata
                })
        
    except Exception as e:
        print(f"❌ Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Ensure cleanup on error as well
        with resource_cleanup():
            pass
            
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

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
                torchaudio.save(buffer, output_int16, config["sample_rate"], format="wav")
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
                torchaudio.save(buffer, output_int16, config["sample_rate"], format="wav")
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
    """
    Generate BPM-aware loop from audio with smart bar calculation.
    Automatically selects optimal bar count based on BPM to fit within ~10s.
    Supports both text-to-audio and style transfer modes.

    Generate BPM-aware loop with smart bar calculation.
    NOW ACCEPTS BOTH JSON AND FORM DATA for better API consistency
    
    Form Data (Style Transfer):
    - audio_file: Input audio file (optional)
    - prompt: Text prompt with BPM (required, e.g., "techno 128bpm")
    - loop_type: "drums" or "instruments" (optional, affects negative prompting)
    - bars: Number of bars to extract (optional, auto-calculated if not provided)
    - style_strength: Style transfer strength if audio_file provided (optional, default 0.8)
    - steps: Diffusion steps (optional, default 8)
    - cfg_scale: CFG scale (optional, default 6.0)
    - seed: Random seed (optional, -1 for random)
    - return_format: "file" or "base64" (optional, default "file")
    
    Smart Bar Selection Logic:
    - 74 BPM: Auto-selects 2 bars (~6.5s)
    - 128 BPM: Auto-selects 4 bars (~7.5s)  
    - 170 BPM: Auto-selects 8 bars (~11.3s)
    - Always chooses largest power-of-2 that fits in ~10s
    """
    try:
        # Detect content type and parse accordingly
        content_type = request.headers.get('Content-Type', '').lower()
        
        if 'application/json' in content_type:
            # JSON input (like /generate endpoint)
            data = request.get_json()
            if not data:
                return jsonify({"error": "JSON body required"}), 400
            
            # Parse JSON parameters
            prompt = data.get('prompt')
            loop_type = data.get('loop_type', 'auto')
            bars = data.get('bars')
            style_strength = float(data.get('style_strength', 0.8))
            steps = int(data.get('steps', 8))
            cfg_scale = float(data.get('cfg_scale', 6.0))
            seed = int(data.get('seed', -1))
            return_format = data.get('return_format', 'file')
            
            # Note: JSON requests cannot include audio files
            audio_file = None
            
        else:
            # Form data input (existing behavior)
            prompt = request.form.get('prompt')
            loop_type = request.form.get('loop_type', 'auto')
            bars = request.form.get('bars')
            style_strength = float(request.form.get('style_strength', 0.8))
            steps = int(request.form.get('steps', 8))
            cfg_scale = float(request.form.get('cfg_scale', 6.0))
            seed = int(request.form.get('seed', -1))
            return_format = request.form.get('return_format', 'file')
            
            # Form data can include audio files
            audio_file = request.files.get('audio_file')
        
        # Check for input audio file (optional for style transfer)
        input_audio = None
        audio_file = request.files.get('audio_file')
        if audio_file and audio_file.filename != '':
            model, config, device = load_model()
            input_sr, input_audio = process_input_audio(audio_file, config["sample_rate"])
        
        # Extract BPM from prompt
        detected_bpm = extract_bpm(prompt)
        if not detected_bpm:
            return jsonify({"error": "BPM must be specified in prompt (e.g., '120bpm')"}), 400
        
        # Calculate bars if not specified - smart selection based on BPM and model output length
        if bars:
            bars = int(bars)
        else:
            # Calculate how long different bar counts would be
            seconds_per_beat = 60.0 / detected_bpm
            seconds_per_bar = seconds_per_beat * 4  # 4/4 time signature
            
            # Available audio is ~11 seconds (leave 1 second buffer for safety)
            max_loop_duration = 10.0
            
            # Find the largest power-of-2 bar count that fits
            possible_bars = [8, 4, 2, 1]
            bars = 1  # fallback
            
            for bar_count in possible_bars:
                loop_duration = seconds_per_bar * bar_count
                if loop_duration <= max_loop_duration:
                    bars = bar_count
                    break
            
            print(f"🎵 Auto-selected {bars} bars ({bars * seconds_per_bar:.2f}s) for {detected_bpm} BPM")
        
        # Validate parameters
        if bars not in [1, 2, 4, 8]:
            return jsonify({"error": "bars must be 1, 2, 4, or 8"}), 400
        
        # Pre-calculate loop timing for validation
        seconds_per_beat = 60.0 / detected_bpm
        seconds_per_bar = seconds_per_beat * 4
        calculated_loop_duration = seconds_per_bar * bars
        
        # Warn if loop might be too long for the generated audio
        if calculated_loop_duration > 11.0:
            print(f"⚠️  Warning: {bars} bars at {detected_bpm}bpm = {calculated_loop_duration:.2f}s (may exceed generated audio)")
            # Auto-reduce bars if way too long
            if calculated_loop_duration > 12.0:
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
                # Remove drum references for instrumental loops
                enhanced_prompt = prompt.replace("drum", "").replace("drums", "").strip()
            negative_prompt = "drums, percussion, kick, snare, hi-hat"
        
        print(f"🔄 Loop generation:")
        print(f"   BPM: {detected_bpm}, Bars: {bars}")
        print(f"   Type: {loop_type}")
        print(f"   Enhanced prompt: {enhanced_prompt}")
        print(f"   Negative: {negative_prompt}")
        print(f"   Input audio: {'Yes' if input_audio is not None else 'No'}")
        
        # Load model if not already loaded
        model, config, device = load_model()
        
        # Set seed for reproducibility
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
        
        # Generate audio
        start_time = time.time()
        
        with resource_cleanup():
            if device == "cuda":
                torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                # Use style transfer if input audio provided, otherwise text-to-audio
                if input_audio is not None:
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
                else:
                    output = generate_diffusion_cond(
                        model,
                        steps=steps,
                        cfg_scale=cfg_scale,
                        conditioning=conditioning,
                        negative_conditioning=negative_conditioning,
                        sample_size=config["sample_size"],
                        sampler_type="pingpong",
                        device=device,
                        seed=seed
                    )
            
            generation_time = time.time() - start_time
            
            # Post-process audio
            output = rearrange(output, "b d n -> d (b n)")
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1)
            
            # Calculate loop slice based on pre-calculated duration
            sample_rate = config["sample_rate"]
            loop_duration = calculated_loop_duration
            loop_samples = int(loop_duration * sample_rate)
            
            # Safety check: don't exceed available audio
            available_samples = output.shape[1]
            available_duration = available_samples / sample_rate
            
            if loop_samples > available_samples:
                print(f"⚠️  Requested loop ({loop_duration:.2f}s) exceeds available audio ({available_duration:.2f}s)")
                print(f"   Using maximum available: {available_duration:.2f}s")
                loop_samples = available_samples
                loop_duration = available_duration
            
            # Extract loop from the beginning (where the beat usually starts cleanest)
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
                torchaudio.save(buffer, loop_output_int16, sample_rate, format="wav")
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
                torchaudio.save(buffer, loop_output_int16, sample_rate, format="wav")
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
                    torchaudio.save(buffer, loop_output_int16, sample_rate, format="wav")
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
                    torchaudio.save(buffer, loop_output_int16, sample_rate, format="wav")
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