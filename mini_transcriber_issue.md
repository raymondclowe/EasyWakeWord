# mini_transcriber Issues - Critical Bugs in app.py

## Problem Summary

The mini_transcriber app.py has critical indentation and variable scoping bugs that cause:
1. All transcription requests to fail with HTTP 500
2. Model never loads because first transcription fails
3. Health endpoint misleadingly reports server is running but model never loads

**Note**: These bugs exist in the cloned repo and prevent mini_transcriber from working.

## Issues Found

### 1. Model Preloading Not Working

The model preloading hooks don't work:

```python
# These don't trigger model loading
try:
    app.before_serving(lambda: load_model(DEFAULT_MODEL))
except Exception:
    try:
        app.before_first_request(lambda: load_model(DEFAULT_MODEL))
    except Exception:
        pass
```

**Result**: 
- Server starts successfully
- `/health` endpoint returns `{"model_loaded": false, "loaded_models": [], "status": "loading"}`
- Model never loads because no transcription succeeds

### 2. Transcription Endpoint Has Critical Bugs

**Error**: `UnboundLocalError: local variable 'b64' referenced before assignment` at line 72 (approx)

**The actual broken code** (from app.py lines 36-76):

```python
@app.route("/transcribe", methods=["POST"])
def transcribe():
    file_path = None
    model_name = request.args.get('model') or request.form.get('model')
    if not model_name and request.is_json:
        payload = request.get_json(silent=True) or {}
        model_name = payload.get('model')
    if not model_name:
        model_name = DEFAULT_MODEL

    if 'file' in request.files:
        f = request.files['file']
        tmp = Path('tmp_upload.wav')
        f.save(tmp)
        file_path = str(tmp)
    else:
        # try JSON payload or form field containing base64 audio
        b64 = None  # ← Variable defined inside else block
        mimetype = None
        filename = None
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            language = request.args.get('language') or request.form.get('language')  # ← language defined in nested if
            # Also support JSON body with 'model' and 'language' field
            b64 = payload.get('b64') or payload.get('audio')
            mimetype = payload.get('mimetype')
            filename = payload.get('filename')
        if not language:  # ← ERROR: language not defined if not request.is_json
                language = payload.get('language')  # ← ERROR: payload not defined outside if block
    if not b64:  # ← ERROR: b64 not defined if 'file' in request.files (took first branch)
        b64 = request.form.get('b64') or request.form.get('audio')
        if not language:  # ← ERROR: language not defined
            language = 'en'
    if not language:  # ← ERROR: language not defined
        language = 'en'
    mimetype = mimetype or request.form.get('mimetype')  # ← ERROR: mimetype not defined if 'file' path
    filename = filename or request.form.get('filename')  # ← ERROR: filename not defined if 'file' path
```

**Multiple scoping errors**:
1. `b64`, `language`, `payload`, `mimetype`, `filename` defined inside nested if/else blocks
2. Variables used outside their definition scope
3. File upload path (`if 'file' in request.files`) defines NO variables except `file_path`
4. Then code tries to use `b64`, `language`, `mimetype`, `filename` which were never defined

**Result**: Even basic file uploads fail with `UnboundLocalError`

### 3. Testing Failure

When sending a standard multipart file upload (the working path):

```bash
curl -X POST http://localhost:8085/transcribe \
  -F "file=@audio.wav" \
  -F "model=tiny" \
  -F "language=en"
```

Server returns: **HTTP 500 Internal Server Error**

## Reproduction

1. Start mini_transcriber:
   ```bash
   python app.py
   ```

2. Check health (works but shows model not loaded):
   ```bash
   curl http://localhost:8085/health
   # Returns: {"loaded_models":[],"model_loaded":false,"status":"loading"}
   ```

3. Try transcription:
   ```bash
   curl -X POST http://localhost:8085/transcribe \
     -F "file=@test.wav" \
     -F "model=tiny" \
     -F "language=en"
   ```
   
4. Result: HTTP 500 error

5. Check server log:
   ```
   [2025-11-30 08:23:20,481] ERROR in app: Exception on /transcribe [POST]
   Traceback (most recent call last):
     ...
     File "app.py", line 72, in transcribe
       if not b64:
   UnboundLocalError: local variable 'b64' referenced before assignment
   ```

## Expected Behavior

1. Model should preload when server starts
2. `/health` should return `{"model_loaded": true, ...}` after model loads
3. Transcription requests should succeed
4. Variables should be properly scoped

## Suggested Fixes

### Fix 1: Model Preloading

Use Flask's application context or eagerly load on startup:

```python
# After creating app
print("Preloading default Whisper model...")
load_model(DEFAULT_MODEL)
print("Model loaded, ready to serve requests")
```

### Fix 2: Fix Variable Scoping in `/transcribe`

Move variable initialization outside nested blocks:

```python
@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Initialize ALL variables at function scope
    file_path = None
    b64 = None
    language = 'en'  # Default
    mimetype = None
    filename = None
    payload = {}
    
    # Get model name
    model_name = request.args.get('model') or request.form.get('model')
    if not model_name and request.is_json:
        payload = request.get_json(silent=True) or {}
        model_name = payload.get('model')
    if not model_name:
        model_name = DEFAULT_MODEL
    
    # Get language (check all sources)
    language = request.args.get('language') or request.form.get('language') or 'en'
    
    # Handle file upload
    if 'file' in request.files:
        f = request.files['file']
        tmp = Path('tmp_upload.wav')
        f.save(tmp)
        file_path = str(tmp)
    else:
        # Handle base64/JSON payload
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            b64 = payload.get('b64') or payload.get('audio')
            mimetype = payload.get('mimetype')
            filename = payload.get('filename')
            language = payload.get('language') or language
        
        if not b64:
            b64 = request.form.get('b64') or request.form.get('audio')
            
        # ... rest of base64 handling ...
```

## Impact

This bug prevents mini_transcriber from being used as a bundled transcription service in downstream projects like EasyWakeWord.

## Environment

- Python 3.10
- Flask 3.1.2
- Windows 11 (but likely affects all platforms)
- Installing from: `git clone https://github.com/raymondclowe/mini_transcriber.git`
