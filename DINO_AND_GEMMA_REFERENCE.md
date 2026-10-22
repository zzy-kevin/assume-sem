# DINOv3 & Gemma-4 12B Reference Guide

This document records the exact configurations, virtual environment locations, model paths, and code patterns for running **DINOv3** vision embeddings and **Gemma-4 12B** local LLM / VLM inference.

---

## 1. Virtual Environments & Hardware

### System Hardware
- **GPU**: NVIDIA GeForce RTX 5070 Ti (16 GB VRAM, Compute Capability 12.0)
- **CUDA**: 13.2 compatible PyTorch and CUDA llama.cpp backend

### Virtual Environment Locations

| Purpose | Virtual Environment Path | Python Executable | Key Packages |
| :--- | :--- | :--- | :--- |
| **DINOv3 & SEM Processing** | `D:\UOFT\Grad1_local\DINO_SEM\.venv` | `D:\UOFT\Grad1_local\DINO_SEM\.venv\Scripts\python.exe` | `torch==2.12.1+cu132`, `transformers==5.13.0`, `accelerate==1.14.0`, `pillow==12.2.0`, `botorch`, `gpytorch` |
| **Gemma-4 12B Inference (`llama_cpp`)** | `D:\UOFT\antigravity_projects\chexpert_vlm_rag\.venv` | `D:\UOFT\antigravity_projects\chexpert_vlm_rag\.venv\Scripts\python.exe` | `llama-cpp-python` (built with CUDA offload), `PIL`, `numpy` |

> [!NOTE]
> `llama_cpp` is installed in `D:\UOFT\antigravity_projects\chexpert_vlm_rag\.venv`. When running Gemma-4 12B GGUF models, use this python interpreter or install `llama-cpp-python` into the local project environment with CUDA support.

---

## 2. DINOv3 Execution Protocol

### Model Weights & HuggingFace Cache
- **Model Hub ID**: `facebook/dinov3-vits16plus-pretrain-lvd1689m` (default ViT-S/16+ model with 384 embedding dimension).
- **Local HuggingFace Cache** (`C:\Users\Kevin\.cache\huggingface\hub`):
  - `models--facebook--dinov3-vits16plus-pretrain-lvd1689m` (ViT-S/16+, dim 384)
  - `models--facebook--dinov3-vitb16-pretrain-lvd1689m` (ViT-B/16, dim 768)
  - `models--facebook--dinov3-vith16plus-pretrain-lvd1689m` (ViT-H/16+, dim 1280)

### SEM Micrograph Preprocessing
1. **Raw Dimensions**: SEM images are `1920 x 1275` (W, H).
2. **Databar Crop**: Bottom ~91px contains the microscope instrument bar. Cropping `(0, 0, 1920, 1184)` isolates pure sample morphology.
3. **Channels & Normalization**: Grayscale (`"L"`) replicated to 3 channels `(3, H, W)` in `[0, 1]`, then normalized by `AutoImageProcessor.from_pretrained(MODEL_ID).image_mean` and `image_std`.
4. **Resolution Variants** (via `exp_common.py`):
   - Full resolution (`DOWNSCALE=1.0`): `1920 x 1184` -> patch grid `120 x 74` = 8,880 patches.
   - Half resolution (`DINO_VARIANT=ds50 DINO_DOWNSCALE=0.5`): `960 x 592` -> patch grid `60 x 37` = 2,220 patches.

### DINOv3 Token Disassembly
Running `model(pixel_values=x).last_hidden_state` produces `(B, total_tokens, 384)`.
- `num_special = total_tokens - num_patches` (typically 5 tokens):
  - `special[:, 0, :]`: `[CLS]` global summary token.
  - `special[:, 1:, :]`: Register tokens.
- `patch = out[:, num_special:, :]`: Per-patch dense embeddings `(B, NUM_PATCHES, 384)`.
- `pooled = patch.mean(dim=1)`: Mean-pooled vector `(B, 384)` for retrieval / classification.

---

## 3. Gemma-4 12B Model Protocol

### Model File Locations
- **Gemma-4 12B GGUF Model**:
  `D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf` (7.12 GB)
- **Multimodal Vision Projector (for VLM)**:
  `D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\mmproj-BF16.gguf` (175 MB)

### Calling Gemma-4 12B with Custom Prompt (Text)
```python
from llama_cpp import Llama

llm = Llama(
    model_path=r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,      # Offload all layers to RTX 5070 Ti (16GB VRAM)
    main_gpu=0,
    split_mode=0,
    n_batch=512,
    n_ubatch=512,
    seed=42,
    verbose=False,
)

completion = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an expert materials science AI assistant."},
        {"role": "user", "content": "Analyze the following experimental synthesis observations..."}
    ],
    max_tokens=1024,
    temperature=0.1,
)
response_text = completion["choices"][0]["message"]["content"]
```

### Grammar-Constrained JSON Output
Using `LlamaGrammar.from_json_schema`:
```python
import json
from llama_cpp import LlamaGrammar

schema = {
    "type": "object",
    "properties": {
        "theme": {"type": "string"},
        "correlation": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"}
    },
    "required": ["theme", "correlation", "confidence"],
    "additionalProperties": False,
}
grammar = LlamaGrammar.from_json_schema(json.dumps(schema, ensure_ascii=False), verbose=False)

completion = llm.create_chat_completion(
    messages=[...],
    grammar=grammar,
    temperature=0.1,
)
```

### Multimodal / VLM Calling with Images
```python
import base64
from pathlib import Path
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler

mmproj = r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\mmproj-BF16.gguf"
model = r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf"

handler = Gemma4ChatHandler(clip_model_path=mmproj, verbose=False)
llm = Llama(model_path=model, chat_handler=handler, n_ctx=4096, n_gpu_layers=-1, verbose=False)

b64_image = base64.b64encode(Path("sample.jpg").read_bytes()).decode("ascii")
data_uri = f"data:image/jpeg;base64,{b64_image}"

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Describe the visible microstructure in the SEM image."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the key morphology features:"},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]
        }
    ],
    max_tokens=512,
    temperature=0.1,
)
```

---

## 4. Reference Scripts in `reference_scripts/`

The following scripts have been copied into `reference_scripts/` for direct reference:

| Script Name | Source Location | Description |
| :--- | :--- | :--- |
| `embed_dinov3.py` | `D:\UOFT\Grad1_local\DINO_SEM\embed_dinov3.py` | Full batch/streaming DINOv3 feature extraction, mmap `.npy` generation (`patch.npy`, `special.npy`, `pooled.npy`). |
| `exp_common.py` | `D:\UOFT\Grad1_local\DINO_SEM\exp_common.py` | Image geometry calculations, dataset joins, 5-fold CV split definition, metrics. |
| `run_gemma_structured_synthesis.py` | `D:\UOFT\Grad1_local\DINO_SEM\archetypal\run_gemma_structured_synthesis.py` | Grammar-constrained structured JSON output using Gemma-4 12B and `LlamaGrammar`. |
| `cross_magnification_synthesis_llm.py` | `D:\UOFT\Grad1_local\DINO_SEM\archetypal\cross_magnification_synthesis_llm.py` | Complex evidence bundling across magnification scales and automated prompt synthesis. |
| `vlm_describe_overpotential_archetypes.py` | `D:\UOFT\Grad1_local\DINO_SEM\archetypal\vlm_describe_overpotential_archetypes.py` | Multimodal VLM evaluation passing up to 10 image inputs via `Gemma4ChatHandler` and `mmproj-BF16.gguf`. |
| `quickstart_examples.py` | Newly created modular helper | Self-contained functions for DINOv3 feature extraction and Gemma-4 12B text / VLM inference. |
