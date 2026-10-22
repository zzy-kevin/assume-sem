"""Quickstart Reference Examples for DINOv3 and Gemma-4 12B.

Environments:
1. DINOv3 environment:
   D:\\UOFT\\Grad1_local\\DINO_SEM\\.venv\\Scripts\\python.exe
   Has torch (CUDA 13.2, RTX 5070 Ti), transformers 5.13.0, torchvision, pillow.

2. Gemma-4 12B environment:
   D:\\UOFT\\antigravity_projects\\chexpert_vlm_rag\\.venv\\Scripts\\python.exe
   Has llama_cpp (compiled with CUDA support, RTX 5070 Ti).
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image

# =====================================================================
# 1. DINOv3 Feature Extraction
# =====================================================================

DINO_MODEL_ID = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
# Also pre-cached on machine:
#   facebook/dinov3-vitb16-pretrain-lvd1689m
#   facebook/dinov3-vith16plus-pretrain-lvd1689m


def extract_dinov3_features(
    image_path: str | Path,
    crop_size: tuple[int, int] = (1920, 1184), # (W, H) without databar
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Extracts patch, special (CLS + registers), and pooled embeddings using DINOv3 ViT-S/16+.
    """
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(DINO_MODEL_ID)
    mean = np.array(proc.image_mean, dtype=np.float32)
    std = np.array(proc.image_std, dtype=np.float32)

    model = AutoModel.from_pretrained(DINO_MODEL_ID, dtype=torch.float16, attn_implementation="sdpa")
    model.eval().to(device)

    # 1. Preprocessing: Greyscale -> crop -> 3-channel replicate -> normalize
    img = Image.open(image_path).convert("L")
    w, h = crop_size
    img = img.crop((0, 0, w, h))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.repeat(arr[None, :, :], 3, axis=0)  # (3, H, W)
    arr = (arr - mean[:, None, None]) / std[:, None, None]

    tensor_input = torch.from_numpy(arr[None, ...]).to(device, dtype=torch.float16)

    # 2. Forward pass
    with torch.no_grad():
        out = model(pixel_values=tensor_input).last_hidden_state  # (B, tokens, D)

    # ViT-S/16+ patch grid: (1184//16, 1920//16) = (74, 120) -> 8880 patches
    num_patches = (h // 16) * (w // 16)
    total_tokens = out.shape[1]
    num_special = total_tokens - num_patches  # CLS + register tokens (typically 5)

    special_tokens = out[:, :num_special, :]  # [:, 0] is CLS token
    patch_tokens = out[:, num_special:, :]    # (B, num_patches, 384)
    pooled_token = patch_tokens.mean(dim=1)   # (B, 384)

    return {
        "special": special_tokens.cpu().numpy(),
        "patch": patch_tokens.cpu().numpy(),
        "pooled": pooled_token.cpu().numpy(),
        "grid_shape": (h // 16, w // 16),
        "num_special": num_special,
    }


# =====================================================================
# 2. Gemma-4 12B Local Text Inference (Custom Prompt & Grammar)
# =====================================================================

GEMMA_MODEL_PATH = r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\gemma-4-12b-it-Q4_K_M.gguf"
MMPROJ_PATH = r"D:\UOFT\antigravity_projects\chexpert_vlm_rag\models\mmproj-BF16.gguf"


def call_gemma4_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.1,
    json_schema: dict | None = None
) -> str:
    """
    Run Gemma-4 12B GGUF with GPU offload via llama-cpp-python.
    """
    from llama_cpp import Llama
    import json

    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=-1,  # Offload all layers to GPU (RTX 5070 Ti 16GB)
        main_gpu=0,
        split_mode=0,
        n_batch=512,
        n_ubatch=512,
        seed=42,
        verbose=False,
    )

    grammar = None
    if json_schema is not None:
        from llama_cpp import LlamaGrammar
        grammar = LlamaGrammar.from_json_schema(json.dumps(json_schema, ensure_ascii=False), verbose=False)

    try:
        completion = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            grammar=grammar,
        )
        return completion["choices"][0]["message"]["content"]
    finally:
        llm.reset()
        llm.close()


# =====================================================================
# 3. Gemma-4 12B Multimodal / VLM Inference (Images + Text)
# =====================================================================

def call_gemma4_vlm(
    system_prompt: str,
    user_text: str,
    image_paths: list[str | Path],
    max_tokens: int = 512,
    temperature: float = 0.1
) -> str:
    """
    Run multimodal Gemma-4 12B using Gemma4ChatHandler and mmproj-BF16.gguf.
    """
    import base64
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Gemma4ChatHandler

    handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH, verbose=False)
    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        chat_handler=handler,
        n_ctx=4096,
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode=0,
        n_batch=512,
        n_ubatch=512,
        seed=42,
        verbose=False,
    )

    user_content = [{"type": "text", "text": user_text}]
    for idx, img_path in enumerate(image_paths, 1):
        b64_data = base64.b64encode(Path(img_path).read_bytes()).decode("ascii")
        data_uri = f"data:image/jpeg;base64,{b64_data}"
        user_content.append({"type": "text", "text": f"Image {idx}:"})
        user_content.append({"type": "image_url", "image_url": {"url": data_uri}})

    try:
        completion = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion["choices"][0]["message"]["content"]
    finally:
        llm.reset()
        llm.close()
