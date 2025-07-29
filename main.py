import json
import subprocess
import uuid
from pathlib import Path
from typing import Dict
import os
import modal
import modal.gpu
import time
import websocket
from fastapi import Response

# Create a persistent volume for the model cache
cache = modal.Volume.from_name("comfyui-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("comfyui-output", create_if_missing=True)


def hf_download():
    from huggingface_hub import hf_hub_download
    import subprocess
    import os

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face token not found in environment variables.")

    # Download the main model (FLUX.1-Redux-dev)
    redux_model_path = hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-Redux-dev",
        filename="flux1-redux-dev.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/style_models && "
        f"ln -s {redux_model_path} /root/comfy/ComfyUI/models/style_models/flux1-redux-dev.safetensors",
        shell=True,
        check=True,
    )

    # Download the CLIP vision model (sigclip_vision_patch14_384)
    clip_vision_model_path = hf_hub_download(
        repo_id="Comfy-Org/sigclip_vision_384",
        filename="sigclip_vision_patch14_384.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/clip_vision && "
        f"ln -s {clip_vision_model_path} /root/comfy/ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors",
        shell=True,
        check=True,
    )

    # Download SAM model (sam-vit-huge)
    sam_model_path = hf_hub_download(
        repo_id="facebook/sam-vit-huge",
        filename="model.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/sam && "
        f"ln -s {sam_model_path} /root/comfy/ComfyUI/models/sam/sam_vit_h.safetensors",
        shell=True,
        check=True,
    )

    # Download Dino SwinT model (microsoft/swin-tiny-patch4-window7-224)
    dino_model_path = hf_hub_download(
        repo_id="microsoft/swin-tiny-patch4-window7-224",
        filename="model.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/clip && "
        f"ln -s {dino_model_path} /root/comfy/ComfyUI/models/clip/dino_swint.safetensors",
        shell=True,
        check=True,
    )

    # Download Flan T5 model (easygoing0114/flan-t5-xxl-fused)
    flan_t5_model_path = hf_hub_download(
        repo_id="easygoing0114/flan-t5-xxl-fused",
        filename="flan_t5_xxl_TE-only_FP32.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {flan_t5_model_path} /root/comfy/ComfyUI/models/clip/flan_t5_xxl_TE-only_FP32.safetensors",
        shell=True,
        check=True,
    )

    # Download VAE model (future-technologies/Floral-High-Dynamic-Range)
    vae_model_path = hf_hub_download(
        repo_id="future-technologies/Floral-High-Dynamic-Range",
        filename="ae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/vae && "
        f"ln -s {vae_model_path} /root/comfy/ComfyUI/models/vae/ae.safetensors",
        shell=True,
        check=True,
    )

    # Download Long CLIP model (zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14)
    long_clip_model_path = hf_hub_download(
        repo_id="zer0int/LongCLIP-Registers-Gated_MLP-ViT-L-14",
        filename="Long-ViT-L-14-REG-GATED-full-model.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {long_clip_model_path} /root/comfy/ComfyUI/models/clip/Long-ViT-L-14-REG-GATED-full-model.safetensors",
        shell=True,
        check=True,
    )

    # Download RealESRGAN model (Rainy-hh/Real-ESRGAN)
    esrgan_model_path = hf_hub_download(
        repo_id="Rainy-hh/Real-ESRGAN",
        filename="RealESRGAN_x4plus.pth",
        cache_dir="/cache",
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/upscale_models && "
        f"ln -s {esrgan_model_path} /root/comfy/ComfyUI/models/upscale_models/RealESRGAN_x4plus.pth",
        shell=True,
        check=True,
    )

    # Download EM_CHECKPOINT_V1 (as KRIS_kHGil_V1-000240.safetensors) model from private repo
    em_checkpoint_path = hf_hub_download(
        repo_id="repushko/flux-dev",
        filename="KRIS_kHGil_V1-000240.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )
    subprocess.run(
        f"mkdir -p /root/comfy/ComfyUI/models/checkpoints && "
        f"ln -s {em_checkpoint_path} /root/comfy/ComfyUI/models/checkpoints/EM_CHECKPOINT_V1.safetensors",
        shell=True,
        check=True,
    )

    # Download diffusion_model_merged.safetensors from private repo
    diffusion_model_path = hf_hub_download(
        repo_id="repushko/flux-dev",
        filename="diffusion_model_merged.safetensors",
        cache_dir="/cache",
        token=hf_token,
    )
    subprocess.run(
        f"ln -s {diffusion_model_path} /root/comfy/ComfyUI/models/checkpoints/diffusion_model_merged.safetensors",
        shell=True,
        check=True,
    )


image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi[standard]==0.115.4",
        "comfy-cli==1.4.1",
        "huggingface_hub==0.24.1",
        "requests==2.32.3",
        "websocket-client==1.8.0",
    )
    .run_commands("comfy --skip-prompt install --fast-deps --nvidia --version 0.3.41")
    .run_function(hf_download, secrets=[modal.Secret.from_name("huggingface")])
    .add_local_file(
        "TOKALON_SUMKI_MIN.json", remote_path="/root/TOKALON_SUMKI_MIN.json"
    )
)

comfy_app = modal.App(
    "comfyui-app",
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={"/cache": cache, "/root/comfy/ComfyUI/output": output_vol},
)


@comfy_app.cls(
    gpu=modal.gpu.A100(size="80GB"),
    volumes={"/cache": cache, "/root/comfy/ComfyUI/output": output_vol},
    # Set up concurrency and scaling parameters
    concurrency_limit=100,
    scaledown_window=300,  # Formerly container_idle_timeout
)
class ComfyUI:
    def __enter__(self):
        # start comfyui
        self.comfy_proc = subprocess.Popen(
            "python main.py --dont-print-server --multi-user",
            shell=True,
            cwd="/root/comfy/ComfyUI",
            env=os.environ,
        )

        # wait for server to be ready
        while True:
            try:
                import requests

                requests.get("http://127.0.0.1:8188/queue")
                print("ComfyUI server is ready")
                break
            except requests.exceptions.ConnectionError:
                time.sleep(1)

    @modal.method()
    def infer(self, workflow_file: str) -> bytes:
        import requests

        workflow = json.loads(Path(workflow_file).read_text())

        # Extract client_id from the workflow itself, which api() will place there.
        client_id = workflow.pop("client_id_for_ws")
        payload = {"prompt": workflow, "client_id": client_id}

        # Send the job to the ComfyUI server
        response = requests.post("http://127.0.0.1:8188/prompt", json=payload)
        response.raise_for_status()

        # Connect to the websocket to get the resulting image
        ws = websocket.WebSocket()
        ws.connect(f"ws://127.0.0.1:8188/ws?clientId={client_id}")
        image_bytes = None
        while True:
            out = ws.recv()
            if isinstance(out, str):
                message = json.loads(out)
                if message["type"] == "executed":
                    # This message indicates completion.
                    # The actual image data is sent as a separate binary message.
                    # We continue listening until we receive it.
                    continue
            else:
                # The image is sent as a binary message.
                image_bytes = out[8:]
                break

        ws.close()
        Path(workflow_file).unlink()

        if not image_bytes:
            raise ValueError("Image not found in websocket message")

        return image_bytes

    def get_image(self, filename, subfolder, folder_type):
        # This function is not used in the infer method, but is kept for potential future use
        # or if it's part of a larger ComfyUI context.
        # For now, it's a placeholder.
        data = f"/root/comfy/ComfyUI/{folder_type}/{subfolder}/{filename}"
        with open(data, "rb") as f:
            return f.read()

    @modal.fastapi_endpoint(method="POST")
    def api(self, item: Dict):
        prompt_text = item.get("prompt")
        if not prompt_text:
            return Response(
                content='{"error": "prompt not found in request"}',
                status_code=400,
                media_type="application/json",
            )

        workflow_path = Path("/root/TOKALON_SUMKI_MIN.json")
        if not workflow_path.exists():
            return Response(
                content='{"error": "workflow file not found"}',
                status_code=500,
                media_type="application/json",
            )

        workflow_data = json.loads(workflow_path.read_text())

        # Inject prompt into node 6
        workflow_data["6"]["inputs"]["text"] = prompt_text

        # Create unique filename and client_id
        client_id = uuid.uuid4().hex
        workflow_data["592"]["inputs"]["filename_prefix"] = client_id

        # Embed client_id for the websocket connection in infer()
        workflow_data["client_id_for_ws"] = client_id

        # Save temporary workflow to a shared volume
        temp_workflow_path = (
            f"/root/comfy/ComfyUI/output/{client_id}_workflow.json"
        )
        Path(temp_workflow_path).write_text(json.dumps(workflow_data))

        img_bytes = self.infer.remote(temp_workflow_path)

        return Response(img_bytes, media_type="image/jpeg")
