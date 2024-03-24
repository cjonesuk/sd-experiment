# SD Experiment

A repo for experimenting with stable diffusion workflows

## Installation

```ps
comfyui --create-directories
```

### Insightface

Installing insightface instructions from the Reactor github
https://github.com/Gourieff/comfyui-reactor-node?tab=readme-ov-file#troubleshooting

https://github.com/Gourieff/Assets/tree/main/Insightface
[insightface-0.7.3-cp310-cp310-win_amd64.whl]()

### onnxruntime

[ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "E:\dev\sd\sd-experiment\.venv\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
https://github.com/microsoft/onnxruntime/issues/11826

See DLL Requirements (CUDA)
https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html

[Install CUDA 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

## Links

[ComfyUI](https://github.com/comfyanonymous/ComfyUI)
[ComfyUI Distributed](https://github.com/hiddenswitch/ComfyUI)
[ComfyScript](https://github.com/Chaoses-Ib/ComfyScript)

### Models

[IP Adapter FaceID](https://huggingface.co/h94/IP-Adapter-FaceID)
[ControlNet v1.1](https://huggingface.co/nolanaatama/controlnetv1.1)

## Tasks

- Change temp directory or stop using temp files for images
  C:\Users\USERNAME\AppData\Local\Temp\gradio
- Display output image
- Display progress
- Checkpoint selection UI
- Use subdirectory for core code
- Installation script
- Custom nodes
- ControlNet
- IP Adaptor
- Text to image workflow
- Inpainting workflow
- Multi stage workflows
- Pix2Pix?

##
