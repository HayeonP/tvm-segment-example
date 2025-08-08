import os
import tvm
from tvm import relax
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import torch
import time

def load_params(path):                
    param_dict = tvm.runtime.load_param_dict_from_file(path)
    ordered_keys = sorted(param_dict.keys(), key=lambda k: int(k.split("_")[1]))
    return {"main": [param_dict[k] for k in ordered_keys]}


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).numpy()  # (1, 3, 224, 224)
    return image, img_tensor


def get_imagenet_labels():
    import urllib.request
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    file_path = "imagenet_classes.txt"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    with open(file_path) as f:
        return [line.strip() for line in f.readlines()]


def save_prediction(image, label, path, title="Prediction"):
    plt.figure(figsize=(5,5))
    plt.imshow(image)
    plt.title(f"{title}: {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)


def infer_with_tvm(image_tensor: np.ndarray, params_path: str, lib_path: str, device="cuda"):
    dev = tvm.device(device, 0)
    ex = tvm.runtime.load_module(lib_path)
    params = load_params(params_path)
    vm = relax.VirtualMachine(ex, dev)

    gpu_input = tvm.nd.array(image_tensor.astype("float32"), dev)
    gpu_params = [tvm.nd.array(p, dev) for p in params["main"]]

    _ = vm["main"](gpu_input, *gpu_params)  # warm-up
    dev.sync()

    start = time.time()
    output = vm["main"](gpu_input, *gpu_params)
    dev.sync()
    end = time.time()

    print('tvm:', end - start)
    output_nd = output[0] if isinstance(output, tvm.ir.Array) else output
    return np.argmax(output_nd.numpy())


def infer_with_torch(image_tensor: np.ndarray):
    image_tensor = torch.from_numpy(image_tensor).to(torch.float32).cuda()
    model = resnet18(weights=ResNet18_Weights.DEFAULT).cuda().eval()
    
    with torch.no_grad():
        _ = model(image_tensor)  # warm-up
        torch.cuda.synchronize()

        start = time.time()
        output = model(image_tensor)
        torch.cuda.synchronize()
        end = time.time()

    print('pytorch:', end - start)
    return int(torch.argmax(output, dim=1).item())


if __name__ == "__main__":
    # 샘플 이미지 로드
    image_path = "dog.jpg"
    if not os.path.exists(image_path):
        import urllib.request
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg", image_path)

    orig_image, image_tensor = preprocess_image(image_path)
    labels = get_imagenet_labels()

    # 🔍 PyTorch 추론
    pred_torch = infer_with_torch(image_tensor.copy())
    save_prediction(orig_image, labels[pred_torch], path='torch.png', title="PyTorch")

    # 🔍 TVM 추론
    pred_tvm = infer_with_tvm(image_tensor.copy(), "resnet18.bin", "resnet18.so")
    
    save_prediction(orig_image, labels[pred_tvm], path='tvm.png', title="TVM")
