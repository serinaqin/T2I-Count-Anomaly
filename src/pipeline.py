import torch


def catalog_attention_sites(unet):
    sites = []
    for name, _ in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            sites.append(name)
    return sites


class ActivationCapture:
    def __init__(self, model, sites):
        self.model = model
        self.sites = sites
        self.acts = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self.acts[name] = out.detach() if hasattr(out, "detach") else out
        return hook

    def __enter__(self):
        modmap = dict(self.model.named_modules())
        for s in self.sites:
            self._handles.append(modmap[s].register_forward_hook(self._make_hook(s)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def load_sdxl(device="cuda", dtype=torch.float16):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype)
    return pipe.to(device)


def generate(pipe, prompt, seed, num_inference_steps=30):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt, generator=g,
                num_inference_steps=num_inference_steps).images[0]
