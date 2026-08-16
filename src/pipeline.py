import torch
import numpy as np


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


def select_probe_sites(sites):
    """Keep one representative transformer block per attentions module (the
    first), for both self- (attn1) and cross- (attn2) attention. ~22 sites,
    ordered by network depth."""
    return [s for s in sites
            if "transformer_blocks.0." in s
            and (s.endswith("attn1") or s.endswith("attn2"))]


def pool_activation(act, cond_index=1):
    """Pool an attention output to one channel-vector: take the conditional
    half of the CFG batch, mean over tokens. (B,T,C)->(C,); (T,C)->(C,)."""
    a = act.detach().cpu().float().numpy() if hasattr(act, "detach") \
        else np.asarray(act, float)
    if a.ndim == 3:
        b = a[cond_index] if a.shape[0] > cond_index else a[-1]
        return b.mean(axis=0)
    if a.ndim == 2:
        return a.mean(axis=0)
    return a.ravel()


def generate_and_capture(pipe, prompt, seed, sites, capture_steps,
                         num_inference_steps=30, cond_index=1, reducer=None):
    """One generation; snapshot reduced activations at each step in
    capture_steps. `reducer(act, cond_index)` defaults to pool_activation
    (channel vector); pass featuremap_saliency for spatial maps. Returns
    (image, {step: {site: reduced}}). GPU only."""
    if reducer is None:
        reducer = pool_activation
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    store = {}
    targets = set(capture_steps)
    with ActivationCapture(pipe.unet, sites) as cap:
        def cb(_p, step, _timestep, kw):
            if step in targets:
                store[step] = {s: reducer(cap.acts[s], cond_index)
                               for s in cap.acts}
            return kw
        img = pipe(prompt, generator=g,
                   num_inference_steps=num_inference_steps,
                   callback_on_step_end=cb).images[0]
    return img, store


def raw_reducer(act, cond_index=1):
    """Reducer that keeps the full activation tensor (on CPU) for injection."""
    return act.detach().cpu()


def make_patch_hook(site, patch_map, state):
    """Forward hook that replaces a module's output with a donor tensor when
    the current step (state['step']) is a patch step for this site."""
    def hook(_m, _i, out):
        st = state["step"]
        if st in patch_map and site in patch_map[st]:
            return patch_map[st][site].to(out.dtype).to(out.device)
        return None
    return hook


def generate_with_patch(pipe, prompt, seed, patch_map, num_inference_steps=30):
    """Generate `prompt` while injecting donor activations: at each step in
    patch_map, replace the listed sites' outputs with the stored tensors.
    patch_map = {step: {site: tensor}}. GPU only."""
    sites = {s for d in patch_map.values() for s in d}
    modmap = dict(pipe.unet.named_modules())
    state = {"step": 0}
    handles = [modmap[s].register_forward_hook(make_patch_hook(s, patch_map, state))
               for s in sites]
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    def cb(_p, step, _t, kw):
        state["step"] = step + 1
        return kw
    try:
        img = pipe(prompt, generator=g,
                   num_inference_steps=num_inference_steps,
                   callback_on_step_end=cb).images[0]
    finally:
        for h in handles:
            h.remove()
    return img


def make_steer_hook(site, directions, alpha, steps, state):
    """Forward hook that ADDS alpha*direction to a module's output at the steer
    steps (a donor-free count knob, vs make_patch_hook which replaces)."""
    def hook(_m, _i, out):
        st = state["step"]
        if st in steps and st in directions and site in directions[st]:
            d = directions[st][site].to(out.dtype).to(out.device)
            return out + alpha * d
        return None
    return hook


def generate_with_steer(pipe, prompt, seed, directions, alpha, steps,
                        num_inference_steps=30):
    """Generate while adding alpha*directions[step][site] at the steer steps.
    directions = {step: {site: (C,) tensor}}. GPU only."""
    steps = set(steps)
    sites = {s for st in directions for s in directions[st]}
    modmap = dict(pipe.unet.named_modules())
    state = {"step": 0}
    handles = [modmap[s].register_forward_hook(
        make_steer_hook(s, directions, alpha, steps, state)) for s in sites]
    g = torch.Generator(device=pipe.device).manual_seed(seed)

    def cb(_p, step, _t, kw):
        state["step"] = step + 1
        return kw
    try:
        img = pipe(prompt, generator=g, num_inference_steps=num_inference_steps,
                   callback_on_step_end=cb).images[0]
    finally:
        for h in handles:
            h.remove()
    return img
