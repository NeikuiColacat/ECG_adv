"""Paired PULSE LoRA inference sharing one frozen base and vision forward.

No adapter merging, quantization, class head, prompt changes or probability
surrogate. Each arm uses the author's autoregressive generation path after an
exactly checked flat visual-token merge. GPU admission belongs to the caller.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path

import torch

from util.pn2021_artifact_contract import sha256_file
from util.evaluation.ecg_image_queue import digest_json
from util.pulse_training_contract import validate_result

REPO = Path(__file__).resolve().parents[2]
IMAGE_TOKEN_INDEX = -200


def matched_training_evidence(paths, *, allow_smoke=False):
    if set(paths) != {"w1", "w2"}:
        raise ValueError("exactly the single and two-chain adapters are required")
    outputs, common = {}, None
    for arm, raw in paths.items():
        path = Path(raw).resolve()
        path.relative_to(Path("/home/linbinhao/ECG_adv_data"))
        result = json.loads(path.read_text())
        validate_result(result, path)
        expected_mode = "smoke" if allow_smoke else "train"
        if result["mode"] != expected_mode or result["protocol"]["width"] != int(arm[1:]):
            raise ValueError("adapter mode or chain count mismatch")
        protocol = dict(result["protocol"])
        if "model_asset_sha256" not in protocol:
            raise ValueError("adapter result lacks the closed PULSE model asset manifest; retrain it")
        from core.pulse_finetune import MODEL_CONFIG_HASHES, MODEL_HASHES
        if protocol["model_asset_sha256"] != {**MODEL_HASHES, **MODEL_CONFIG_HASHES}:
            raise ValueError("adapter model asset manifest is not the locked PULSE closure")
        protocol.pop("width")
        common = protocol if common is None else common
        if protocol != common:
            raise ValueError("adapter pair differs in more than chain count")
        for name, digest in protocol["implementation_sha256"].items():
            if sha256_file(REPO / name) != digest:
                raise ValueError("training loader/source identity drift before adapter inference")
        outputs[arm] = {"path": path, "sha256": sha256_file(path), "result": result}
    return outputs


def pack_shared_prompt_features(base, input_ids, features):
    """Vectorized author flat merge, for identical unpadded Super5 prompts."""
    if input_ids.ndim != 2 or not 1 <= len(input_ids) <= 4:
        raise ValueError("unsupported paired inference batch")
    if not torch.equal(input_ids, input_ids[:1].expand_as(input_ids)):
        raise ValueError("paired PULSE batch must use one identical unpadded prompt")
    if tuple(features.shape) != (len(input_ids) * 5, 576, 1024) or base.config.mm_patch_merge_type != "flat":
        raise ValueError("paired inference requires exactly five flat 576-token visual tiles")
    positions = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
    if len(positions) != 1:
        raise ValueError("expected one image placeholder per prompt")
    position = int(positions.item())
    ids = torch.cat((input_ids[:, :position], input_ids[:, position + 1:]), dim=1)
    text = base.get_model().embed_tokens(ids)
    # Project all tiles together, matching the author's batch GEMM shape.
    visual = base.get_model().mm_projector(features).reshape(len(input_ids), 5 * 576, text.shape[-1])
    packed = torch.cat((text[:, :position], visual, text[:, position:]), dim=1)
    if packed.shape[1] > base.config.tokenizer_model_max_length:
        raise ValueError("paired input would exceed the frozen context limit")
    return packed


def decode_generated(tokenizer, output, *, max_new_tokens):
    # Transformers 4.37 initializes embedded-prompt generation with one dummy
    # BOS; it is not a generated answer token and must not inflate truncation.
    if output.ndim != 2 or not bool((output[:, 0] == tokenizer.bos_token_id).all()):
        raise ValueError("unexpected embedded-prompt generation prefix")
    answers = output[:, 1:].cpu()
    rows = []
    for tokens in answers:
        endings = (tokens == tokenizer.eos_token_id).nonzero().flatten()
        length = int(endings[0]) + 1 if len(endings) else len(tokens)
        rows.append({"response": tokenizer.decode(tokens[:length], skip_special_tokens=True).strip(),
                     "generated_tokens": length, "hit_max_new_tokens": not len(endings) and length >= max_new_tokens})
    return rows


class PairedPulseBackend:
    """Two GPU-resident adapter states; immutable CLIP features shared per batch."""
    def __init__(self, evidence, *, max_new_tokens=32, reservation=None):
        os.environ["HF_HOME"] = "/home/linbinhao/ECG_adv_data/hf_cache/pulse"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from core.pulse_finetune import load_pulse_for_training, verify_model_assets
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX as AUTHOR_IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from util.evaluation.ecg_image_data import QUESTION

        if max_new_tokens != 32 or AUTHOR_IMAGE_TOKEN_INDEX != IMAGE_TOKEN_INDEX:
            raise ValueError("PULSE generation contract changed")
        protocol = evidence["w1"]["result"]["protocol"]
        verify_model_assets(Path(protocol["model"]["directory"]))
        self.tokenizer, self.model, self.loading = load_pulse_for_training(protocol, model_assets_verified=True)
        if reservation is not None:
            reservation.clear()
            torch.cuda.empty_cache()
        self.model.to("cuda")
        self.base = self.model.get_base_model()
        self.parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        # A visual LoRA state changes CLIP features themselves.  It cannot
        # share one feature tensor across adapter arms; language-only arms can.
        self.visual_lora = any(".vision_tower." in name for name in self.parameters)
        self.base_identity = protocol["model"]
        self.states = {}
        self.set_pair(evidence)
        self.model.requires_grad_(False)
        self.model.eval()
        self.base.gradient_checkpointing_disable()
        self.base.disable_input_require_grads()
        self.base.config.use_cache = True
        self.max_new_tokens = max_new_tokens
        pad_id = self.tokenizer.pad_token_id
        self.generation_kwargs = {"do_sample": False, "max_new_tokens": max_new_tokens, "use_cache": True,
                                  "pad_token_id": self.tokenizer.eos_token_id if pad_id is None else pad_id}
        conversation = conv_templates["llava_v1"].copy()
        conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + QUESTION)
        conversation.append_message(conversation.roles[1], None)
        self.prompt = conversation.get_prompt()
        self.prompt_ids = tokenizer_image_token(self.prompt, self.tokenizer, IMAGE_TOKEN_INDEX,
                                               return_tensors="pt").unsqueeze(0).cuda()
        self.admissions = {}

    def set_pair(self, evidence):
        """Switch target-center adapters without reloading the frozen 7B base."""
        states = {}
        for arm, item in evidence.items():
            protocol = item["result"]["protocol"]
            if protocol["model"] != self.base_identity:
                raise ValueError("center switch changed the shared PULSE base")
            checkpoint = torch.load(item["path"].parent / "checkpoint.pt", map_location="cpu")
            state = checkpoint["trainables"]
            if (set(state) != set(self.parameters) or checkpoint["step"] != item["result"]["optimizer_steps"]
                    or checkpoint["protocol_identity"] != digest_json(protocol)):
                raise ValueError("adapter parameter keys or completed step mismatch")
            for name, value in state.items():
                if value.shape != self.parameters[name].shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
                    raise ValueError("invalid saved FP32 adapter/projector tensor")
            states[arm] = {name: value.to("cuda") for name, value in state.items()}
        self.states = states

    def _select(self, arm):
        for name, parameter in self.parameters.items():
            parameter.copy_(self.states[arm][name])

    def _generate(self, packed):
        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        # This is the same super().generate called by the author immediately
        # after visual packing. No text or image token is dropped or resampled.
        return super(LlavaLlamaForCausalLM, self.base).generate(inputs_embeds=packed, **self.generation_kwargs)

    @torch.inference_mode()
    def generate_pair(self, pixels, *, verify_author=False):
        if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (5, 3, 336, 336):
            raise ValueError("PULSE expects Bx5x3x336x336 preprocessed image tiles")
        pixels = pixels.to(device="cuda", dtype=torch.float16)
        ids = self.prompt_ids.expand(len(pixels), -1)
        outputs = {}
        self.last_token_ids = {}
        features = None
        if not getattr(self, "visual_lora", False):
            with torch.autocast("cuda", dtype=torch.float16):
                features = self.base.get_vision_tower()(pixels.flatten(0, 1))
        for arm in ("w1", "w2"):
            self._select(arm)
            if getattr(self, "visual_lora", False):
                with torch.autocast("cuda", dtype=torch.float16):
                    features = self.base.get_vision_tower()(pixels.flatten(0, 1))
            # End the previous AMP context before overwriting FP32 trainables:
            # a cached FP16 weight must never survive an adapter switch.
            with torch.autocast("cuda", dtype=torch.float16):
                packed = pack_shared_prompt_features(self.base, ids, features)
                generated = self._generate(packed)
                if verify_author:
                    reference = self.base.prepare_inputs_labels_for_multimodal(ids, None, None, None, None,
                        pixels, image_sizes=[(2200, 1700)] * len(pixels))[4]
                    if not torch.equal(reference, packed):
                        raise ValueError("paired visual packing is not author-exact")
                    reference_ids = self.base.generate(ids, images=pixels, image_sizes=[(2200, 1700)] * len(pixels),
                                                       **self.generation_kwargs)
                    if not torch.equal(reference_ids, generated):
                        raise ValueError("shared-feature and independent author generation differ")
                    self.admissions[f"{arm}_batch{len(pixels)}"] = {"packing_exact": True, "generated_token_ids_exact": True}
                generated_cpu = generated.cpu()
                self.last_token_ids[arm] = generated_cpu.clone()
                outputs[arm] = decode_generated(self.tokenizer, generated_cpu, max_new_tokens=self.max_new_tokens)
        return outputs

class HybridPulseBackend(PairedPulseBackend):
    """Original/single/three inference with explicit adapter and precision policy."""
    def __init__(self, evidence, reservation, *, arms, fp16_adapters=False, bypass_original_lora=False):
        self.arms = tuple(arms)
        self.fp16_adapters = fp16_adapters
        self.bypass_original_lora = bypass_original_lora
        source_arm = next(a for a in self.arms if a != "original")
        super().__init__({"w1": evidence[source_arm]}, reservation=reservation)
        # set_pair stores states but has not applied an adapter yet.
        original = {n: p.detach().clone() for n, p in self.parameters.items()}
        self.set_pair(evidence)
        self.states["original"] = original
        if bypass_original_lora:
            self._validate_original_lora()

    def _validate_original_lora(self):
        factors = [p for name, p in self.states["original"].items() if ".lora_B." in name]
        if not factors or any(torch.count_nonzero(p).item() for p in factors):
            raise ValueError("original LoRA bypass requires exactly zero saved B factors")

    @contextmanager
    def _without_original_lora(self):
        # Only the original arm has a provably zero LoRA update. The projector
        # is still restored and evaluated, and trained arms keep their branches.
        if self.model.training or torch.is_grad_enabled() or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("original LoRA bypass requires frozen eval/no-grad inference")
        if any(getattr(m, "merged", False) is True or getattr(m, "disable_adapters", False) is True
               for m in self.model.modules()):
            raise ValueError("original LoRA bypass requires enabled, unmerged adapters")
        try:
            with self.model.disable_adapter():
                yield
        finally:
            # PEFT 0.7 re-enables gradients when it re-enables the adapters.
            self.model.requires_grad_(False)

    def _prepare_prompt_cache(self):
        # Only the frozen prompt embeddings may be reused. Visual features
        # and projector output are recomputed for each adapter and view.
        if any("embed_tokens" in name for name in self.parameters):
            raise ValueError("prompt cache requires frozen text embeddings")
        positions = (self.prompt_ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
        if len(positions) != 1:
            raise ValueError("expected one image placeholder per prompt")
        self.prompt_position = int(positions.item())
        ids = torch.cat((self.prompt_ids[:, :self.prompt_position],
                         self.prompt_ids[:, self.prompt_position + 1:]), dim=1)
        with torch.inference_mode():
            self.prompt_text = self.base.get_model().embed_tokens(ids).detach()

    @torch.inference_mode()
    def generate_views(self, views, *, bypass_original_lora=None):
        bypass = self.bypass_original_lora if bypass_original_lora is None else bypass_original_lora
        if bypass and not self.bypass_original_lora:
            raise ValueError("original LoRA bypass was not validated at model load")
        if not self.fp16_adapters:
            return self._generate_views(views, bypass_original_lora=bypass)
        # PEFT otherwise promotes each FP16 activation to its FP32 LoRA
        # storage type, then autocast immediately converts it back. Keep
        # exact reference FP32 buffers and checkpoints; use the same FP16
        # linear operands once per arm for this inference-only region.
        if any(p.dtype != torch.float32 or not any(marker in name for marker in
                   (".lora_A.", ".lora_B.", ".mm_projector."))
               for name, p in self.parameters.items()):
            raise ValueError("FP16 inference storage is restricted to FP32 LoRA/projector linears")
        saved = {name: parameter.data for name, parameter in self.parameters.items()}
        try:
            for parameter in self.parameters.values():
                # Every arm copies its full state before use; casting the
                # previous arm here would produce values immediately overwritten.
                parameter.data = torch.empty_like(parameter, dtype=torch.float16)
            return self._generate_views(views, bypass_original_lora=bypass)
        finally:
            for name, parameter in self.parameters.items():
                parameter.data = saved[name]

    @torch.inference_mode()
    def _generate_views(self, views, *, bypass_original_lora=False):
        """Amortize exact adapter copies; validate against the same batch reference."""
        if not hasattr(self, "prompt_text"):
            self._prepare_prompt_cache()
        if not views or any(p.ndim != 5 or not 1 <= len(p) <= 4
               or len(p) != len(views[0]) or tuple(p.shape[1:]) != (5, 3, 336, 336) or p.dtype != torch.float16
               or p.device.type != "cuda" for p in views):
            raise ValueError("optimized C5 requires matched native FP16 CUDA batches")
        outputs = [{} for _ in views]
        tokens = [{} for _ in views]
        for arm in self.arms:
            with torch.cuda.nvtx.range("adapter_copy"):
                torch._foreach_copy_(list(self.parameters.values()),
                    [self.states[arm][name] for name in self.parameters])
            # All cached casts expire before the next adapter is selected.
            adapter_context = self._without_original_lora() if bypass_original_lora and arm == "original" else nullcontext()
            with adapter_context, torch.autocast("cuda", dtype=torch.float16):
                for index, pixels in enumerate(views):
                    with torch.cuda.nvtx.range("vision"):
                        features = self.base.get_vision_tower()(pixels.flatten(0, 1))
                    with torch.cuda.nvtx.range("pack"):
                        if tuple(features.shape) != (len(pixels) * 5, 576, 1024) or self.base.config.mm_patch_merge_type != "flat":
                            raise ValueError("optimized C5 visual packing changed")
                        visual = self.base.get_model().mm_projector(features).reshape(len(pixels), 2880, self.prompt_text.shape[-1])
                        prompt = self.prompt_text.expand(len(pixels), -1, -1)
                        packed = torch.cat((prompt[:, :self.prompt_position], visual,
                                            prompt[:, self.prompt_position:]), dim=1)
                        if packed.shape[1] > self.base.config.tokenizer_model_max_length:
                            raise ValueError("optimized C5 context overflow")
                    with torch.cuda.nvtx.range("generate"):
                        generated = self._generate(packed)
                    host = generated.cpu()
                    tokens[index][arm] = host.tolist()
                    outputs[index][arm] = decode_generated(self.tokenizer, host, max_new_tokens=32)
        self.last_view_token_ids = tokens
        return outputs

    @torch.inference_mode()
    def generate_pair(self, pixels, *, verify_author=False):
        if pixels.ndim != 5 or tuple(pixels.shape[1:]) != (5, 3, 336, 336):
            raise ValueError("PULSE expects Bx5x3x336x336")
        pixels = pixels.to(device="cuda", dtype=torch.float16)
        ids = self.prompt_ids.expand(len(pixels), -1)
        outputs = {}
        self.last_token_ids = {}
        original_tokens = None
        for arm in self.arms:
            self._select(arm)
            # End autocast before switching trainables; never reuse CLIP features.
            with torch.autocast("cuda", dtype=torch.float16):
                features = self.base.get_vision_tower()(pixels.flatten(0, 1))
                packed = pack_shared_prompt_features(self.base, ids, features)
                generated = self._generate(packed)
                if arm == "original" and verify_author:
                    original_tokens = generated.clone()
                if verify_author:
                    author = self.base.prepare_inputs_labels_for_multimodal(ids, None, None, None, None,
                        pixels, image_sizes=[(2200, 1700)] * len(pixels))[4]
                    reference = self.base.generate(ids, images=pixels,
                        image_sizes=[(2200, 1700)] * len(pixels), **self.generation_kwargs)
                    if not torch.equal(author, packed) or not torch.equal(reference, generated):
                        raise ValueError("hybrid generation differs from author packing/tokens")
                    self.admissions[f"{arm}_batch{len(pixels)}"] = {"packing_exact": True, "tokens_exact": True}
                host = generated.cpu()
                self.last_token_ids[arm] = host.tolist()
                outputs[arm] = decode_generated(self.tokenizer, host, max_new_tokens=32)
        if verify_author:
            # A second original pass after all adapter switches catches
            # leaked weights or cached visual features in the live path.
            self._select("original")
            with torch.autocast("cuda", dtype=torch.float16):
                features = self.base.get_vision_tower()(pixels.flatten(0, 1))
                restored = self._generate(pack_shared_prompt_features(self.base, ids, features))
            if original_tokens is None or not torch.equal(original_tokens, restored):
                raise ValueError("original generation changed after restoring adapter state")
            self.original_restore_admission = {"tokens_exact_after_adapter_switches": True}
        return outputs
