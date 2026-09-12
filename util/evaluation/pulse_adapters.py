"""Paired PULSE LoRA inference sharing one frozen base and vision forward.

No adapter merging, quantization, class head, prompt changes or probability
surrogate. Each arm uses the author's autoregressive generation path after an
exactly checked flat visual-token merge. GPU admission belongs to the caller.
"""
from __future__ import annotations

import json
import os
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
        from core.pulse_finetune import MODEL_HASHES, load_pulse_for_training
        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX as AUTHOR_IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token
        from util.evaluation.ecg_image_data import QUESTION

        if max_new_tokens != 32 or AUTHOR_IMAGE_TOKEN_INDEX != IMAGE_TOKEN_INDEX:
            raise ValueError("PULSE generation contract changed")
        protocol = evidence["w1"]["result"]["protocol"]
        for name, digest in MODEL_HASHES.items():
            if sha256_file(Path(protocol["model"]["directory"]) / name) != digest:
                raise ValueError("source PULSE model shard changed")
        self.tokenizer, self.model, self.loading = load_pulse_for_training(protocol)
        if reservation is not None:
            reservation.clear()
            torch.cuda.empty_cache()
        self.model.to("cuda")
        self.base = self.model.get_base_model()
        self.parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
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
        with torch.autocast("cuda", dtype=torch.float16):
            features = self.base.get_vision_tower()(pixels.flatten(0, 1))
        for arm in ("w1", "w2"):
            self._select(arm)
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
