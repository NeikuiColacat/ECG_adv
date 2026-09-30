"""Device-local ECG-Image-Kit quilting and wrinkle blending (BSD-3-Clause).

Reference: 27b90f56896c9fc78b05a83ca14844ea2637aa0b, CreasesWrinkles/creases.py.
The author wrapper uses one 250-pixel block: all overlap costs are zero and
the first candidate wins. General multi-block quilting is exposed separately.
Inputs are RGB BCHW floats in [0,1]. Search is exact over the author's candidate
extent, with bounded temporary storage; integer byte-domain SSE removes floating
tie ambiguity. This preparation operation is not intended for every model step.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

import torch
import torch.nn.functional as F

ECG_IMAGE_KIT_COMMIT = '27b90f56896c9fc78b05a83ca14844ea2637aa0b'

WRINKLE_TEXTURE_ROOT = Path('/home/linbinhao/ECG_adv_data/assets/paper_ecg_wrinkles_27b90f56896c')
WRINKLE_TEXTURE_SHA256 = {
    '1.jpg': 'ede9b0ad96d42e914b98f78042db53b2cd9266d1394f355f8c4ce017bc04c3a8',
    '10.jpg': '680ef261514a11ac900968654dfd3cd52cc66ec82bda9d3c5fdf284aa2508b9e',
    '11.jpg': 'ed564559cc62e535cc712407b22d1cb5bb1b83156f56fed7fe6ea1c22fb025f4',
    '12.jpg': '6e784ebf3573c30b12ec118d4546ad8a5854b2c42da778ddfb1ddfd64ccabc50',
    '13.jpg': '3d11cbb33d767ec25ea8a79ad9ae6bcaecd25312602c81d3dfd1f1eb318aeec8',
    '14.jpg': '43ea93c6b36f9be73d87d604399548c9c226b4821d9ead2342e6bc32befaa41d',
    '15.jpg': '6d3ae793760667b28607958c7c0c0d5d1bfdbcb4a00d566bba33c2c06ae6e481',
    '17.jpg': '7c536c6f19ae979dc778b721780c735edb90cec3e4b67a43d8b96b7b40dad22e',
    '18.jpg': 'e8dcea511d134f0ea87a2861e8d183d29587ada18c96aef4c853d75e5ae275b1',
    '19.jpg': '7c11650ebb8ee0ba73b65c98776e0759bf68af121313d9c2563d120ed504ceb9',
    '2.jpg': '0f05a01564c16d9c86b148494dadf2ebc3948e40526df70243f10f2a5aded23a',
    '20.jpg': '8f2cd32aa0ead542a6523dca7eb87348c41c84237076dc6d517b40734eccde1f',
    '3.jpg': '2e62a9099159483a40bfd56177a3bdcb0e77ab9e473043f78c2890c3a982b5f2',
    '4.jpg': 'f7f629380a74654d43f7fc1e94eca1795cb17dde363004700b638e537ffe8501',
    '5.jpg': 'ef354e3ed964d8bdfa4c4d9b7b5e17aa6a3c4d201cc435ccf21f7181d992d753',
    '6.jpg': 'd00e758efcfecd1ff4d8df22839be4c768e73eb48c9013e87c6bc99b4aa2b78f',
    '7.jpg': 'b55e79df46574b16bbf6dfebec60af6f31dc27121a3e6a16d08b9f776ab5823d',
    '8.jpg': '3236f72f8a0d00834ae60a0a59d2b382da2c0eab17be41a3ce8d00dc6378a041',
    '9.jpg': 'f9e36f628cd8846ce76ec14c48086c6f51e062e6ba5d7c786a878d78c38e6a36',
}
WRINKLE_PATCH_SHA256 = 'cb47acb9472cafe11c2278e516d8091a973f34664746571a665bb1795cc4f1ec'
WRINKLE_BANK_ID = hashlib.sha256((WRINKLE_PATCH_SHA256+''.join(name+digest for name, digest in WRINKLE_TEXTURE_SHA256.items())).encode()).hexdigest()


def fetch_original_textures(directory=WRINKLE_TEXTURE_ROOT):
    """Explicit asset setup, never called by an operator or an experiment.

    Download only the pinned author files; verify bytes before creating them.
    Existing mismatched files fail closed. No patient images are involved.
    """
    from urllib.request import urlopen
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    prefix = ('https://raw.githubusercontent.com/alphanumericslab/ecg-image-kit/'
              + ECG_IMAGE_KIT_COMMIT + '/codes/ecg-image-generator/CreasesWrinkles/wrinkles-dataset/')
    for name, digest in WRINKLE_TEXTURE_SHA256.items():
        path = root/name
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'existing texture differs from pinned source: {path}')
            continue
        with urlopen(prefix+name, timeout=30) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f'downloaded texture differs from pinned source: {name}')
        with path.open('xb') as output:
            output.write(content)
    return root


@lru_cache(maxsize=4)
def load_original_texture_bank(device, directory=WRINKLE_TEXTURE_ROOT):
    """One-time verified decode/upload of the author default 19 texture patches.

    Raw photos remain outside Git. No file access or host image copy occurs
    after this device/directory pair is warmed. Call explicitly before timing.
    """
    import numpy as np
    from PIL import Image
    patches = []
    for name, digest in WRINKLE_TEXTURE_SHA256.items():
        path = Path(directory)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'missing or changed pinned wrinkle texture: {path}')
        with Image.open(path) as image:
            if min(image.size) <= 250:
                raise ValueError('author wrinkle textures must exceed 250 pixels')
            patch = np.array(image.convert('RGB').crop((0, 0, 250, 250)), copy=True)
        patches.append(torch.from_numpy(patch).permute(2, 0, 1))
    pixels = torch.stack(patches)
    if hashlib.sha256(pixels.numpy().tobytes()).hexdigest() != WRINKLE_PATCH_SHA256:
        raise ValueError('decoded wrinkle pixels differ from the pinned bank')
    return pixels.to(device=device, dtype=torch.float32).div_(255)



def minimum_cut_path(errors):
    """Minimum vertical seam; equal costs use lexicographically first paths."""
    batch, height, width = errors.shape
    cost = errors[:, 0].to(torch.int64)
    columns = torch.arange(width, device=errors.device).expand(batch, -1)
    rank = columns
    parents = []
    infinity = torch.iinfo(torch.int64).max // 4
    for y in range(1, height):
        padded = F.pad(cost, (1, 1), value=infinity)
        previous = torch.stack([padded[:, d:d+width] for d in range(3)], -1)
        padded_rank = F.pad(rank, (1, 1), value=infinity)
        ranks = torch.stack([padded_rank[:, d:d+width] for d in range(3)], -1)
        best = previous.amin(-1, keepdim=True)
        choice = torch.where(previous == best, ranks, infinity).argmin(-1)
        parent = columns + choice - 1
        parents.append(parent)
        prefix_rank = rank.gather(1, parent)
        rank = (prefix_rank * width + columns).argsort(1).argsort(1)
        cost = best.squeeze(-1) + errors[:, y]
    end = torch.where(cost == cost.amin(1, keepdim=True), rank, infinity).argmin(1)
    path = [end]
    for parent in reversed(parents):
        end = parent.gather(1, end[:, None]).squeeze(1)
        path.append(end)
    return torch.stack(path[::-1], 1)


def _best_patch(texture, result, y, x, block, overlap):
    batch, channels, height, width = texture.shape
    rows, columns = height-block, width-block  # Preserve author's excluded endpoint.
    if y == 0 and x == 0:
        return texture[:, :, :block, :block].clone()
    windows = texture.unfold(2, block, 1).unfold(3, block, 1)
    target = result[:, :, y:y+block, x:x+block]
    chunk = max(1, 2_000_000 // (batch * channels * columns * block * overlap))
    best_error = torch.full((batch,), torch.iinfo(torch.int64).max, device=texture.device)
    best_index = torch.zeros(batch, dtype=torch.long, device=texture.device)
    for start in range(0, rows, chunk):
        candidate = windows[:, :, start:min(rows, start+chunk), :columns]
        error = torch.zeros((batch, candidate.shape[2], columns), device=texture.device, dtype=torch.int64)
        if x:
            diff = candidate[..., :overlap] - target[:, :, None, None, :, :overlap]
            error += diff.square().sum((1, 4, 5))
        if y:
            diff = candidate[..., :overlap, :] - target[:, :, None, None, :overlap, :]
            error += diff.square().sum((1, 4, 5))
        if x and y:
            diff = candidate[..., :overlap, :overlap] - target[:, :, None, None, :overlap, :overlap]
            error -= diff.square().sum((1, 4, 5))
        local_error, local_index = error.flatten(1).min(1)
        improve = local_error < best_error
        best_index = torch.where(improve, local_index + start*columns, best_index)
        best_error = torch.minimum(local_error, best_error)
    yy = best_index[:, None, None] // columns + torch.arange(block, device=texture.device)[None, :, None]
    xx = best_index[:, None, None] % columns + torch.arange(block, device=texture.device)[None, None, :]
    index = (yy*width+xx).flatten(1)[:, None].expand(-1, channels, -1)
    return texture.flatten(2).gather(2, index).reshape(batch, channels, block, block)


def quilt_texture(texture, block_size=250, num_blocks=(1, 1)):
    """Port of exhaustive best-patch/min-cut quilting; never copies to the CPU."""
    if (texture.ndim != 4 or texture.shape[1] != 3 or not texture.is_floating_point()
            or type(block_size) is not int or block_size < 6
            or min(texture.shape[-2:]) <= block_size
            or len(num_blocks) != 2 or any(type(n) is not int or n < 1 for n in num_blocks)):
        raise ValueError('quilting requires RGB BCHW, texture larger than block, and positive grid')
    if not bool(torch.isfinite(texture).all()) or bool(((texture < 0) | (texture > 1)).any()):
        raise ValueError('quilting requires finite RGB in [0,1]')
    block, overlap = block_size, block_size // 6
    step = block-overlap
    height, width = [n*block-(n-1)*overlap for n in num_blocks]
    pixels = texture.float().mul(255).round().to(torch.int32)
    result = torch.zeros((*texture.shape[:2], height, width), dtype=torch.int32, device=texture.device)
    axis = torch.arange(block, device=texture.device)
    for row in range(num_blocks[0]):
        for column in range(num_blocks[1]):
            y, x = row*step, column*step
            patch = _best_patch(pixels, result, y, x, block, overlap)
            old = result[:, :, y:y+block, x:x+block]
            keep = torch.zeros((texture.shape[0], block, block), dtype=torch.bool, device=texture.device)
            if column:
                errors = (patch[..., :overlap]-old[..., :overlap]).square().sum(1)
                seam = minimum_cut_path(errors)
                keep |= axis[None, None, :] < seam[:, :, None]
            if row:
                errors = (patch[:, :, :overlap]-old[:, :, :overlap]).square().sum(1).transpose(1, 2)
                seam = minimum_cut_path(errors)
                keep |= axis[None, :, None] < seam[:, None, :]
            old.copy_(torch.where(keep[:, None], old, patch))
    return result.to(texture.dtype).div(255)


def apply_wrinkle_texture(image, texture, strength=1.0):
    """Author's resize/mean-shift/thresholded overlay, with explicit strength.

    The source reads RGB through PIL then uses BGR2GRAY. Keep those reversed
    red/blue luminance weights here for faithful fixed-texture comparisons.
    """
    texture = texture.float()
    gray = .114*texture[:, :1] + .587*texture[:, 1:2] + .299*texture[:, 2:3]
    gray = gray.mul(255).round().div(255)
    gray = F.interpolate(gray, image.shape[-2:], mode='bilinear', align_corners=False)
    gray = gray - gray.mean((-1, -2), keepdim=True) + .4
    value = image.float()
    transformed = torch.where(gray > .6, 1-2*(1-value)*(1-gray), 2*value*gray)
    return (value + strength*(transformed-value)).clamp(0, 1).mul(255).floor().div(255).to(image.dtype)
