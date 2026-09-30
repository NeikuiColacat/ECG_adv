"""Torch ports of Augraphy printing/scanning algorithms (MIT; see license).

RGB-only contract. Small variable-size parameter plans use a private Python
RNG derived from the caller generator's seed/counter. CUDA images/random fields
never return to the host. This explicitly versioned sampler is not NumPy/Numba
seed equivalence. Fixed-parameter kernels have separate reference tests.
"""
from functools import lru_cache
import math
import random

import torch
import torch.nn.functional as F

from core.paper_ecg import PAPER_HELDOUT_OPERATORS as HELDOUT_V1, validate_paper_image
from core.paper_ecg_upstream import PAPER_TRAIN_OPERATORS as TRAIN_V2, apply_paper_operator as apply_v2
from core.paper_ecg_quilting import apply_wrinkle_texture, load_original_texture_bank

PAPER_IMPLEMENTATION = 'paper_ecg_torch_v3'
PRINT_OPERATORS = ('dirty_rollers', 'dirty_drum', 'folding', 'faxify', 'bad_photocopy', 'quilting_wrinkle')
PAPER_TRAIN_OPERATORS = TRAIN_V2 + ('dirty_rollers', 'dirty_drum', 'quilting_wrinkle')
PAPER_HELDOUT_OPERATORS = HELDOUT_V1 + ('folding', 'bad_photocopy', 'faxify')
PAPER_EVAL_OPERATORS = PAPER_TRAIN_OPERATORS + PAPER_HELDOUT_OPERATORS


def _parameter_rng(rng):
    if torch.device(rng.device).type == 'cuda':
        counter = rng.get_offset()  # Host-side Philox metadata; no tensor readback.
        torch.rand((), device=rng.device, generator=rng)
    else:
        counter = int(torch.randint(2**31-1, (), generator=rng))
    return random.Random((rng.initial_seed() ^ ((counter+1)*0x9e3779b97f4a7c15)) & (2**64-1))


def _bytes(image):
    return image.float().mul(255).round().clamp(0, 255)


def _gray(value):
    return (.299*value[:, :1] + .587*value[:, 1:2] + .114*value[:, 2:3]).round()


def _resize(value, size, *, mode='bilinear', quantize=False):
    out = F.interpolate(value, size, mode=mode, align_corners=False)
    return out.round().clamp(0, 255) if quantize else out


def _reflect(value, radius, dimension):
    n = value.shape[dimension]
    positions = torch.arange(-radius, n+radius, device=value.device)
    if n == 1:
        positions = torch.zeros_like(positions)
    else:
        positions = positions.remainder(2*(n-1))
        positions = torch.minimum(positions, 2*(n-1)-positions)
    return value.index_select(dimension, positions)


@lru_cache(maxsize=32)
def _gaussian_kernel(size, sigma, device):
    exact = {3: [1, 2, 1], 5: [1, 4, 6, 4, 1], 7: [2, 7, 14, 18, 14, 7, 2],
             9: [4, 13, 30, 51, 60, 51, 30, 13, 4]}
    if sigma == 0 and size in exact:
        kernel = torch.tensor(exact[size], dtype=torch.float32, device=device)
    else:
        sigma = sigma or (.3*((size-1)*.5-1)+.8)
        axis = torch.arange(size, device=device, dtype=torch.float32)-(size-1)/2
        kernel = (-axis.square()/(2*sigma*sigma)).exp()
    return kernel/kernel.sum()


def gaussian_blur(value, size=3, sigma=0, *, quantize=False):
    kernel = _gaussian_kernel(size, sigma, str(value.device))
    channels = value.shape[1]
    value = F.conv2d(_reflect(value, size//2, -1), kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1), groups=channels)
    value = F.conv2d(_reflect(value, size//2, -2), kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1), groups=channels)
    return value.round().clamp(0, 255) if quantize else value


def _pixel_sample(value, x, y, *, fill=0, quantized_coordinates=True):
    batch, _, height, width = value.shape
    if quantized_coordinates:
        x, y = (x*32).round()/32, (y*32).round()/32
    x, y = torch.broadcast_tensors(x, y)
    grid = torch.stack(((x+.5)*2/width-1, (y+.5)*2/height-1), -1).expand(batch, -1, -1, -1)
    return F.grid_sample(value-fill, grid, align_corners=False, padding_mode='zeros')+fill


def rotate_expanded(value, angle, fill=0):
    """OpenCV-center affine rotation; interpolation uses its 1/32-pixel grid."""
    height, width = value.shape[-2:]
    cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    out_h = max(1, int(height*abs(cosine)+width*abs(sine)))
    out_w = max(1, int(height*abs(sine)+width*abs(cosine)))
    y = torch.arange(out_h, device=value.device, dtype=torch.float32)[:, None]-out_h/2
    x = torch.arange(out_w, device=value.device, dtype=torch.float32)[None, :]-out_w/2
    return _pixel_sample(value, cosine*x-sine*y+width/2, sine*x+cosine*y+height/2, fill=fill)


def fold_image(image, *, fold_x, fold_width, fold_shift, darken=.995, noise=0., rng=None, backdrop=1.):
    """Two-sided author fold at angle zero; caller may rotate around this step."""
    height, width = image.shape[-2:]
    if (fold_width < 1 or fold_shift < 0 or fold_x-fold_width < 0 or fold_x+fold_width > width
            or not 0 <= noise <= 1 or not 0 <= darken <= 1):
        raise ValueError('invalid folding geometry/noise')
    if fold_shift == 0:
        return image.clone()
    if noise and rng is None:
        raise ValueError('folding noise requires a private generator')
    value = _bytes(image)
    result = value.clone()
    x = torch.arange(fold_width, device=image.device, dtype=torch.float32)[None, :]
    y = torch.arange(height, device=image.device, dtype=torch.float32)[:, None]
    for side, start in (('left', fold_x-fold_width), ('right', fold_x)):
        distance = x/fold_width if side == 'left' else 1-x/fold_width
        source = (value[..., start:start+fold_width]*darken).floor()
        source_y = ((y-fold_shift*distance)*32).round()/32
        warped = _pixel_sample(source, x, source_y).floor()
        # Constant-mask sampling through normalized floats can produce
        # 254.99998 and incorrectly turn interior pixels into the backdrop.
        covered = (source_y >= 0) & (source_y <= height-1)
        warped = torch.where(covered, warped, 255*backdrop)
        if noise:
            selected = torch.rand((image.shape[0], 1, height, fold_width), device=image.device, generator=rng) < distance.pow(3)*(noise/2)
            warped = torch.where(selected, 0., warped)
        result[..., start:start+fold_width] = warped
    return result.div(255).to(image.dtype)


def _roller_mask(width, line_width, p, device):
    high, low = p.randint(86, 99), p.randint(70, 85)
    middle2, end3 = p.randint(1, 6), p.randint(1, 6)
    lower_high, lower_low = high+p.randint(-3, 3), low-p.randint(5, 8)
    middle5, end6 = p.randint(1, 6), p.randint(1, 6)
    centers, tails = [0, middle2, 0, 0, middle5, 0], [0, 0, end3, 0, 0, end6]
    lengths = [2*line_width+c+t for c, t in zip(centers, tails)]
    ids, starts, total = [], [], 0
    while total < width:
        chosen = p.randrange(6)
        ids.append(chosen); starts.append(total); total += lengths[chosen]
    repeats = torch.tensor([lengths[i] for i in ids], device=device)
    segment = torch.repeat_interleave(torch.arange(len(ids), device=device), repeats, output_size=total)[:width]
    kind = torch.tensor(ids, device=device)[segment]
    position = torch.arange(width, device=device)-torch.tensor(starts, device=device)[segment]
    center = torch.tensor(centers, device=device)[kind]
    peak = torch.where(kind < 3, float(high), float(lower_high))
    bottom = torch.where(kind < 3, float(low), float(lower_low))
    fraction = torch.where(position < line_width, position, 2*line_width+center-1-position).clamp(0, max(1, line_width-1))/max(1, line_width-1)
    return (peak+(bottom-peak)*fraction).reshape(1, 1, 1, width)


def roller_blend(image, mask, meta_mask, *, dark_background=False):
    effective = (mask*meta_mask/100).clamp_min(0) if dark_background else (mask*(2-meta_mask/100)).clamp_max(99)
    gain = 2-effective/100 if dark_background else effective/100
    return (_bytes(image)*gain).clamp(0, 255).floor().div(255).to(image.dtype)


def dirty_rollers(image, p, *, dark_background=False, line_width=None, rotate=None):
    line_width = line_width or p.randint(2, 32)
    rotate = p.choice((True, False)) if rotate is None else rotate
    value = image.transpose(-1, -2).flip(-1) if rotate else image
    width = value.shape[-1]
    mask = _roller_mask(width, line_width, p, image.device)
    meta = _roller_mask(width, line_width*p.randint(10, 25), p, image.device)
    result = roller_blend(value, mask, meta, dark_background=dark_background)
    return result.flip(-1).transpose(-1, -2) if rotate else result


def _cluster_points(counts, centers, stds, batch, rng, device):
    """Generate ragged Gaussian clusters with a known output size (no readback)."""
    total = sum(counts)
    repeats = torch.tensor(counts, device=device)
    mean = torch.tensor(centers, device=device, dtype=torch.float32)
    sigma = torch.tensor(stds, device=device, dtype=torch.float32)
    labels = torch.repeat_interleave(torch.arange(len(counts), device=device), repeats, output_size=total)
    values = mean[labels]+sigma[labels]*torch.randn((batch, total), device=device, generator=rng)
    return values.to(torch.int64)


def _scatter_last(shape, x, y, values):
    """Integer max of (write order, value) reproduces last-write semantics."""
    batch, height, width = shape
    index = y*width+x
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    index = torch.where(valid, index, height*width)
    out = torch.zeros((batch, height*width+1), device=x.device, dtype=torch.int64)
    out.scatter_reduce_(1, index, values.expand(batch, -1), reduce='amax', include_self=True)
    out = out[:, :-1].reshape(batch, 1, height, width)
    return torch.where(out == 0, 255, out.remainder(256)).float()


def _drum_mask(shape, p, rng, device, *, concentration=.1, intensity=.5):
    batch, height, width = shape
    x, last, stripe = 0, False, 0
    counts, centers_x, centers_y, sigmas, values, boundaries = [], [], [], [], [], []
    current = p.randint(1, 4)*p.randint(1, 5)
    while True:
        if p.random() > 1-concentration and current > 0:
            span = current*2
            deviation = max(1, int(min(span, height)/10))
            count_range = [max(int(span*height*intensity/70)+sign*deviation, 1) for sign in (-1, 1)]
            clusters = p.randint(*[max(int(span*height*intensity/150)+sign*deviation, 1) for sign in (-1, 1)])
            sigma = p.randint(max(int(span/2)-deviation, 1), max(int(span/2)+deviation, 1))
            level = p.randint(0, 30)
            stripe += 1
            for _ in range(clusters):
                counts.append(p.randint(*count_range)); centers_x.append(x+span//2)
                centers_y.append(p.uniform(0, height)); sigmas.append(sigma)
                values.append(stripe*256+level); boundaries.append((x+span)*3)
        x += current*p.randint(1, 3)
        current = p.randint(1, 4)*p.randint(1, 5)
        if x+current > width-1:
            current = int((width-1-x)/2)
            if last:
                break
            last = True
    if not counts:
        return torch.full((batch, 1, height, width), 255., device=device)
    xx = _cluster_points(counts, centers_x, sigmas, batch, rng, device)
    yy = _cluster_points(counts, centers_y, sigmas, batch, rng, device)
    repeats = torch.tensor(counts, device=device)
    # Source make_blobs shuffles dimensions independently; shuffle within each
    # stripe's samples, retaining that stripe's write value and bounds.
    offset = 0
    for key in sorted(set(values)):
        length = sum(n for n, v in zip(counts, values) if v == key)
        yy[:, offset:offset+length] = yy[:, offset:offset+length][:, torch.randperm(length, device=device, generator=rng)]
        offset += length
    end = torch.repeat_interleave(torch.tensor(boundaries, device=device), repeats, output_size=sum(counts))
    xx = torch.where(xx < end, xx, -1)
    levels = torch.repeat_interleave(torch.tensor(values, device=device), repeats, output_size=sum(counts))[None]
    return _scatter_last(shape, xx, yy, levels)


def dirty_drum(image, p, rng, *, direction=None):
    direction = p.randrange(3) if direction is None else direction
    if direction not in (0, 1, 2):
        raise ValueError('drum direction must be horizontal, vertical or both')
    masks = []
    for horizontal in ((True, False) if direction == 2 else (direction == 0,)):
        mask = _drum_mask((image.shape[0], *image.shape[-2:]), p, rng, image.device)
        if horizontal:
            mask = _resize(torch.rot90(mask, p.choice((1, 3)), (-2, -1)), image.shape[-2:], quantize=True)
        masks.append(gaussian_blur(mask, quantize=True))
    mask = torch.minimum(*masks) if len(masks) == 2 else masks[0]
    return torch.minimum(_bytes(image), mask).div(255).to(image.dtype)


FAX_METHODS = ('mean', 'otsu', 'li', 'triangle', 'sauvola')


def _byte_histogram(gray):
    """Exact tiled integer counts, avoiding atomic contention on white paper."""
    values = gray.to(torch.int64).flatten(1)
    padding = (-values.shape[1]) % 1024
    values = F.pad(values, (0, padding)).reshape(values.shape[0], -1, 1024)
    counts = torch.zeros((*values.shape[:2], 256), device=gray.device, dtype=torch.int64)
    counts.scatter_add_(2, values, torch.ones_like(values))
    histogram = counts.sum(1)
    histogram[:, 0] -= padding
    return histogram


def grayscale_threshold(gray, method):
    """The five algorithms in Faxify's random mode, without dynamic eval."""
    if method not in FAX_METHODS:
        raise ValueError('unsupported fax threshold')
    if method == 'sauvola':
        padded = _reflect(_reflect(gray, 7, -1), 7, -2)
        mean = F.avg_pool2d(padded, 15, stride=1)
        std = (F.avg_pool2d(padded.square(), 15, stride=1)-mean.square()).clamp_min(0).sqrt()
        return mean*(1+.2*(std/127.5-1))
    histogram = _byte_histogram(gray)
    counts = histogram.double()
    integer_bins = torch.arange(256, device=gray.device, dtype=torch.int64)[None]
    bins = integer_bins.double()
    total = counts.sum(1)
    low = (counts > 0).to(torch.int64).argmax(1)
    high = 255-(counts.flip(1) > 0).to(torch.int64).argmax(1)
    average = (counts*bins).sum(1)/total
    if method == 'mean':
        threshold = average
    elif method == 'otsu':
        # CUDA float cumsum is rejected by deterministic mode in Torch 2.1.
        # Counts and byte-valued moments are exact integers; convert only
        # after the prefix sums, before computing between-class variance.
        weight = histogram.cumsum(1).double()
        moment = (histogram*integer_bins).cumsum(1).double()
        reverse_weight = total[:, None]-weight
        reverse_moment = moment[:, -1:]-moment
        variance = weight*reverse_weight*(moment/weight.clamp_min(1)-reverse_moment/reverse_weight.clamp_min(1)).square()
        threshold = variance[:, :-1].argmax(1)
    elif method == 'triangle':
        peak = counts.argmax(1)
        flip = peak-low < high-peak
        counts = torch.where(flip[:, None], counts.flip(1), counts)
        peak, low2 = torch.where(flip, 255-peak, peak), torch.where(flip, 255-high, low)
        height = counts.gather(1, peak[:, None])
        length = height*(bins-low2[:, None])-(peak-low2)[:, None]*counts
        length = length.masked_fill((bins < low2[:, None]) | (bins >= peak[:, None]), -torch.inf)
        threshold = length.argmax(1)
        threshold = torch.where(flip, 255-threshold, threshold)
    else:
        # A byte image has only 256 possible histogram partitions. Compute
        # Li's next threshold for every partition once. Its monotone finite
        # transition map reaches a terminal state within 256 steps; eight
        # pointer-doubling gathers replace hundreds of tiny CUDA kernels.
        back_weight = histogram.cumsum(1).double()
        moment = (histogram*integer_bins).cumsum(1).double()-low[:, None]*back_weight
        front_weight = total[:, None]-back_weight
        mean_back = moment/back_weight.clamp_min(1)
        mean_front = (moment[:, -1:]-moment)/front_weight.clamp_min(1)
        blocked = (mean_back == 0) | (front_weight == 0)
        denominator = mean_back.clamp_min(1e-100).log()-mean_front.clamp_min(1e-100).log()
        next_value = torch.where(blocked, bins, (mean_back-mean_front)/denominator.clamp_max(-1e-100)+low[:, None])
        next_partition = next_value.floor().long().clamp(0, 255)
        after = next_value.gather(1, next_partition)
        next_blocked = blocked.gather(1, next_partition)
        terminal = next_blocked | ((after-next_value).abs() <= .5)
        terminal_value = torch.where(next_blocked, next_value, after)
        pointers = torch.where(terminal, integer_bins, next_partition)
        for _ in range(8):
            pointers = pointers.gather(1, pointers)
        start = average.floor().long().clamp(0, 255)[:, None]
        first = next_value.gather(1, start).squeeze(1)
        threshold = terminal_value.gather(1, pointers.gather(1, start)).squeeze(1)
        threshold = torch.where((first-average).abs() <= .5, first, threshold)
        threshold = torch.where(blocked.gather(1, start).squeeze(1), average, threshold)
    threshold = torch.where(low == high, low, threshold)
    return threshold.to(gray.dtype).reshape(-1, 1, 1, 1)


def halftone_image(gray, *, half_kernel=2, angle=45, sigma=2):
    """Rotate, block-average times blurred impulse, rotate back, center crop."""
    kernel_size = 2*half_kernel+1
    rotated = rotate_expanded(gray, angle)
    height, width = rotated.shape[-2:]
    impulse = torch.zeros((1, 1, kernel_size, kernel_size), device=gray.device)
    impulse[:, :, half_kernel, half_kernel] = 1
    kernel = gaussian_blur(impulse, kernel_size, sigma)
    kernel = kernel/kernel.amax()
    if min(height, width) < kernel_size:
        raise ValueError('halftone kernel exceeds the image')
    means = F.avg_pool2d(rotated, kernel_size, stride=kernel_size)
    out_h, out_w = means.shape[-2]*kernel_size, means.shape[-1]*kernel_size
    tiled = means.repeat_interleave(kernel_size, -2).repeat_interleave(kernel_size, -1)
    tiled = tiled*kernel.repeat(1, 1, means.shape[-2], means.shape[-1])
    tiled = F.pad(tiled, (0, width-out_w, 0, height-out_h))
    restored = rotate_expanded(tiled, -angle)
    y, x = [(a-b)//2 for a, b in zip(restored.shape[-2:], gray.shape[-2:])]
    return restored[..., y:y+gray.shape[-2], x:x+gray.shape[-1]]


def faxify(image, *, scale=1.25, method='otsu', halftone=True, half_kernel=2, angle=45, sigma=2, invert=True):
    if not math.isfinite(scale) or scale < 1 or half_kernel < 1 or sigma <= 0:
        raise ValueError('invalid fax scale/kernel')
    height, width = image.shape[-2:]
    size = (max(2, int(height//scale)), max(2, int(width//scale)))
    value = _resize(_bytes(image), size, quantize=True)
    if method is not None:
        gray = _gray(value)
        value = (gray > grayscale_threshold(gray, method)).float()*255
    if halftone:
        gray = (255-value.amax(1, keepdim=True))/255
        if not invert:
            gray = 1-gray
        value = halftone_image(gray, half_kernel=half_kernel, angle=angle, sigma=sigma)
        value = ((1-value if invert else value)*255).clamp(0, 255).floor()
    value = _resize(value, (height, width), quantize=True).div(255)
    return value.expand(-1, 3, -1, -1).to(image.dtype)


NOISE_SIDES = ('none', 'all', 'left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right')


def _normalize(value):
    low = value.amin((-1, -2), keepdim=True)
    return (value-low)/(value.amax((-1, -2), keepdim=True)-low).clamp_min(1e-12)


def perlin_noise(x, y, permutation):
    """Author's four axis-aligned gradients and quintic interpolation."""
    xi, yi = x.long(), y.long()
    xx, yy = x-xi, y-yi
    fx, fy = 6*xx**5-15*xx**4+10*xx**3, 6*yy**5-15*yy**4+10*yy**3
    def gradient(dx, dy):
        code = permutation[permutation[xi+dx]+yi+dy].remainder(4)
        return torch.where(code == 0, yy-dy, torch.where(code == 1, dy-yy, torch.where(code == 2, xx-dx, dx-xx)))
    bottom = torch.lerp(gradient(0, 0), gradient(1, 0), fx)
    top = torch.lerp(gradient(0, 1), gradient(1, 1), fx)
    return torch.lerp(bottom, top, fy)


def worley_noise(height, width, points):
    """Nearest-point distance, chunked over sites to bound memory."""
    x = torch.arange(width, device=points.device)[None, None, None, :]
    y = torch.arange(height, device=points.device)[None, None, :, None]
    distance = torch.full((points.shape[0], height, width), torch.inf, device=points.device)
    for start in range(0, points.shape[1], 8):
        current = points[:, start:start+8]
        candidate = (x-current[:, :, 0, None, None]).square()+(y-current[:, :, 1, None, None]).square()
        distance = torch.minimum(distance, candidate.amin(1))
    return distance.sqrt()[:, None]


def _blob_noise(shape, p, rng, device, *, rectangular=False, side='none', sparsity=.5):
    batch, height, width = shape
    end_y = height if not rectangular or side == 'none' else max(1, int(height*sparsity))
    maximum = max(width, end_y)
    n_clusters = p.randint(max(1, int(.4*maximum)), max(1, int(.6*maximum)))
    base_counts = [p.randint(max(10, int(.4*maximum)), max(10, int(.6*maximum))) for _ in range(n_clusters)]
    std = max(50, p.randint(int(.4*maximum), int(.6*maximum))//5) if not rectangular else p.randint(int(.4*maximum/5), int(.6*maximum/5))//25
    groups = []
    if rectangular:
        step_x, step_y = max(1, width//p.randint(3, 6)), max(1, end_y//p.randint(8, 16))
        count_index, y0 = max(1, int(len(base_counts)*.05)), 0
        while y0 < end_y:
            count_index = max(1, int(count_index*(p.uniform(.5, 1.) if side == 'none' else .8)))
            base_counts = base_counts[:count_index]
            x0 = 0
            while x0 < width:
                groups.append((list(base_counts), (x0, x0+step_x), (y0, y0+step_y)))
                x0 += 2*step_x+p.randint(5, 10)
            y0 += step_y+p.randint(5, 15)
    else:
        groups.append((base_counts, (-std, width+std), (-std, height+std)))
    counts, xs, ys = [], [], []
    for group, xr, yr in groups:
        counts.extend(group)
        xs.extend(p.uniform(*xr) for _ in group)
        ys.extend(p.uniform(*yr) for _ in group)
    sigmas = [std]*len(counts)
    xx = _cluster_points(counts, xs, sigmas, batch, rng, device)
    yy = _cluster_points(counts, ys, sigmas, batch, rng, device)
    start = 0
    for group, _, _ in groups:
        length = sum(group)
        yy[:, start:start+length] = yy[:, start:start+length][:, torch.randperm(length, device=device, generator=rng)]
        start += length
    coverage = _scatter_last(shape, xx, yy, torch.full((1, sum(counts)), 256, device=device, dtype=torch.int64)) == 0
    noise = torch.randint(0, 129, (batch, 1, height, width), device=device, generator=rng).float()
    mask = torch.where(coverage, noise, 255.)
    if rectangular:
        if side in ('left', 'right', 'bottom', 'bottom_left', 'bottom_right'):
            turns = 1 if side == 'left' else 3 if side == 'right' else 2
            mask = _resize(torch.rot90(mask, turns, (-2, -1)), (height, width), quantize=True)
        elif side == 'all':
            for turns in (1, 2, 3):
                mask = torch.minimum(mask, _resize(torch.rot90(mask, turns, (-2, -1)), (height, width), quantize=True))
    return mask


def _side_fade(mask, side, sparsity):
    if side == 'none':
        return mask
    height, width = mask.shape[-2:]
    nh, nw = max(1, int(height*sparsity)), max(1, int(width*sparsity))
    count = math.ceil(1.3*nh)
    ramp = (torch.arange(count, device=mask.device, dtype=torch.float32)/nh-.3).clamp_min(0)
    top = _resize(ramp.reshape(1, 1, -1, 1), (nh, width))*255
    opacity = torch.full((1, 1, height, width), 255., device=mask.device)
    if side in ('top', 'all'):
        opacity[..., :nh, :] = top
    if side in ('bottom', 'all'):
        opacity[..., -nh:, :] = torch.minimum(opacity[..., -nh:, :], top.flip(-2))
    if side in ('left', 'all'):
        opacity[..., :nw] = torch.minimum(opacity[..., :nw], _resize(torch.rot90(top, 1, (-2, -1)), (height, nw)))
    if side in ('right', 'all'):
        opacity[..., -nw:] = torch.minimum(opacity[..., -nw:], _resize(torch.rot90(top, 3, (-2, -1)), (height, nw)))
    if '_' in side:
        turns = 3 if side.endswith('left') else 1
        corner = 255*(1-_normalize(top*_resize(torch.rot90(top, turns, (-2, -1)), top.shape[-2:])))
        if side.startswith('top'):
            opacity[..., :nh, :] = corner.flip(-2)
        else:
            opacity[..., -nh:, :] = corner
    return 255-((255-mask)*(255-opacity.floor())/255).round()


def photocopy_noise(shape, p, rng, device, *, noise_type=1, side='none'):
    """Five author noise families; private sampling and guarded degeneracies."""
    if noise_type not in (1, 2, 3, 4, 5) or side not in NOISE_SIDES:
        raise ValueError('invalid photocopy noise family or side')
    batch, height, width = shape
    sparsity = p.uniform(.4, .6)
    if noise_type in (1, 5):
        mask = _blob_noise(shape, p, rng, device, rectangular=noise_type == 5, side=side, sparsity=sparsity)
    elif noise_type == 2:
        mask = torch.full((batch, 1, height, width), 255., device=device)
        for _ in range(p.randint(2, 3)):
            mean, sigma, ratio = p.randint(0, 255), p.randint(0, 255), p.randint(5, 10)
            noise = mean+sigma*torch.randn((batch, 1, max(1, width//ratio), max(1, height//ratio)), device=device, generator=rng)
            mask += _resize(noise, (height, width))
        mask = mask.trunc().remainder(256)
        background = mask >= int(p.uniform(.4, .6)*255)
        mask = torch.where(background, mask.amin((-1, -2), keepdim=True), mask)
        mask = (_normalize(mask)*128).floor()
        mask = torch.where(background, 255., mask)
    elif noise_type == 3:
        hh, ww = max(2, height//2), max(2, width//2)
        x = torch.arange(ww, device=device, dtype=torch.float32)[None, :]*p.randrange(8, 13)/ww
        y = torch.arange(hh, device=device, dtype=torch.float32)[:, None]*p.randrange(8, 13)/hh
        permutation = torch.randperm(256, device=device, generator=rng).repeat(2)
        mask = perlin_noise(x, y, permutation)[None, None].expand(batch, -1, -1, -1)
        mask = (_normalize(_resize(mask, (height, width)))*128).floor()
    else:
        hh, ww = max(2, height//2), max(2, width//2)
        npoints = max(1, int((p.uniform(.4, .6)+p.uniform(.4, .6))*250))
        points = torch.stack((torch.randint(ww+1, (batch, npoints), device=device, generator=rng),
                              torch.randint(hh+1, (batch, npoints), device=device, generator=rng)), -1).float()
        mask = (_normalize(_resize(worley_noise(hh, ww, points), (height, width)))*128).floor()
    return _side_fade(mask, side, sparsity).clamp(0, 255)


def _wave_noise(mask, side, p):
    height, width = mask.shape[-2:]
    n = p.randint(6, 12)
    points = [(0., p.uniform(height//12, height*3//4))]
    points += [((i-1)*(width-1)/(n-1), p.uniform(height//12, height*3//4)) for i in range(1, n-1)]
    points += [(float(width-1), p.uniform(height//12, height*3//4))]
    curve = torch.tensor(points, device=mask.device, dtype=torch.float64)
    for _ in range(12):
        pairs = torch.stack((.75*curve[:-1]+.25*curve[1:], .25*curve[:-1]+.75*curve[1:]), 1).flatten(0, 1)
        curve = torch.cat((curve[:1], pairs, curve[-1:]))
    heights = torch.zeros(width, device=mask.device, dtype=torch.int64)
    heights.scatter_reduce_(0, curve[:, 0].long().clamp(0, width-1), curve[:, 1].long(), reduce='amax', include_self=True)
    wave = (torch.arange(height, device=mask.device)[:, None] < heights).float()[None, None]*255
    divisor, size = p.randint(2, 3), p.randrange(151, 302, 2)
    wave = _resize(wave, (max(100, height//divisor), max(100, width//divisor)), quantize=True)
    wave = gaussian_blur(wave, size, quantize=True)
    wave = _resize(wave, (height, width), quantize=True)
    if side == 'bottom':
        wave = wave.flip(-2)
    elif side in ('left', 'right'):
        wave = F.interpolate(torch.rot90(wave, 1 if side == 'left' else 3, (-2, -1)), (height, width), mode='area').round()
    return 255-(wave*(255-mask)/255).round()


def _scharr(value):
    kernel = value.new_tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).reshape(1, 1, 3, 3)
    padded = _reflect(_reflect(value, 1, -1), 1, -2)
    return (F.conv2d(padded, kernel)-F.conv2d(padded, kernel.transpose(-1, -2))).abs().round().clamp(0, 255)


def photocopy_from_mask(image, mask, *, edge_effect=False, rng=None):
    value = _bytes(image)
    result = (value*mask/255).round()
    if edge_effect:
        if rng is None:
            raise ValueError('photocopy edges require a private generator')
        edge = gaussian_blur(_scharr(_gray(value)), quantize=True)
        edge = torch.where(edge.remainder(255) != 0, 255., edge)
        # Upstream passes tuples (15,15) and (5,5), which OpenCV interprets as
        # 2x1 kernels, not square ones. Preserve the observed column dilation.
        edge = F.max_pool2d(F.pad(edge, (0, 0, 5, 0)), (6, 1), stride=1)
        outline = F.max_pool2d(F.pad(_scharr(edge), (0, 0, 2, 0)), (3, 1), stride=1)
        levels = torch.randint(0, 128, edge.shape, device=image.device, generator=rng).float()
        chosen = (outline == 255) & (torch.rand(edge.shape, device=image.device, generator=rng) < .7)
        edge = gaussian_blur(torch.where(chosen, levels, edge), 5, quantize=True)
        result = torch.minimum(value, result+edge)
    return result.clamp(0, 255).div(255).to(image.dtype)


def bad_photocopy(image, p, rng, *, noise_type=None, side=None, blur=None, wave=None, edge=None):
    noise_type = p.randint(1, 5) if noise_type is None else noise_type
    side = p.choice(NOISE_SIDES) if side is None else side
    height, width = image.shape[-2:]
    shape = (image.shape[0], max(2, height//p.randint(1, 3)), max(2, width//p.randint(1, 3)))
    mask = photocopy_noise(shape, p, rng, image.device, noise_type=noise_type, side=side)
    mask = _resize(mask, (height, width), mode='bicubic', quantize=True)
    blur, wave, edge = [p.choice((True, False)) if v is None else v for v in (blur, wave, edge)]
    if blur:
        mask = gaussian_blur(mask, 5, quantize=True)
    if wave:
        mask = _wave_noise(mask, side, p)
    if not blur:
        selected = mask > torch.randint(0, 255, mask.shape, device=image.device, generator=rng)
        mask = torch.where(selected, 255., mask)
    return photocopy_from_mask(image, mask, edge_effect=edge, rng=rng)


def apply_paper_operator(image, operator, *, severity, rng, validate=True, texture_bank=None):
    """Registered RGB printing pool; no image readback after texture warmup."""
    if (operator not in PAPER_EVAL_OPERATORS or type(severity) not in (int, float)
            or not math.isfinite(severity) or not 0 <= severity <= 5):
        raise ValueError('invalid print operator/severity')
    if rng is None or torch.device(rng.device) != image.device:
        raise ValueError('print operators require a private generator on the image device')
    if validate:
        validate_paper_image(image)
    if severity == 0:
        return image.clone()
    if operator not in PRINT_OPERATORS:
        return apply_v2(image, operator, severity=severity, rng=rng, validate=False)
    p, strength = _parameter_rng(rng), severity/5
    with torch.autocast(image.device.type, enabled=False):
        if operator == 'quilting_wrinkle':
            bank = load_original_texture_bank(str(image.device)) if texture_bank is None else texture_bank
            if bank.device != image.device or bank.ndim != 4 or bank.shape[1] != 3 or bank.shape[0] < 1:
                raise ValueError('wrinkle bank must be nonempty device-local RGB BCHW')
            selected = torch.randint(bank.shape[0], (image.shape[0],), device=image.device, generator=rng)
            return apply_wrinkle_texture(image, bank[selected], strength)
        if operator == 'folding':
            angle = p.uniform(-15, 15)*strength
            rotated = rotate_expanded(_bytes(image), angle, fill=255).round().div(255)
            for _ in range(2):
                height, width = rotated.shape[-2:]
                half = min(max(1, int(width*p.uniform(.1, .2))), (width-2)//2)
                rotated = fold_image(rotated, fold_x=p.randint(half+1, width-half-1), fold_width=half,
                    fold_shift=max(1, round(height*p.uniform(.01, .02)*strength)),
                    darken=p.uniform(.99, 1.), noise=.01*strength, rng=rng)
            restored = rotate_expanded(rotated, -angle, fill=1.)
            y, x = [(a-b)//2 for a, b in zip(restored.shape[-2:], image.shape[-2:])]
            result = restored[..., y:y+image.shape[-2], x:x+image.shape[-1]]
        elif operator == 'dirty_rollers':
            result = dirty_rollers(image, p)
        elif operator == 'dirty_drum':
            result = dirty_drum(image, p, rng)
        elif operator == 'faxify':
            result = faxify(image, scale=p.uniform(1, 1.25), method=p.choice(FAX_METHODS),
                halftone=p.choice((True, False)), half_kernel=p.randint(1, 2), angle=p.randint(0, 360), sigma=p.randint(1, 3))
        else:
            result = bad_photocopy(image, p, rng)
        if operator != 'folding':
            result = image.float()+strength*(result.float()-image.float())
        return result.clamp(0, 1).to(image.dtype)


def paper_conditions(severities=(1, 2, 3, 4, 5)):
    if not severities or any(type(s) is not int or not 1 <= s <= 5 for s in severities) or len(set(severities)) != len(severities):
        raise ValueError('print stress severities must be unique integers from one to five')
    return [{'condition_id': f'paper_v3/{op}/s{severity}', 'family': 'image',
        'image_operator': op, 'image_severity': severity,
        'stress_group': 'seen_family' if op in PAPER_TRAIN_OPERATORS else 'held_out_family',
        'implementation': PAPER_IMPLEMENTATION} for op in PAPER_EVAL_OPERATORS for severity in severities]
