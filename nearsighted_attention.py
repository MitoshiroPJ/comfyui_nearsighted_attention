
import math
import time
import torch
from random import Random
from torch import nn
from einops import rearrange, repeat

REDUCTION_MODES = ["1D_AVG", "1D_MAX", "2D_AVG", "2D_MAX"]

def clip(num, min_value, max_value):
  return max(min(num, max_value), min_value)

def lerp(a, b, ratio):
  return a * (1 - ratio) + b * ratio

def get_time_ratio(time_center, time_decay, timestep):
  if time_decay <= 0:
    return 1
  time = clip(1 - (timestep / 1000), 0, 1) # 1000->0, 0->1
  sig2 = (1 / time_decay) ** 2
  return math.exp(-((time - time_center) ** 2) / sig2)

def pooling(q, size, stride, mode, k_blend, v_blend, h, w):
  if stride <= 1: # skip pooling
    return q, q, q

  if mode == "1D_AVG" or mode == "1D_MAX":
    stride = (clip(math.ceil(stride), 1, q.shape[1]), 1)
    size = (clip(math.ceil(size), 1, q.shape[1]), 1)
    one = nn.functional.avg_pool2d(q, (1, 1), stride=stride)

    if k_blend == 0 and v_blend == 0:
      return q, one, one
    
    pool = nn.functional.avg_pool2d(q, size, stride=stride) if mode == "1D_AVG" else nn.functional.max_pool2d(q, size, stride=stride)

    # align shape
    one = nn.functional.pad(one, (0, 0, 0, pool.shape[1] - one.shape[1]))

    k = torch.lerp(one, pool, k_blend)
    v = torch.lerp(one, pool, v_blend) if k_blend != v_blend else k
    del pool, one
    return q, k, v
  
  else: # 2D

    sty = clip(math.ceil(math.sqrt(stride)), 1, h)
    stx = clip(math.ceil(stride / sty), 1, w)
    stride = (sty, stx, 1)
    szy = clip(math.ceil(math.sqrt(size)), 1, h)
    szx = clip(math.ceil(size / szy), 1, w)
    size = (szy, szx, 1)

    k2d = rearrange(q, "b (h w) c -> b h w c", h=h, w=w)

    one2d = nn.functional.avg_pool3d(k2d, (1, 1, 1), stride=stride)

    if k_blend == 0 and v_blend == 0:
      one = rearrange(one2d, "b h w c -> b (h w) c", h=one2d.shape[1], w=one2d.shape[2])
      return q, one, one

    pool2d = nn.functional.avg_pool3d(k2d, size, stride=stride) if mode == "2D_AVG" else nn.functional.max_pool3d(k2d, size, stride=stride)

    # align shape
    one2d = nn.functional.pad(one2d, (0, 0, 0, pool2d.shape[2] - one2d.shape[2], 0, pool2d.shape[1] - one2d.shape[1]))

    one = rearrange(one2d, "b h w c -> b (h w) c", h=one2d.shape[1], w=one2d.shape[2])
    pool = rearrange(pool2d, "b h w c -> b (h w) c", h=pool2d.shape[1], w=pool2d.shape[2])

    k = torch.lerp(one, pool, k_blend)
    v = torch.lerp(one, pool, v_blend) if k_blend != v_blend else k
    del pool, one, pool2d, one2d, k2d

    return q, k, v


class SlothfulAttention:
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "peak_time": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
          "time_decay": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.5}),

          "in_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "in_depth_decay": ("FLOAT", {"default": 2.0, "min": 0, "max": 4.0, "step": 0.5}),
          "in_slothful": ("FLOAT", {"default": 6.0, "min": 0, "max": 50.0, "step": 0.5}),
          "in_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "in_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),

          "out_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "out_depth_decay": ("FLOAT", {"default": 2.0, "min": 0, "max": 4.0, "step": 0.5}),
          "out_slothful": ("FLOAT", {"default": 4.0, "min": 0, "max": 50.0, "step": 0.5}),
          "out_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "out_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
        }
      }
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model"

  CATEGORY = "NearSightedAttention"

  def patch_model(self, model, peak_time, time_decay,
                  in_mode, in_depth_decay, in_slothful, in_k_blend, in_v_blend,
                  out_mode, out_depth_decay, out_slothful, out_k_blend, out_v_blend):
    model_channels = model.model.model_config.unet_config["model_channels"]

    def attn_patch(q, k, v, extra_options):
      if extra_options['block'][0] == 'middle':
        return q, k, v

      timestep = model.model.model_sampling.timestep(extra_options['sigmas'][0]).item()
      depth = q.shape[2] // model_channels # depth^2

      original_shape = extra_options["original_shape"]
      sample_ratio = math.sqrt(q.shape[1] / (original_shape[2] * original_shape[3]))
      w = math.ceil(original_shape[3] * sample_ratio)
      h = q.shape[1] // w
      if w * h != q.shape[1]:
        w = math.floor(original_shape[3] * sample_ratio)
        h = q.shape[1] // w

      is_output = extra_options['block'][0] == 'output'
      mode = out_mode if is_output else in_mode
      depth_decay = out_depth_decay if is_output else in_depth_decay      
      slothful = in_slothful if is_output else out_slothful
      k_blend = out_k_blend if is_output else in_k_blend
      v_blend = out_v_blend if is_output else in_v_blend

      is_xl = extra_options['n_heads'] != 8
      depth_rate = 2 / depth if is_xl else 1 / depth
      depth_ratio = (depth_rate ** depth_decay)
      time_ratio = get_time_ratio(peak_time, time_decay, timestep)

      stride = slothful * time_ratio * depth_ratio

      if stride <= 1: # skip pooling
        return q, k, v

      return pooling(q, stride, stride, mode, k_blend, v_blend, h, w)

    model_patched = model.clone()
    model_patched.set_model_attn1_patch(attn_patch)

    return (model_patched,)

class NearSightedAttention:
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "tiling_max_depth": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
          "peak_time": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
          "time_decay": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.5}),
          "base_tile_size": ("INT", {"default": 32, "min": 16, "max": 96, "step": 2}),
          "peak_tile_size": ("INT", {"default": 48, "min": 16, "max": 96, "step": 2}),
          "base_global_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
          "peak_global_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
          "in_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "in_depth_decay": ("FLOAT", {"default": 2, "min": 0, "max": 4.0, "step": 0.5}),
          "in_slothful": ("FLOAT", {"default": 6.0, "min": 0, "max": 50.0, "step": 0.5}),
          "in_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "in_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "out_mode": (REDUCTION_MODES, {"default": "2D_AVG"}),
          "out_depth_decay": ("FLOAT", {"default": 2, "min": 0, "max": 4.0, "step": 0.5}),
          "out_slothful": ("FLOAT", {"default": 4.0, "min": 0, "max": 50.0, "step": 0.5}),
          "out_k_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "out_v_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "concat_local": ("BOOLEAN", {"default": False}),
          "concat_global": ("BOOLEAN", {"default": False}),
        }
      }
  
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model"

  CATEGORY = "NearSightedAttention"

  def patch_model(self, model, tiling_max_depth, peak_time, time_decay, 
                  base_tile_size, peak_tile_size, base_global_ratio, peak_global_ratio,
                  in_mode, in_depth_decay, in_slothful, in_k_blend, in_v_blend,
                  out_mode, out_depth_decay, out_slothful, out_k_blend, out_v_blend,
                  concat_local, concat_global):
    model_channels = model.model.model_config.unet_config["model_channels"]

    # (ty, tx, th, tw, py, px, sy, sx)
    self.temp = None
    self.rand = Random()
    self.last_block_name = None

    def attn_patch_in(q, k, v, extra_options):
      timestep = model.model.model_sampling.timestep(extra_options['sigmas'][0]).item()
      depth = q.shape[2] // model_channels # depth^2

      # initilize seed on beginning of steps
      block_name = extra_options['block'][0]
      if block_name != self.last_block_name and block_name == 'input':
        self.rand.seed(int(timestep))
      self.last_block_name = block_name

      original_shape = extra_options["original_shape"]
      sample_ratio = math.sqrt(q.shape[1] / (original_shape[2] * original_shape[3]))
      w = math.ceil(original_shape[3] * sample_ratio)
      h = q.shape[1] // w
      if w * h != q.shape[1]:
        w = math.floor(original_shape[3] * sample_ratio)
        h = q.shape[1] // w
  
      is_output = extra_options['block'][0] == 'output'
      mode = out_mode if is_output else in_mode
      depth_decay = out_depth_decay if is_output else in_depth_decay
      slothful = out_slothful if is_output else in_slothful
      k_blend = out_k_blend if is_output else in_k_blend
      v_blend = out_v_blend if is_output else in_v_blend

      time_ratio = get_time_ratio(peak_time, time_decay, timestep)
      tile_size = round(lerp(base_tile_size, peak_tile_size, time_ratio))
      ty, tx = math.ceil(h / tile_size), math.ceil(w / tile_size)


      is_xl = extra_options['n_heads'] != 8
      depth_rate = 2 / depth if is_xl else 1 / depth

      stride = lerp(0, slothful, time_ratio * depth_rate ** depth_decay)
      global_ratio = lerp(base_global_ratio, peak_global_ratio, time_ratio)

      depth = q.shape[2] // model_channels # depth^2
      if depth > 2 ** (tiling_max_depth - 1) or ty * tx <= 1:
        return pooling(q, stride, stride, mode, k_blend, v_blend, h, w)

      th, tw = math.ceil(h / ty), math.ceil(w / tx)
      py = ty * th - h
      px = tx * tw - w
      sy = self.rand.randrange(th)
      sx = self.rand.randrange(tw)

      lq = rearrange(q, "b (h w) c -> b h w c", h=h, w=w)
      lq = nn.functional.pad(lq, (0, 0, 0, px, 0, py))
      lq = rearrange(lq, "b (ty th) (tx tw) c -> (b ty tx) (th tw) c", th=th, tw=tw, ty=ty, tx=tx)
      self.temp = (ty, tx, th, tw, py, px, sy, sx)

      _lq, lk, lv = pooling(lq, stride, stride, mode, k_blend, v_blend, th, tw)

      if concat_local:
        cond_or_uncond_size = len(extra_options['cond_or_uncond'])
        batch_size = q.shape[0] // cond_or_uncond_size
        lk = rearrange(lk, "(cuc bs tile) hw c -> (cuc tile) (bs hw) c", cuc=cond_or_uncond_size, bs=batch_size, tile=ty * tx)
        lv = rearrange(lv, "(cuc bs tile) hw c -> (cuc tile) (bs hw) c", cuc=cond_or_uncond_size, bs=batch_size, tile=ty * tx)
        lk = repeat(lk, "(cuc tile) hw c -> (cuc bs tile) hw c", cuc=cond_or_uncond_size, bs=batch_size, tile=ty * tx)
        lv = repeat(lv, "(cuc tile) hw c -> (cuc bs tile) hw c", cuc=cond_or_uncond_size, bs=batch_size, tile=ty * tx)

      if global_ratio <= 0:
        # skip global pooling
        return lq, lk, lv

      # globalの K, V を取得する
      global_stride = (max(1, stride) * ty * tx) / global_ratio

      # pooling size は localと同じで strideのみ大きくして サンプル数を調整
      _gq, gk, gv = pooling(q, stride, global_stride, mode, k_blend, v_blend, h, w)

      if concat_global:
        cond_or_uncond_size = len(extra_options['cond_or_uncond'])
        batch_size = q.shape[0] // cond_or_uncond_size
        gk = rearrange(gk, "(cuc bs) hw c -> cuc (bs hw) c", cuc=cond_or_uncond_size)
        gv = rearrange(gv, "(cuc bs) hw c -> cuc (bs hw) c", cuc=cond_or_uncond_size)
        gk = repeat(gk, "cuc hw c -> (cuc bs tile) hw c", bs=batch_size, tile=ty * tx)
        gv = repeat(gv, "cuc hw c -> (cuc bs tile) hw c", bs=batch_size, tile=ty * tx)
      else:
        gk = repeat(gk, "b hw c -> (b bs) hw c", bs=ty * tx)
        gv = repeat(gv, "b hw c -> (b bs) hw c", bs=ty * tx)

      # concat local and global for k, v
      ck = torch.concat([lk, gk], dim=1)
      cv = torch.concat([lv, gv], dim=1)
      del lk, lv, gk, gv

      return lq, ck, cv

    def attn_patch_out(out, extra_options):
      if self.temp is not None:
        ty, tx, th, tw, py, px, sy, sx = self.temp
        self.temp = None
        h = ty * th - py
        w = tx * tw - px

        out = rearrange(out, "(b ty tx) (th tw) c -> b (ty th) (tx tw) c", th=th, tw=tw, ty=ty, tx=tx)
        out = nn.functional.pad(out, (0, 0, 0, -px, 0, -py))
        out = rearrange(out, "b h w c -> b (h w) c", h=h, w=w)
      return out

    m = model.clone()
    m.set_model_attn1_patch(attn_patch_in)
    m.set_model_attn1_output_patch(attn_patch_out)
    return (m, )


class NearSightedAttentionSimple(NearSightedAttention):
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "tiling_max_depth": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
          "peak_time": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
          "time_decay": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.5}),
          "base_tile_size": ("INT", {"default": 32, "min": 16, "max": 96, "step": 2}),
          "peak_tile_size": ("INT", {"default": 48, "min": 16, "max": 96, "step": 2}),
          "base_global_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
          "peak_global_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
          "slothful": ("FLOAT", {"default": 4.0, "min": 0, "max": 50.0, "step": 0.5}),
          "in_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "out_blend": ("FLOAT", {"default": 0.0, "min": 0, "max": 1.0, "step": 0.05}),
          "concat_local": ("BOOLEAN", {"default": False}),
          "concat_global": ("BOOLEAN", {"default": False}),
        }
      }
  
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model_simple"

  CATEGORY = "NearSightedAttention"

  def patch_model_simple(self, model, tiling_max_depth, peak_time, time_decay, 
                         base_tile_size, peak_tile_size, base_global_ratio, peak_global_ratio, 
                         slothful, in_blend, out_blend, concat_local, concat_global):

    return self.patch_model(model, tiling_max_depth, peak_time, time_decay,
                            base_tile_size, peak_tile_size, base_global_ratio, peak_global_ratio,
                            "2D_AVG", 2, slothful, in_blend, in_blend,
                            "2D_AVG", 2, slothful, out_blend, out_blend, concat_local, concat_global)


class NearSightedTile(NearSightedAttention):
  @classmethod
  def INPUT_TYPES(s):
      return {
        "required": {
          "model": ("MODEL",),
          "tiling_max_depth": ("INT", {"default": 2, "min": 1, "max": 3, "step": 1}),
          "peak_time": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
          "time_decay": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.5}),
          "base_tile_size": ("INT", {"default": 32, "min": 16, "max": 96, "step": 2}),
          "peak_tile_size": ("INT", {"default": 48, "min": 16, "max": 96, "step": 2}),
          "base_global_ratio": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
          "peak_global_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.1}),
          "concat_local": ("BOOLEAN", {"default": False}),
          "concat_global": ("BOOLEAN", {"default": False}),
        }
      }
  RETURN_TYPES = ("MODEL",)
  FUNCTION = "patch_model_near_sighted_tile"

  CATEGORY = "NearSightedAttention"

  def patch_model_near_sighted_tile(self, model, tiling_max_depth, peak_time, time_decay,
                  base_tile_size, peak_tile_size, base_global_ratio, peak_global_ratio,
                  concat_local, concat_global):

    return self.patch_model(model, tiling_max_depth, peak_time, time_decay,
                            base_tile_size, peak_tile_size, base_global_ratio, peak_global_ratio,
                            "1D_AVG", 2, 0, 0, 0,
                            "1D_AVG", 2, 0, 0, 0, 
                            concat_local, concat_global)



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
  "SlothfulAttention": SlothfulAttention,
  "NearSightedAttention": NearSightedAttention,
  "NearSightedAttentionSimple": NearSightedAttentionSimple,
  "NearSightedTile": NearSightedTile,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
  "SlothfulAttention": "Slothful Attention",
  "NearSightedAttention": "Near-sighted Attention",
  "NearSightedAttentionSimple": "Near-sighted Attention (Simple)",
  "NearSightedTile": "Near-sighted Tile",
}

