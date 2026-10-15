"""CUDA-first structure-aware ASCII rendering; frame state stays on the device."""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from .glyphs import GlyphAtlas, RAMP

PALETTE = {
    'white': (238,238,238),
    'amber': (14,185,255),
    'green': (95,240,125),
    'blue': (255,175,75),
}


class GpuAsciiProcessor:
    def __init__(self, args, input_width: int, input_height: int):
        self.a = args
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available to PyTorch. Check your CUDA wheel and NVIDIA driver.')
        self.ow, self.oh = args.width, args.height
        self.cw, self.ch = args.cell_width, args.cell_height
        if self.ow % self.cw or self.oh % self.ch:
            raise ValueError('Output dimensions must be divisible by cell dimensions')
        self.cols, self.rows = self.ow//self.cw, self.oh//self.ch
        # Never expand the analysis beyond the information in the source.
        self.aw = min(self.cols*args.analysis_scale, max(self.cols,input_width))
        self.ah = min(self.rows*args.analysis_scale, max(self.rows,input_height))
        self.atlas = GlyphAtlas(self.cw,self.ch,self.device,args.font)
        self.fg = torch.tensor(PALETTE[args.palette],device=self.device,dtype=torch.float32)
        self.bg = torch.tensor(tuple(reversed(args.bg_rgb)),device=self.device,dtype=torch.float32)
        self.edge_codes = {c:self.atlas.index[c] for c in '|-/\\+'}
        sx = torch.tensor([[-1.,0,1],[-2.,0,2],[-1.,0,1]], device=self.device).reshape(1,1,3,3)
        sy = torch.tensor([[-1.,-2,-1],[0.,0,0],[1,2,1]], device=self.device).reshape(1,1,3,3)
        self.sx,self.sy = sx,sy
        self.prev_clean = None
        self.prev_indices = None
        self.prev_edge = None
        self.prev_anchor = None
        self.timings = {name:0.0 for name in ('upload','preprocess','select','render','composite','download')}
        self.count=0

    def _mark(self, name, begun):
        if self.a.profile_stages:
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            self.timings[name] += time.perf_counter()-begun
        return time.perf_counter()

    def _resize(self, t, h, w):
        if t.shape[-2:] == (h,w):
            return t
        if h < t.shape[-2] and w < t.shape[-1]:
            return F.interpolate(t, size=(h,w), mode='area')
        return F.interpolate(t, size=(h,w), mode='bilinear', align_corners=False)

    def _fit(self, gray):
        a = self.a
        sh,sw = gray.shape[-2:]
        if a.background == 'cover':
            if sw * self.ah > sh * self.aw:
                cw = max(1, round(sh*self.aw/self.ah))
                left = (sw-cw)//2
                gray = gray[:,:,:,left:left+cw]
            else:
                ch = max(1, round(sw*self.ah/self.aw))
                top = (sh-ch)//2
                gray = gray[:,:,top:top+ch,:]
            return self._resize(gray,self.ah,self.aw)
        scale = min(self.aw/sw,self.ah/sh)
        nw,nh = max(1,round(sw*scale)),max(1,round(sh*scale))
        resized=self._resize(gray,nh,nw)
        left,top=(self.aw-nw)//2,(self.ah-nh)//2
        return F.pad(resized, (left,self.aw-nw-left,top,self.ah-nh-top))

    def _spatial(self, gray):
        method = self.a.spatial
        if method == 'none':
            return gray
        if method == 'gaussian':
            # Fast 5x5 separable blur, on device. Reflect-pad avoids dark frame borders.
            k = torch.tensor([1.,4,6,4,1],device=self.device)/16.
            out = F.conv2d(F.pad(gray,(2,2,0,0),mode='replicate'),k.reshape(1,1,1,5))
            return F.conv2d(F.pad(out,(0,0,2,2),mode='replicate'),k.reshape(1,1,5,1))
        if method == 'guided':
            # Edge-aware self-guided smoothing; local contrast protects edges.
            r=2
            mu=F.avg_pool2d(gray,5,1,2,count_include_pad=False)
            variance=(F.avg_pool2d(gray*gray,5,1,2,count_include_pad=False)-mu*mu).clamp_min(0.)
            weight=variance/(variance + self.a.guided_epsilon**2)
            # Smooth the local mean more aggressively on flat areas.
            return mu + weight*(gray-mu)
        raise ValueError(f'Unknown spatial method: {method}')

    def _preprocess(self, image):
        # Image is uint8 GPU HWC BGR. Keep luminance and state on GPU.
        f=image.float()
        gray=(f[:,:,0]*0.114 + f[:,:,1]*0.587 + f[:,:,2]*0.299)[None,None,:,:]
        curr=self._spatial(self._fit(gray))
        if self.prev_clean is None or self.a.temporal_alpha >= 1.0:
            clean=curr
        else:
            diff=(curr-self.prev_clean).abs()
            moving=F.max_pool2d((F.avg_pool2d(diff,3,1,1)>self.a.motion_threshold).float(),3,1,1)>0
            clean=torch.where(moving,curr,
                              self.a.temporal_alpha*curr+(1.-self.a.temporal_alpha)*self.prev_clean)
        self.prev_clean=clean
        return clean,curr

    def _grid(self, frame):
        return self._resize(frame,self.rows,self.cols)

    def _select(self, clean, curr):
        a = self.a
        luma=self._grid(clean)
        raw=luma.squeeze(0).squeeze(0)
        norm=((raw/255. - 0.5)*a.contrast + 0.5).clamp(0,1)
        levels=(norm*(len(RAMP)-1)).round().long().clamp_(0,len(RAMP)-1)
        idx=self.atlas.ramp_indices[levels]
        edge_input = curr if a.edge_source == 'current' else clean
        grid=self._grid(edge_input)
        # Pre-blur only edges so high-frequency shimmer isn't detected as structure.
        blurred=F.avg_pool2d(grid,3,1,1,count_include_pad=False)
        gx=F.conv2d(F.pad(blurred,(1,1,1,1),mode='replicate'),self.sx)[0,0]
        gy=F.conv2d(F.pad(blurred,(1,1,1,1),mode='replicate'),self.sy)[0,0]
        magnitude=torch.sqrt(gx*gx+gy*gy)
        is_edge = magnitude >= a.edge_threshold
        if a.mode == 'luminance':
            is_edge=torch.zeros_like(is_edge)
        else:
            vertical=gx.abs()>1.6*gy.abs()
            horizontal=gy.abs()>1.6*gx.abs()
            slash=gx*gy>0
            edge_idx=torch.where(slash,self.edge_codes['/'],self.edge_codes['\\'])
            edge_idx=torch.where(horizontal,self.edge_codes['-'],edge_idx)
            edge_idx=torch.where(vertical,self.edge_codes['|'],edge_idx)
            if a.mode == 'edges':
                idx=torch.full_like(idx,self.atlas.index[' '])
            idx=torch.where(is_edge,edge_idx,idx)
        if self.prev_indices is not None and a.stability>0:
            hold=(raw-self.prev_anchor).abs()<a.stability
            # Never persist a vanished/new edge or an outdated edge orientation.
            hold=hold & (is_edge==self.prev_edge) & ((~is_edge) | (idx==self.prev_indices))
            idx=torch.where(hold,self.prev_indices,idx)
            self.prev_anchor=torch.where(hold,self.prev_anchor,raw)
        else:
            self.prev_anchor=raw.clone()
        self.prev_indices=idx
        self.prev_edge=is_edge
        return idx,magnitude

    def _render(self, idx):
        mask=self.atlas.render(idx).float().unsqueeze(-1)/255.
        return (self.bg+mask*(self.fg-self.bg)).clamp(0,255).to(torch.uint8)

    def _compose(self, rgb, source):
        a=self.a
        if a.inset_width == 0:
            return rgb
        sh,sw=source.shape[:2]
        iw=a.inset_width
        ih=max(2,round(iw*sh/sw))
        max_h=self.oh-2*(a.inset_border+8)
        if ih>max_h:
            ih=max_h
            iw=max(2,round(ih*sw/sh))
        inset=self._resize(source.permute(2,0,1).unsqueeze(0).float(),ih,iw)
        inset=inset.squeeze(0).permute(1,2,0).round().to(torch.uint8)
        x,y=(self.ow-iw)//2,(self.oh-ih)//2
        b=a.inset_border
        if b:
            rgb[y-b:y+ih+b,x-b:x+iw+b,:]=0
            # Single-pixel light outline around the black border.
            rgb[y-b,x-b:x+iw+b,:]=160
            rgb[y+ih+b-1,x-b:x+iw+b,:]=160
            rgb[y-b:y+ih+b,x-b,:]=160
            rgb[y-b:y+ih+b,x+iw+b-1,:]=160
        rgb[y:y+ih,x:x+iw,:]=inset
        return rgb

    @torch.inference_mode()
    def process(self, frame:np.ndarray, debug=False):
        start=time.perf_counter()
        source=torch.from_numpy(frame).to(self.device,non_blocking=False)
        start=self._mark('upload',start)
        clean,curr=self._preprocess(source)
        start=self._mark('preprocess',start)
        idx,edges=self._select(clean,curr)
        start=self._mark('select',start)
        ascii_img=self._render(idx)
        start=self._mark('render',start)
        final=self._compose(ascii_img.clone() if debug else ascii_img, source)
        start=self._mark('composite',start)
        result=final.contiguous().cpu().numpy()
        extras=None
        if debug:
            def gray_bytes(x):
                return x.clamp(0,255).byte().squeeze().cpu().numpy()
            extras={
                'original':frame,
                'denoised':gray_bytes(clean),
                'edges':gray_bytes(edges),
                'ascii':ascii_img.cpu().numpy(),
                'composite':result,
            }
        self._mark('download',start)
        self.count+=1
        return result, extras

    def report(self):
        if not self.a.profile_stages or not self.count:
            return ''
        return '\n'.join(f'  {k:12s} {v*1000/self.count:7.2f} ms/frame'
                         for k,v in self.timings.items())
