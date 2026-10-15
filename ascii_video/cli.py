"""Stream decoding/encoding through FFmpeg, image calculations on CUDA."""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from .ffmpeg_io import decoder, encoder, exact_frame, probe
from .processing import GpuAsciiProcessor


def parser():
    p=argparse.ArgumentParser(description='ASCII Video Lab v0.2: GPU-native structure-aware video renderer')
    p.add_argument('input',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--start',type=float,default=0)
    p.add_argument('--duration',type=float)
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--cell-width',type=int,default=8)
    p.add_argument('--cell-height',type=int,default=12)
    p.add_argument('--analysis-scale',type=int,default=2)
    p.add_argument('--font')
    p.add_argument('--mode',choices=['hybrid','luminance','edges'],default='hybrid')
    p.add_argument('--spatial',choices=['guided','gaussian','none'],default='guided')
    p.add_argument('--guided-epsilon',type=float,default=18.,help='Higher means stronger flat-area smoothing')
    p.add_argument('--temporal-alpha',type=float,default=.25)
    p.add_argument('--motion-threshold',type=float,default=8.)
    p.add_argument('--edge-threshold',type=float,default=80.)
    p.add_argument('--edge-source',choices=['current','temporal'],default='current')
    p.add_argument('--stability',type=float,default=5.)
    p.add_argument('--contrast',type=float,default=1.15)
    p.add_argument('--background',choices=['cover','contain'],default='cover')
    p.add_argument('--palette',choices=['white','amber','green','blue'],default='blue')
    p.add_argument('--bg-rgb',type=lambda x:tuple(int(v) for v in x.split(',')),default=(5,8,11))
    p.add_argument('--inset-width',type=int,default=420)
    p.add_argument('--inset-border',type=int,default=6)
    p.add_argument('--encoder',choices=['h264_nvenc','libx264'],default='h264_nvenc')
    p.add_argument('--preset',default='veryfast')
    p.add_argument('--crf',type=int,default=20)
    p.add_argument('--audio',choices=['copy','aac','none'],default='copy')
    p.add_argument('--debug-frame',type=int)
    p.add_argument('--debug-dir',type=Path,default=Path('ascii-debug'))
    p.add_argument('--profile-stages',action='store_true',help='Synchronise CUDA and report time per pipeline stage (slower)')
    p.add_argument('--overwrite',action='store_true')
    return p


def run(a):
    for cmd in ('ffprobe','ffmpeg'):
        if not shutil.which(cmd):
            raise RuntimeError(f'{cmd} not found. Install FFmpeg.')
    a.input=a.input.expanduser().resolve()
    a.output=a.output.expanduser().resolve()
    if not a.input.is_file():
        raise FileNotFoundError(a.input)
    if a.input==a.output:
        raise ValueError('Output cannot overwrite the source')
    if a.output.exists() and not a.overwrite:
        raise FileExistsError(f'{a.output} already exists. Use --overwrite.')
    if min(a.width,a.height,a.cell_width,a.cell_height,a.analysis_scale)<=0:
        raise ValueError('Dimensions and scale must be positive')
    if a.width %2 or a.height%2:
        raise ValueError('Video output dimensions must be even')
    if a.width%a.cell_width or a.height%a.cell_height:
        raise ValueError('Output dimensions must be divisible by cell dimensions')
    if a.inset_width<0 or a.inset_width>=a.width:
        raise ValueError('Inset width must be >= 0 and less than output width')
    if a.inset_border<0 or (a.inset_width and a.height <= 2*(a.inset_border+8)):
        raise ValueError('Inset border/output height is invalid')
    if a.start<0 or (a.duration is not None and a.duration<=0):
        raise ValueError('Start must be >=0; duration must be >0')
    if not 0<=a.temporal_alpha<=1 or a.stability<0 or a.motion_threshold<0 or a.edge_threshold<0:
        raise ValueError('Alpha must be 0..1; thresholds/stability nonnegative')
    if len(a.bg_rgb)!=3 or not all(0<=v<=255 for v in a.bg_rgb):
        raise ValueError('Background RGB values must be 0..255')
    if a.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in PyTorch. Install a CUDA-enabled PyTorch wheel (see README).')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    info=probe(a.input)
    fps=info['fps']
    print(f"Video: {info['width']}x{info['height']} @ {fps} fps | audio: {info['has_audio']}",flush=True)
    if a.audio!='none' and not info['has_audio']:
        a.audio='none'
    proc=GpuAsciiProcessor(a,info['width'],info['height'])
    print(f'Compute device: {proc.device} | atlas grid: {proc.cols}x{proc.rows} | '
          f'analysis: {proc.aw}x{proc.ah} | output: {a.width}x{a.height}',flush=True)
    if a.cell_width<5 or a.cell_height<7:
        print('Note: tiny cells use micro-pixel tiles rather than legible font glyphs.',flush=True)
    temp=a.output.with_name(a.output.stem+'.incomplete'+a.output.suffix)
    log=a.output.with_name(a.output.stem+'.ffmpeg-error.log')
    if temp.exists():
        if a.overwrite: temp.unlink()
        else: raise FileExistsError(f'Interrupted file exists: {temp}. Use --overwrite.')
    dec=decoder(a.input,info['width'],info['height'],fps,a.start,a.duration)
    enc,handle=None,None
    framebytes=info['width']*info['height']*3
    start=time.perf_counter()
    count=0
    ok=False
    try:
        enc,handle=encoder(a.input,temp,a.width,a.height,fps,a.start,a.duration,
                           a.encoder,a.audio,a.preset,a.crf,log)
        while True:
            data=exact_frame(dec.stdout,framebytes)
            if data is None: break
            frame=np.frombuffer(data,dtype=np.uint8).reshape(info['height'],info['width'],3)
            processed,extras=proc.process(frame,debug=a.debug_frame==count)
            if extras:
                a.debug_dir.mkdir(parents=True,exist_ok=True)
                for name,image in extras.items():
                    if image.ndim==3:
                        image=image[:,:,::-1] # BGR -> RGB for Pillow
                    Image.fromarray(image).save(a.debug_dir/(name+'.png'))
                print(f'Debug images: {a.debug_dir}',flush=True)
            enc.stdin.write(processed.tobytes())
            count+=1
            if count%100==0:
                dt=time.perf_counter()-start
                print(f'  {count} frames | {count/dt:.1f} fps end-to-end',flush=True)
        dec.stdout.close()
        err=dec.stderr.read().decode('utf-8','replace')
        if dec.wait()!=0:
            raise RuntimeError('Decoder failed: '+err[-2000:])
        if count==0:
            raise RuntimeError('No decoded frames')
        enc.stdin.close()
        if enc.wait()!=0:
            handle.close();handle=None
            raise RuntimeError('Encoder failed: '+log.read_text(errors='replace')[-2400:])
        handle.close();handle=None
        if not temp.exists() or temp.stat().st_size==0:
            raise RuntimeError('No output produced')
        out_info=probe(temp)
        if (out_info['width'],out_info['height'])!=(a.width,a.height):
            raise RuntimeError('Unexpected output dimensions')
        os.replace(temp,a.output)
        ok=True
        print(f'Finished {count} frames in {time.perf_counter()-start:.2f}s '
              f'({count/(time.perf_counter()-start):.2f} end-to-end fps)\n'
              f'Saved: {a.output}',flush=True)
        if a.device=='cuda':
            print(f'Peak allocated CUDA memory: {torch.cuda.max_memory_allocated()/2**20:.1f} MiB',flush=True)
        if a.profile_stages:
            print('GPU stage timings (instrumented, slower):\n'+proc.report(),flush=True)
    finally:
        if dec.poll() is None: dec.kill()
        dec.wait()
        if enc is not None:
            if enc.stdin and not enc.stdin.closed:
                enc.stdin.close()
            if enc.poll() is None: enc.kill()
            enc.wait()
        if handle is not None: handle.close()
        if ok and log.exists() and log.stat().st_size==0: log.unlink()


def main():
    try:
        run(parser().parse_args())
    except (RuntimeError,ValueError,FileNotFoundError,FileExistsError,OSError,BrokenPipeError) as e:
        print(f'Error: {e}',file=sys.stderr)
        sys.exit(1)
