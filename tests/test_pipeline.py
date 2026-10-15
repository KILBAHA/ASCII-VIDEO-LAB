import unittest
from argparse import Namespace
from pathlib import Path
import numpy as np
import torch
from ascii_video.glyphs import GlyphAtlas
from ascii_video.processing import GpuAsciiProcessor


def options(**kw):
    d=dict(device='cpu',width=160,height=96,cell_width=5,cell_height=8,
           analysis_scale=2,font=None,palette='blue',bg_rgb=(5,8,11),
           spatial='guided',guided_epsilon=18.,background='cover',
           temporal_alpha=.25,motion_threshold=8.,contrast=1.15,
           edge_threshold=80.,edge_source='current',mode='hybrid',
           stability=5.,inset_width=0,inset_border=6,profile_stages=False)
    d.update(kw)
    return Namespace(**d)


class TestProcessor(unittest.TestCase):
    def test_dimensions(self):
        p=GpuAsciiProcessor(options(),160,90)
        image=np.zeros((90,160,3),dtype=np.uint8)
        out,extras=p.process(image,debug=True)
        self.assertEqual(out.shape,(96,160,3))
        self.assertIn('edges',extras)
        self.assertEqual(p.count,1)

    def test_analysis_never_enlarged_past_source(self):
        p=GpuAsciiProcessor(options(cell_width=2,cell_height=2,analysis_scale=8),160,90)
        self.assertEqual((p.aw,p.ah),(160,90))

    def test_tiny_glyph_atlas(self):
        atlas=GlyphAtlas(2,2,torch.device('cpu'))
        self.assertEqual(atlas.atlas.shape[1:],(2,2))
        self.assertEqual(int(atlas.atlas[atlas.index['@']].sum()),4*255)
        out=atlas.render(torch.tensor([[atlas.index['@'],atlas.index[' ']]]))
        self.assertEqual(out.shape,(2,4))
        self.assertEqual(int(out[:,:2].sum()),4*255)
        self.assertEqual(int(out[:,2:].sum()),0)

    def test_ghosted_edge_does_not_persist(self):
        p=GpuAsciiProcessor(options(stability=50.),160,90)
        rows,cols=p.rows,p.cols
        p.prev_indices=torch.full((rows,cols),p.atlas.index['|'],dtype=torch.long)
        p.prev_anchor=torch.ones((rows,cols))*128
        p.prev_edge=torch.ones((rows,cols),dtype=torch.bool)
        # Smooth current image does not contain an edge; stabiliser must drop old '|'.
        clean=torch.ones((1,1,p.ah,p.aw))*128
        idx,mag=p._select(clean,clean)
        self.assertFalse(bool((idx==p.atlas.index['|']).any()))

    def test_inset(self):
        p=GpuAsciiProcessor(options(inset_width=60),160,90)
        image=np.zeros((90,160,3),dtype=np.uint8)
        image[:,:,2]=200
        out,_=p.process(image)
        self.assertEqual(out.shape,(96,160,3))
        self.assertGreater(int(out[48,80,2]),170)


if __name__=='__main__':
    unittest.main()
