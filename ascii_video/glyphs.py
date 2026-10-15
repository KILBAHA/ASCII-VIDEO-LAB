"""Character atlas generated on CPU once, transferred to PyTorch device once."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

RAMP = " .:-=+*#%@"
EDGE_GLYPHS = "|-/\\+"
FONTS = [
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/LiberationMono-Regular.ttf",
]


def font_path_for(path):
    if path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Font does not exist: {path}")
        return path
    for path in FONTS:
        if Path(path).is_file():
            return path
    raise FileNotFoundError("Install ttf-dejavu, or use --font /path/to/mono.ttf")


def micro_mask(char, w, h):
    """For tiny cells, use micro-pixel masks rather than cropped unreadable font."""
    if char in RAMP:
        level = RAMP.index(char)
        # Fill progressively, spatially distributed, not one left-aligned rectangle.
        yy, xx = np.mgrid[0:h, 0:w]
        n = w * h
        # Small ordered-dither threshold map (Bayer 4x4), scaled to the cell.
        bayer = np.array([[0,8,2,10],[12,4,14,6],
                          [3,11,1,9],[15,7,13,5]],dtype=np.float32)
        phase=bayer[yy%4,xx%4]
        # Rank the values inside this cell so all n pixels eventually fill.
        order=np.argsort(phase.ravel(),kind='stable')
        rank=np.empty(n,dtype=np.int32)
        rank[order]=np.arange(n)
        return (rank.reshape(h,w) < round(level*n/(len(RAMP)-1))).astype(np.uint8)*255
    yy, xx = np.mgrid[0:h, 0:w]
    if char == '|':
        mask = np.abs(xx - (w-1)/2) <= max(0.5,w/4)
    elif char == '-':
        mask = np.abs(yy - (h-1)/2) <= max(0.5,h/4)
    elif char == '/':
        mask = np.abs(xx/(max(1,w-1)) + yy/max(1,h-1) - 1) <= 0.3
    elif char == '\\':
        mask = np.abs(xx/max(1,w-1) - yy/max(1,h-1)) <= 0.3
    else:
        mask = ((np.abs(xx-(w-1)/2) <= max(0.5,w/4)) |
                (np.abs(yy-(h-1)/2) <= max(0.5,h/4)))
    return (mask * 255).astype(np.uint8)


class GlyphAtlas:
    def __init__(self, cell_width, cell_height, device, font_path=None):
        self.cw, self.ch = cell_width, cell_height
        self.characters = list(dict.fromkeys(RAMP + EDGE_GLYPHS))
        self.index = {c: i for i, c in enumerate(self.characters)}
        masks = []
        tiny = cell_width < 5 or cell_height < 7
        font = None if tiny else ImageFont.truetype(font_path_for(font_path), max(7, cell_height-2))
        for c in self.characters:
            if tiny:
                masks.append(micro_mask(c, cell_width, cell_height))
            else:
                im = Image.new("L", (cell_width, cell_height), 0)
                draw = ImageDraw.Draw(im)
                box = draw.textbbox((0, 0), c, font=font)
                x = (cell_width - (box[2]-box[0])) // 2 - box[0]
                y = (cell_height - (box[3]-box[1])) // 2 - box[1]
                draw.text((x,y), c, font=font, fill=255)
                masks.append(np.asarray(im, dtype=np.uint8))
        self.atlas = torch.from_numpy(np.stack(masks)).to(device)
        self.ramp_indices = torch.tensor([self.index[c] for c in RAMP], dtype=torch.long, device=device)

    def render(self, indices):
        rows, cols = indices.shape
        return self.atlas[indices].permute(0,2,1,3).reshape(rows*self.ch, cols*self.cw)
