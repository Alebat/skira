import cv2
import numpy as np
import pandas as pd
import colorsys

from tqdm import tqdm


def hsv2rgb(h, s, v):
    return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))


def read_images(vc, rotate90=False):
    yes = True
    f = 0
    while yes:
        yes, img = vc.read()
        if yes:
            if rotate90:
                yield f, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            else:
                yield f, img
            f += 1


def save(name, images, fps):
    out = None
    for i in images:
        if out is None:
            h, w, c = i.shape
            out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out.write(i)
    out.release()


def multisave(name, extension, images, fps):
    out = {}
    for n, i in images:
        if n not in out:
            h, w, c = i.shape
            out[n] = cv2.VideoWriter(f'{name}-%03d.{extension}' % n, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        out[n].write(i)
    for o in out:
        out[o].release()


def draw_bb(tracks, images, rel=True):
    for f, i in images:
        if f in tracks.index:
            dets = tracks.loc[f]
            for _, row in (dets.iterrows() if len(dets.shape) == 2 else [(0, dets)]):
                n, x, y, xx, yy, _, _, _, _ = row
                if rel:
                    xx = x + xx
                    yy = y + yy
                cv2.rectangle(i, (int(x), int(y)), (int(xx), int(yy)), hsv2rgb(n * 49 % 15 / 15, 1, 1), 2)
        yield i


def process(tracks, images, rel=True, parallel_chunk=0, max_parallel=50):
    a = .9

    def low_pass(prev, value):
        return prev * a + value * (1 - a)

    box = {}
    for frame_num, frame in images:
        if frame_num in tracks.index:
            detections = tracks.loc[[frame_num]]
            for _, row in detections.iterrows():
                n, x, y, xx, yy, _, _, _, _ = row
                n = int(n)
                if True:  # n // max_parallel == parallel_chunk:
                    if rel:
                        xx += x
                        yy += y
                    cx = x / 2 + xx / 2
                    cy = y / 2 + yy / 2
                    edge = int(max(xx - x, yy - y, 24) * 1.5)

                    h, w, _ = frame.shape

                    if n in box:
                        pcx, pcy, pedge = box[n]

                        cx = low_pass(pcx, cx)
                        cy = low_pass(pcy, cy)
                        edge = low_pass(pedge, edge)

                    box[n] = cx, cy, edge

                    # double the larger edge of the rectangle
                    sqx1 = cx - edge
                    sqx2 = cx + edge
                    sqy1 = cy - edge
                    sqy2 = cy + edge

                    if sqx1 < 0 or sqx2 > w or sqy1 < 0 or sqy2 > h:
                        color = np.mean(frame, axis=(0, 1))
                        safe = cv2.copyMakeBorder(frame, int(edge), int(edge), int(edge), int(edge), cv2.BORDER_CONSTANT,
                                                  value=tuple(color))
                        sqx1 += edge
                        sqx2 += edge
                        sqy1 += edge
                        sqy2 += edge
                    else:
                        safe = frame

                    cropped = safe[int(sqy1):int(sqy2), int(sqx1):int(sqx2), :]
                    # print(n, sqx1, sqx2, sqy1, sqy2, safe.shape, cropped.shape)
                    try:
                        yield n, cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
                    except Exception as e:
                        print(e)


def highlight(input_video, input_detections, output_video, relative_bboxes=True, rotate90=False):
    tracks = pd.read_csv(input_detections,
                         sep=',',
                         header=None,
                         index_col=0,
                         names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    vc = cv2.VideoCapture(input_video)
    save(output_video,
         tqdm(draw_bb(tracks, read_images(vc, rotate90), rel=relative_bboxes), total=vc.get(7), unit='frame'),
         vc.get(5))
    vc.release()


def crop(input_video, input_detections, output_video, output_videos_extension, relative_bboxes=True, rotate90=False):
    tracks = pd.read_csv(input_detections,
                         sep=',',
                         header=None,
                         index_col=0,
                         names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x', 'y', 'z'])
    vc = cv2.VideoCapture(input_video)
    multisave(output_video,
              output_videos_extension,
              tqdm(process(tracks, read_images(vc, rotate90), rel=relative_bboxes), total=len(tracks), unit='frame'),
              vc.get(5))
    vc.release()
