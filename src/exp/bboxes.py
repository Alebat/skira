import colorsys

import cv2
import numpy as np
import pandas as pd
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


def draw_bb(tracks, images, rel=True, frame_n=None):
    all = frame_n is not None
    for f, i in images:
        if all or f in tracks.index:
            dets = tracks.loc[f] if not all else tracks
            for n, row in (dets.iterrows() if len(dets.shape) == 2 else [(0, dets)]):
                if all:
                    if n[0] > frame_n:
                        continue
                    n = n[1]
                x, y, xx, yy, _, _, _, _ = row
                if rel:
                    xx = x + xx
                    yy = y + yy
                cv2.rectangle(i, (int(x), int(y)), (int(xx), int(yy)), hsv2rgb(n * 49 % 15 / 15, 1, 1), 1)
                if not all:
                    cv2.putText(i, str(int(n)), (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                                color=hsv2rgb(n * 49 % 15 / 15, 1, 0))
        yield i


def low_pass(prev, value, a=.95):
    return prev * a + value * (1 - a)


def apply_filter(boxes, n, rel, x, xx, y, yy):
    if rel:
        xx += x
        yy += y
    cx = x / 2 + xx / 2
    cy = y / 2 + yy / 2
    edge = int(max(xx - x, yy - y, 24) * 1.5)
    if n in boxes:
        pcx, pcy, pedge = boxes[n]

        cx = low_pass(pcx, cx)
        cy = low_pass(pcy, cy)
        edge = low_pass(pedge, edge)
    boxes[n] = cx, cy, edge
    # double the edge
    sqx1 = cx - edge
    sqx2 = cx + edge
    sqy1 = cy - edge
    sqy2 = cy + edge
    return edge, sqx1, sqx2, sqy1, sqy2


def process(tracks, images, rel=True):
    tracks.sort_index(inplace=True)
    inv_tracks = pd.DataFrame(index=tracks.index, columns=['data'])

    # filter tracks in opposite direction to compose a zero-phase filter
    boxes = {}
    for frame_num, track_id in sorted(tracks.index, key=lambda x: -x[0]):
        x, y, xx, yy, _, _, _, _ = tracks.loc[frame_num, track_id]
        inv_tracks.loc[frame_num, track_id] = [apply_filter(boxes, track_id, rel, x, xx, y, yy)]

    # filter tracks to compose a zero-phase filter
    boxes = {}
    images = iter(images)
    frame_n, frame = next(images)
    for frame_num, n in tracks.index:
        while frame_n < frame_num:
            # fast forward
            frame_n, frame = next(images)
        x, y, xx, yy, _, _, _, _ = tracks.loc[frame_num, n]

        edge, sqx1, sqx2, sqy1, sqy2 = apply_filter(boxes, n, rel, x, xx, y, yy)
        if_edge, if_sqx1, if_sqx2, if_sqy1, if_sqy2 = inv_tracks.loc[frame_num, n][0]

        sqx1 = (sqx1 + if_sqx1) / 2
        sqx2 = (sqx2 + if_sqx2) / 2
        sqy1 = (sqy1 + if_sqy1) / 2
        sqy2 = (sqy2 + if_sqy2) / 2
        edge = (edge + if_edge) / 2

        h, w, _ = frame.shape

        if sqx1 < 0 or sqx2 > w or sqy1 < 0 or sqy2 > h:
            color = np.mean(frame, axis=(0, 1))
            safe = cv2.copyMakeBorder(frame, int(edge), int(edge), int(edge), int(edge),
                                      cv2.BORDER_CONSTANT,
                                      value=tuple(color))
            sqx1 += edge
            sqx2 += edge
            sqy1 += edge
            sqy2 += edge
        else:
            safe = frame

        cropped = safe[int(sqy1):int(sqy2), int(sqx1):int(sqx2), :]
        try:
            yield n, cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            print(w, h, int(sqy1), int(sqy2), int(sqx1), int(sqx2), e)


def highlight(input_video, input_detections, output_video, relative_bboxes=True, rotate90=False):
    if isinstance(input_detections, str):
        input_detections = pd.read_csv(input_detections,
                                       sep=',',
                                       header=None,
                                       index_col=0,
                                       names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x',
                                              'y', 'z'])
    vc = cv2.VideoCapture(input_video)
    save(output_video,
         tqdm(draw_bb(input_detections, read_images(vc, rotate90), rel=relative_bboxes), total=vc.get(7), unit='frame'),
         vc.get(5))
    vc.release()


def nth_image(n, x):
    last = None
    for i, img in x:
        last = img
        if i == n:
            break
    return last


def highlight_img(input_video, input_detections, output, frame_n, relative_bboxes=True, rotate90=False):
    if isinstance(input_detections, str):
        input_detections = pd.read_csv(input_detections,
                                       sep=',',
                                       header=None,
                                       index_col=0,
                                       names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x',
                                              'y', 'z'])
    vc = cv2.VideoCapture(input_video)
    cv2.imwrite(output, list(draw_bb(input_detections, [(0, nth_image(frame_n, read_images(vc, rotate90)))], rel=relative_bboxes, frame_n=frame_n))[0])
    vc.release()


def crop(input_video, input_tracks, output_video, output_videos_extension, relative_bboxes=True, rotate90=False, parallel=75):
    if isinstance(input_tracks, str):
        input_tracks = pd.read_csv(input_tracks,
                                   sep=',',
                                   header=None,
                                   index_col=[0, 1],
                                   names=['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'x',
                                          'y', 'z'])
    if len(input_tracks) > 0:
        n_tracks = int(max(map(lambda x: x[1], input_tracks.index)))
        for i in range(0, n_tracks, parallel):
            print(f'{i} to {i + parallel - 1} / {n_tracks}')
            vc = cv2.VideoCapture(input_video)
            length = vc.get(7)
            chunk = input_tracks.query(f'{i} < id < {i + parallel}')
            multisave(output_video,
                      output_videos_extension,
                      process(chunk,
                              tqdm(read_images(vc, rotate90),
                                   total=length,
                                   unit='frames'),
                              rel=relative_bboxes),
                      vc.get(5))
            vc.release()
