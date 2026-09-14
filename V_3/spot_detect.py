"""Detect any spot on facial skin -- acne, pigmentation, or anything else
that isn't smooth regular skin -- as a class-agnostic local color/texture
anomaly. No trained classifier: a spot is just "skin that looks different
from its own smoothed-out local neighborhood." Classification into
acne/pigmentation comes later, on top of these boxes.

Restricted to the CelebAMask-HQ "skin" class only (from face_segment's
parsing pass), so eyebrows, eyes, nose, lips etc. don't get flagged as
false "spots" just for having different color/texture than cheek skin.
"""
import cv2
import numpy as np

SKIN_CLASS = 1
# nose(10) and ears(7,8) are real skin -- the parser just labels them as
# their own classes instead of lumping them into "skin". They must be
# *searched*, not excluded (nose especially: it's one of the most acne-prone
# zones on the face -- excluding it was a bug, not a safety margin).
SKIN_SEARCH_CLASSES = {1, 7, 8, 10}
# eyebrows, eyes, glasses, earring, mouth, lips -- these are genuinely not
# skin (earring(9) is jewelry, not ear skin), so they're dilated and
# subtracted from the search area so eyelid/eyebrow creases, lip edges,
# earring edges etc. never get evaluated as skin in the first place.
NON_SKIN_FEATURE_CLASSES = {2, 3, 4, 5, 6, 9, 11, 12, 13}
NOSE_CLASS = 10


def skin_only_mask(class_mask, img_bgr=None):
    """A downstream classifier will see every box this produces and can reject
    the ones that aren't really acne/pigmentation -- a false positive is
    recoverable. A spot that never gets boxed here is gone for good. But that
    only argues for loosening filters that judge *candidate skin regions* --
    it is not a reason to shrink the buffer around eyes/eyebrows/mouth: those
    pixels can never be a skin spot no matter how it's tuned, and a thin
    buffer just reopens eyelid/eyebrow-crease false positives (confirmed by
    testing: shrinking this buffer for "recall" produced boxes around the
    entire eye, which helps no one). So that buffer stays at its proven-safe
    size; the recall-vs-precision knobs below live in find_spots() instead,
    where they act on genuine skin candidates only."""
    skin = np.isin(class_mask, list(SKIN_SEARCH_CLASSES)).astype(np.uint8) * 255
    features = np.isin(class_mask, list(NON_SKIN_FEATURE_CLASSES)).astype(np.uint8) * 255
    face_h, face_w = class_mask.shape
    k = max(5, (min(face_h, face_w) // 25) | 1)
    features_dilated = cv2.dilate(features, np.ones((k, k), np.uint8))

    # pull back from the raw crop border -- a hand/ear/hair intruding into the
    # crop margin will touch that border; this one *is* safe to loosen since
    # unlike the feature buffer above, real jaw/temple/hairline skin can
    # legitimately extend closer to the edge, so there's real recall on the table.
    border = max(1, int(0.02 * min(face_h, face_w)))
    border_mask = np.zeros((face_h, face_w), np.uint8)
    border_mask[border:-border, border:-border] = 255

    result = cv2.bitwise_and(cv2.bitwise_and(skin, cv2.bitwise_not(features_dilated)), border_mask)

    # the nostril openings are actual cavities, not skin -- the parser has no
    # separate class for them so they get lumped into "nose", and being far
    # darker than surrounding nose skin they'd otherwise register as a huge
    # false "spot" every time. Cut the darkest fraction of the nose region
    # (relative to that nose's own brightness, not a fixed number, so it
    # still works across lighting conditions).
    if img_bgr is not None:
        nose_region = class_mask == NOSE_CLASS
        if nose_region.any():
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            dark_cutoff = np.percentile(gray[nose_region], 15)
            nostril = (nose_region & (gray < dark_cutoff)).astype(np.uint8) * 255
            nostril = cv2.dilate(nostril, np.ones((5, 5), np.uint8))
            result = cv2.bitwise_and(result, cv2.bitwise_not(nostril))

    return result


def _deviation_map(crop_bgr, skin_mask):
    """Deviation from a *masked* local average, so the blur baseline is built
    only from nearby skin pixels -- otherwise a plain blur leaks in eyebrow/
    eye/hair/background color at every mask edge and falsely flags the whole
    boundary as "anomalous"."""
    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    face_h, face_w = skin_mask.shape
    k = max(9, (min(face_h, face_w) // 8) | 1)  # odd, scales with face size

    mask01 = (skin_mask > 0).astype(np.float32)
    mask_sum = cv2.GaussianBlur(mask01, (k, k), 0)
    lab_masked = lab * mask01[:, :, None]
    baseline = cv2.GaussianBlur(lab_masked, (k, k), 0) / (mask_sum[:, :, None] + 1e-6)

    deviation = np.linalg.norm(lab - baseline, axis=2)
    deviation[skin_mask == 0] = 0
    return deviation


def find_spots(crop_bgr, skin_mask, min_area_frac=0.00008, max_area_frac=0.06, min_fill=0.12,
               percentile=72.0, max_aspect=5.0):
    """Returns a list of (x0, y0, x1, y1) boxes, in crop_bgr's coordinate frame.

    Tuned for recall, not precision: classification happens later, on top of
    these boxes, so a false positive here just gets rejected downstream as
    "not acne, not pigmentation" -- cheap. A spot that never gets boxed here
    never reaches that stage at all -- it's just gone. Every filter below is
    kept as loose as it can be while still rejecting things that categorically
    cannot be a spot (a stray hair strand, a sliver of background); anything
    that's merely *unusual-shaped-for-a-typical-spot* is let through, because
    "unusual" real spots (streaky pigmentation, irregular acne clusters) are
    exactly the ones a tight filter would silently discard.

    The cutoff is the Nth percentile of this face's own interior-skin
    deviation values, not a fixed number and not Otsu. Otsu calibrates to
    the *strongest* split in the image, so one dark mole makes it set the
    bar too high for many fainter, more numerous marks elsewhere -- a face
    covered in small low-contrast spots gets systematically under-detected.
    A fixed absolute number has the opposite problem: it doesn't adapt to
    lighting/exposure at all. Percentile rank sidesteps both: it always
    keeps the top fraction of *that* face's most locally-deviant skin,
    whether those deviant pixels form one obvious cluster or are spread
    thinly across many small marks."""
    face_h, face_w = skin_mask.shape

    # evaluate only *interior* skin, well away from any class boundary (eyes,
    # brows, hairline, jaw) -- right at those edges the masked-blur baseline
    # has too few real skin samples nearby to be reliable, and reliably
    # produces false "deviation" all along the boundary. Unlike the filters
    # below, shrinking this doesn't buy real recall -- it just reopens the
    # eyelid/eyebrow-crease false-positive bug (confirmed by testing), so it
    # stays at its proven-safe size.
    erode_px = max(3, min(face_h, face_w) // 40)
    interior_mask = cv2.erode(skin_mask, np.ones((erode_px * 2 + 1,) * 2, np.uint8))

    deviation = _deviation_map(crop_bgr, skin_mask)
    interior_vals = deviation[interior_mask > 0]
    if interior_vals.size == 0:
        return []
    thresh_val = max(np.percentile(interior_vals, percentile), 5.0)
    anomaly = ((deviation > thresh_val) & (interior_mask > 0)).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_OPEN, kernel)
    anomaly = cv2.morphologyEx(anomaly, cv2.MORPH_CLOSE, kernel)

    face_area = (skin_mask > 0).sum()
    min_area = max(4, int(min_area_frac * face_area))
    max_area = int(max_area_frac * face_area)

    contours, _ = cv2.findContours(anomaly, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # reject scattered/elongated detections (e.g. stray hair strands)
        # that morphological closing merged into one sparse bounding box --
        # a real spot is a roughly compact blob that fills most of its box.
        if area / (w * h) < min_fill:
            continue
        # reject thin, elongated shapes -- skin creases and fold shadows
        # (nasolabial fold, under-eye line, nostril rim) trace long curves,
        # while a real spot (acne, pigmentation) is roughly round/oval.
        if max(w, h) / max(1, min(w, h)) > max_aspect:
            continue
        boxes.append((x, y, x + w, y + h))
    return boxes


def draw_spots(crop_bgr, boxes, color=(0, 255, 0), thickness=2):
    out = crop_bgr.copy()
    for x0, y0, x1, y1 in boxes:
        cv2.rectangle(out, (x0, y0), (x1, y1), color, thickness)
    return out


if __name__ == "__main__":
    import sys
    from face_segment import analyze_face

    img = cv2.imread(sys.argv[1])
    analysis = analyze_face(img)
    if analysis is None:
        print("no face detected")
        sys.exit()

    boxes = find_spots(analysis["crop"], skin_only_mask(analysis["class_mask"], analysis["crop"]))
    print(f"found {len(boxes)} spots")
    out = draw_spots(analysis["crop"], boxes)
    out_path = sys.argv[2] if len(sys.argv) > 2 else "spots.jpg"
    cv2.imwrite(out_path, out)
    print("saved", out_path)
