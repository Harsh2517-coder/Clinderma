"""Central tuning knobs for the V_3 candidate-generation stage.

This stage is a high-recall *proposal generator*: false positives are fine (the downstream acne/pigmentation model rejects
them), a missed lesion is lost for good. So every threshold below errs permissive. Functions read these as `cfg.NAME` at
call time, so eval_candidates.py can override any of them from the command line (NAME=value) for tuning sweeps."""

# --- 1. CLAHE (first stage; applied to the LAB L channel only so colour/redness isn't distorted)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
# CLAHE only evens out *local* contrast; a globally over/under-exposed photo still defeats the face parser. So, only when the
# median L is far from the target, a gamma is applied first (clipped so it can't over-correct). Normal photos are untouched.
EXPOSURE_NORMALIZE = True
EXPOSURE_TARGET_L = 120         # OpenCV LAB L scale, 0-255
EXPOSURE_TOLERANCE = 30
EXPOSURE_GAMMA_RANGE = (0.4, 2.5)

# --- 2. face detection / crop (unchanged from the original pipeline)
FACE_CONF = 0.3
FACE_CROP_MARGIN = 0.3

# --- 3. skin mask. CelebAMask-HQ parser classes searched as skin: skin 1, ears 7/8, nose 10 (real skin the parser gives their
# own labels). RESCUE_CLASSES: the parser labels some sun-lit / oily / acne-covered cheeks (profile and over-exposed shots) as
# 'hat' 18 -- measured: only 0.7-1.0% of all labelled lesions, but near-total loss on the affected images. Those pixels are
# searched only within RESCUE_DILATE_FRAC of the parsed face, so a curtain or wall labelled 'hat' far from the face stays out.
SEARCH_CLASSES = (1, 7, 8, 10)
RESCUE_CLASSES = (18,)
RESCUE_DILATE_FRAC = 0.05       # fraction of min(face crop height, width)
# Only the actual eye/brow/lip pixels are removed, plus a thin safety rim: periorbital and upper-lip
# pigmentation sits right next to those structures, so the old 1/25 buffer (which deleted it) is gone.
# All fractions are of min(face crop height, width).
EYE_BUFFER_FRAC = 1 / 20        # extra margin around eyes/eyebrows/glasses: lids and eye-socket shading otherwise light up as big blobs
FEATURE_BUFFER_FRAC = 1 / 50   # dilation of mouth/lip pixels (was 1/25 for everything)
BORDER_FRAC = 0.01              # inset from the crop border (was 0.02)
INTERIOR_ERODE_FRAC = 1 / 120   # inset from every mask edge before evaluating deviation (was 1/40)
NOSTRIL_DARK_PERCENTILE = 15    # darkest share of the nose region treated as nostril cavity
NOSTRIL_DILATE_PX = 5
MIN_SKIN_FRAC = 0.05            # if the parser finds less skin than this (close-ups, odd crops) search the whole crop instead of returning nothing

# --- 4. candidate detection. Thresholds are ABSOLUTE contrast floors (LAB units, after exposure normalisation + CLAHE), not a share of
# each face's pixels -- a percentile cutoff forces every face, however clear, to flag its top N% pixels (that gave ~900 boxes/face).
# Measured on the labelled Clinderma acne/pigmentation images (boxes/face | acne lesions found | pigmentation found, IoU>=0.2):
#   floors (dev/channel) 10/4  -> ~330 boxes | 0.83 | 0.69     14/6 (default) -> ~125 | 0.64 centre-0.88 | 0.51
#   18/8 -> ~90 | 0.55 centre-0.79 | 0.43      24/10 -> ~65 | 0.43 centre-0.65 | 0.41     (old pipeline: ~140 | 0.54 centre-0.72 | 0.19)
# Raise the floors for fewer boxes, lower them for more recall. Recall cannot be had for free: skin texture and lesions overlap in
# contrast, so an unsupervised stage trades boxes for recall along this curve. (multi-scale, several colour signals, union of everything)
WORK_MAX_SIDE = 1600            # candidate detection runs on a crop whose long side is capped at this (boxes mapped back)
BASELINE_SCALES = (1 / 24, 1 / 10, 1 / 5, 1 / 3)  # local-baseline blur size as a fraction of face size: small lesions -> patches
SIGNALS = ("deviation", "redness", "darkness", "brownness")  # deviation = LAB distance from local skin; redness = a* up; darkness = L down; brownness = b* up and L down
DEVIATION_PERCENTILE = 70.0       # secondary rank cutoff within the face; the absolute floors below are what actually bind
REDNESS_PERCENTILE = 70.0       # secondary rank cutoff within the face; the absolute floors below are what actually bind
DARKNESS_PERCENTILE = 70.0       # secondary rank cutoff within the face; the absolute floors below are what actually bind
BROWNNESS_PERCENTILE = 70.0       # secondary rank cutoff within the face; the absolute floors below are what actually bind
SMOOTH_FRAC_OF_SCALE = 0.08     # signal smoothing sigma as a fraction of the baseline size (blob scale, suppresses pixel noise)
COARSE_SCALE_FROM = 0.15        # baseline scales at/above this are 'patch' scales...
COARSE_FLOOR_MULT = 1.5         # ...and need this x the contrast floor (shading on forehead/cheeks is not a lesion)
MAX_BOX_SIDE_MULT = 1.0   # a box longer than this x the baseline size isn't a feature at that scale (larger scale handles it)
DEVIATION_FLOOR = 14.0   # absolute minimum cutoff so flat, clear skin doesn't turn noise into candidates
CHANNEL_FLOOR = 6.0   # same, for the single-channel signals

# --- 5. light candidate filtering (only categorically-not-a-spot shapes are rejected)
MIN_CANDIDATE_AREA_PX = 4
MIN_CANDIDATE_AREA_FRAC = 0.00004   # of valid face area (was 0.00008)
MAX_CANDIDATE_AREA_FRAC = 0.10      # of valid face area (was 0.06): large pigmentation patches / melasma are valid
MIN_BOX_SIDE_FRAC = 0.025   # candidates whose longest side is below this fraction of face size are dropped (specks)
MIN_FILL = 0.05                     # blob area / bbox area (was 0.12)
MAX_ASPECT = 8.0                    # (was 5.0)
OVERSIZE_RESPLIT_DEPTH = 3          # blobs above the max area are re-thresholded higher instead of dropped
OVERSIZE_RESPLIT_PERCENTILE = 70.0

# --- 6. boxes
MERGE_DISTANCE_FRAC = 0.004     # blobs closer than this (fraction of face size) are one lesion; farther apart stay separate
BOX_PADDING_RATIO = 0.10        # padding added around each tight box, never into non-skin
NMS_CONTAINMENT = 0.80   # >1 disables; e.g. 0.8 drops a weaker box lying 80% inside a stronger one
NMS_IOU_THRESHOLD = 0.30   # only near-duplicate boxes (e.g. same lesion found by two signals) are collapsed
