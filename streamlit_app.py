import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from typing import Optional
from rembg import new_session, remove


st.set_page_config(page_title="Ugly Duckling - Streamlit", layout="wide")

st.markdown(
    """
    <style>
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            background: #d6f0d6 !important;
        }
        .stApp {
            background: linear-gradient(180deg, #e8f7e8 0%, #d6f0d6 100%) !important;
        }
        [data-testid="stSidebar"] {
            background: #bfe6bf !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_segmenter():
    return mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)


@st.cache_resource
def get_rembg_session(model_name: str = "u2net_human_seg"):
    return new_session(model_name)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_label] = 255
    return out


def build_body_mask(image_rgb: np.ndarray, threshold: float) -> np.ndarray:
    segmenter = get_segmenter()
    result = segmenter.process(image_rgb)
    if result.segmentation_mask is None:
        return np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    raw = (result.segmentation_mask >= threshold).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel5, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3, iterations=1)
    return keep_largest_component(cleaned)


def build_body_mask_rembg(image_rgb: np.ndarray, threshold: float) -> np.ndarray:
    try:
        session = get_rembg_session("u2net_human_seg")
    except Exception:
        session = get_rembg_session("u2net")

    raw_mask = remove(
        image_rgb,
        session=session,
        only_mask=True,
        post_process_mask=True,
    )

    if isinstance(raw_mask, Image.Image):
        raw_mask = np.array(raw_mask)

    raw_mask = np.asarray(raw_mask)
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    raw_mask = raw_mask.astype(np.float32)
    if raw_mask.max() <= 1.0:
        raw_mask *= 255.0

    binary = (raw_mask >= (threshold * 255.0)).astype(np.uint8) * 255
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel7, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel3, iterations=1)
    return keep_largest_component(cleaned)


def get_crop_bounds(width: int, height: int, top: int, bottom: int, left: int, right: int):
    x0 = int(width * (left / 100))
    x1 = int(width * (right / 100))
    y0 = int(height * (top / 100))
    y1 = int(height * (bottom / 100))
    return x0, y0, x1, y1


def apply_single_scale_retinex(gray: np.ndarray, sigma: float = 26.0) -> np.ndarray:
    gray_f = gray.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blur = np.maximum(blur, 1.0)
    retinex = np.log(gray_f) - np.log(blur)
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def build_black_tophat_retin_map(gray: np.ndarray, kernel_size: int, gain: float):
    # Kernel for local maxima search (dilation).
    k = int(max(3, kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

    # Approximate black top-hat: dilation-subtraction.
    dilated = cv2.dilate(gray, kernel, iterations=1)
    delta = cv2.subtract(dilated, gray)

    # Contrast scaling then clinical inversion.
    amplified = np.clip(delta.astype(np.float32) * float(gain), 0, 255).astype(np.uint8)
    retin = cv2.bitwise_not(amplified)
    return retin, amplified


def detect_lesions_circles(
    crop_rgb: np.ndarray,
    crop_fg_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
    upscale_factor: float,
    use_retinex: bool,
    tophat_kernel_size: int,
    tophat_gain: float,
    light_gray_threshold: int,
):
    crop_h, crop_w = crop_rgb.shape[:2]
    if crop_h == 0 or crop_w == 0:
        return [], np.zeros((crop_h, crop_w), dtype=np.uint8)

    scale = max(1.0, float(upscale_factor))
    if scale > 1.0:
        up_w = max(2, int(round(crop_w * scale)))
        up_h = max(2, int(round(crop_h * scale)))
        work_rgb = cv2.resize(crop_rgb, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        work_mask = cv2.resize(crop_fg_mask, (up_w, up_h), interpolation=cv2.INTER_NEAREST)
    else:
        work_rgb = crop_rgb
        work_mask = crop_fg_mask

    # Explicit grayscale mean, as requested.
    gray = ((work_rgb[:, :, 0].astype(np.float32) + work_rgb[:, :, 1].astype(np.float32) + work_rgb[:, :, 2].astype(np.float32)) / 3.0).astype(np.uint8)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if use_retinex:
        gray = apply_single_scale_retinex(gray, sigma=26.0)

    fg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    safe_fg = cv2.erode((work_mask > 0).astype(np.uint8) * 255, fg_kernel, iterations=1)
    if np.count_nonzero(safe_fg) < 200:
        safe_fg = (work_mask > 0).astype(np.uint8) * 255
    seg_binary = (work_mask > 0).astype(np.uint8) * 255
    seg_inner = cv2.erode(seg_binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    seg_boundary = cv2.subtract(seg_binary, seg_inner)
    seg_dist = cv2.distanceTransform((seg_binary > 0).astype(np.uint8), cv2.DIST_L2, 3)
    fg_area_up = float(np.count_nonzero(safe_fg))
    max_lesion_area_up = max(20.0, fg_area_up * 0.01)  # lesions must be clearly smaller than skin region

    # black_tophat Retin-style map by dilation subtraction.
    retin_map, delta_map = build_black_tophat_retin_map(gray, tophat_kernel_size, tophat_gain)

    # Rule requested: each dot within segmented skin darker than light gray is a lesion candidate.
    dark_pixels = (retin_map < int(light_gray_threshold)).astype(np.uint8) * 255

    # Remove very faint deltas to reduce pore/noise activation.
    fg_delta = delta_map[safe_fg > 0]
    if fg_delta.size > 0:
        delta_thresh = int(np.percentile(fg_delta, 45))
    else:
        delta_thresh = 6
    delta_thresh = max(4, min(80, delta_thresh))
    dark_response = (delta_map >= delta_thresh).astype(np.uint8) * 255

    lesion_map = cv2.bitwise_and(dark_pixels, dark_response)
    lesion_map = cv2.bitwise_and(lesion_map, safe_fg)
    lesion_map = cv2.morphologyEx(
        lesion_map,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    lesion_map = cv2.morphologyEx(
        lesion_map,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lesion_map, connectivity=8)

    lesions = []
    for label in range(1, num_labels):
        area_up = float(stats[label, cv2.CC_STAT_AREA])
        if area_up < 4:
            continue
        if area_up > max_lesion_area_up:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if x <= 1 or y <= 1 or (x + w) >= (work_rgb.shape[1] - 1) or (y + h) >= (work_rgb.shape[0] - 1):
            continue

        w_orig = w / scale
        h_orig = h / scale
        if w_orig < 2 or h_orig < 2:
            continue
        if (w_orig * h_orig) > 10000:
            continue

        aspect = max(w_orig, h_orig) / max(1.0, min(w_orig, h_orig))
        if aspect > 3.0:
            continue

        component = (labels[y:y + h, x:x + w] == label).astype(np.uint8) * 255
        boundary_overlap = np.any((component > 0) & (seg_boundary[y:y + h, x:x + w] > 0))
        if boundary_overlap:
            continue

        contours, hierarchy = cv2.findContours(component, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not contours or hierarchy is None:
            continue

        hierarchy = hierarchy[0]
        outer_indices = [i for i, hinfo in enumerate(hierarchy) if hinfo[3] == -1]
        if not outer_indices:
            continue
        outer_idx = max(outer_indices, key=lambda i: cv2.contourArea(contours[i]))
        contour = contours[outer_idx]
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue

        hole_area = 0.0
        for i, hinfo in enumerate(hierarchy):
            if hinfo[3] == outer_idx:
                hole_area += float(cv2.contourArea(contours[i]))

        has_hole = hole_area > 1.0
        ring_hole_ratio = hole_area / max(1.0, contour_area)
        component_fill_outer = area_up / max(1.0, contour_area)
        annular_candidate = (has_hole and ring_hole_ratio >= 0.03) or component_fill_outer < 0.35

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = (4.0 * np.pi * contour_area) / (perimeter * perimeter)
        if circularity < (0.16 if annular_candidate else 0.2):
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            continue
        solidity = contour_area / hull_area
        if solidity < (0.5 if annular_candidate else 0.68):
            continue

        shape = "circle"
        angle = 0.0
        axes_orig = (0, 0)
        shape_valid = False
        if len(contour) >= 5:
            (ecx, ecy), (major_axis, minor_axis), angle = cv2.fitEllipse(contour)
            ellipse_aspect = max(major_axis, minor_axis) / max(1e-6, min(major_axis, minor_axis))
            ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)
            ellipse_fill = contour_area / max(1.0, ellipse_area)

            pts = contour[:, 0, :].astype(np.float32)
            theta = np.deg2rad(angle)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            dx = pts[:, 0] - ecx
            dy = pts[:, 1] - ecy
            xr = dx * cos_t + dy * sin_t
            yr = -dx * sin_t + dy * cos_t
            norm = (xr / (major_axis / 2.0 + 1e-6)) ** 2 + (yr / (minor_axis / 2.0 + 1e-6)) ** 2
            ellipse_residual = float(np.mean(np.abs(np.sqrt(np.clip(norm, 1e-6, None)) - 1.0)))

            if annular_candidate:
                shape_valid = (
                    ellipse_aspect <= 2.8
                    and ellipse_fill >= 0.08
                    and ellipse_fill <= 1.25
                    and ellipse_residual <= 0.72
                    and solidity >= 0.5
                    and circularity >= 0.16
                    and component_fill_outer >= 0.05
                )
            else:
                shape_valid = (
                    ellipse_aspect <= 3.0
                    and ellipse_fill >= 0.22
                    and ellipse_fill <= 1.25
                    and ellipse_residual <= 0.7
                    and solidity >= 0.58
                    and circularity >= 0.18
                )

            if shape_valid:
                cx = x + ecx
                cy = y + ecy
                # Reject if fitted enclosure would cross segmentation boundary.
                center_x_i = int(np.clip(round(cx), 0, seg_dist.shape[1] - 1))
                center_y_i = int(np.clip(round(cy), 0, seg_dist.shape[0] - 1))
                required_clearance = (max(major_axis, minor_axis) / 2.0) * 1.25 + 2.0
                if seg_dist[center_y_i, center_x_i] < required_clearance:
                    shape_valid = False
                if not shape_valid:
                    pass
                else:
                    axis_major = max(3.0, (major_axis / 2.0) / scale * 1.25)
                    axis_minor = max(3.0, (minor_axis / 2.0) / scale * 1.25)
                    axes_orig = (int(round(axis_major)), int(round(axis_minor)))
                    shape = "ellipse"
                    radius_orig = max(axis_major, axis_minor)

        if not shape_valid:
            (cx_local, cy_local), radius_local = cv2.minEnclosingCircle(contour)
            circle_area = np.pi * (radius_local ** 2)
            circle_fill = area_up / max(1.0, circle_area)
            if annular_candidate:
                shape_valid = (
                    aspect <= 2.3
                    and circularity >= 0.34
                    and solidity >= 0.5
                    and circle_fill >= 0.04
                    and circle_fill <= 0.7
                )
            else:
                shape_valid = (
                    aspect <= 2.4
                    and circularity >= 0.30
                    and solidity >= 0.58
                    and circle_fill >= 0.16
                    and circle_fill <= 1.2
                )

            if not shape_valid:
                continue

            shape = "circle"
            axes_orig = (0, 0)
            angle = 0.0
            cx = x + cx_local
            cy = y + cy_local
            center_x_i = int(np.clip(round(cx), 0, seg_dist.shape[1] - 1))
            center_y_i = int(np.clip(round(cy), 0, seg_dist.shape[0] - 1))
            required_clearance = (radius_local * 1.35) + 2.0
            if seg_dist[center_y_i, center_x_i] < required_clearance:
                continue
            radius_orig = max((radius_local / scale) * 1.35, 3.5)

        roi = work_rgb[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        std_dev = float(np.std(roi_gray))
        mean_gray = float(np.mean(roi_gray))
        score = 1 if std_dev > 25 else 0
        if mean_gray < light_gray_threshold:
            score += 1

        lesions.append(
            {
                "center": (int((cx / scale) + offset_x), int((cy / scale) + offset_y)),
                "radius": int(radius_orig),
                "shape": shape,
                "axes": axes_orig,
                "angle": float(angle),
                "score": score,
                "std_dev": std_dev,
            }
        )

    lesions.sort(key=lambda x: x["score"], reverse=True)

    retin_display = np.full_like(retin_map, 220)
    retin_display[safe_fg > 0] = retin_map[safe_fg > 0]
    edges = cv2.Canny((safe_fg > 0).astype(np.uint8) * 255, 20, 60)
    retin_display[edges > 0] = 0

    if scale > 1.0:
        retin_display = cv2.resize(retin_display, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    return lesions, retin_display


def draw_overlay(
    image_rgb: np.ndarray,
    lesions,
    crop_bounds,
    body_mask: Optional[np.ndarray],
    show_mode: str,
):
    out = image_rgb.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = crop_bounds

    if body_mask is not None:
        background = body_mask == 0
        # Hard-remove background in output preview.
        out[background] = 0

        # Explicit demarcation line between skin (foreground) and background.
        boundary_mask = (body_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 2)

    # Dim outside manual crop.
    dim = out.copy()
    dim[:] = (dim * 0.45).astype(np.uint8)
    dim[y0:y1, x0:x1] = out[y0:y1, x0:x1]
    out = dim

    cv2.rectangle(out, (x0, y0), (x1, y1), (59, 130, 246), 2)

    lesion_layer_blue = np.zeros_like(out)
    lesion_layer_red = np.zeros_like(out)
    line_thickness = 1
    line_type = cv2.LINE_8
    label_items = []
    suspicious_view = show_mode == "Suspicious only"
    render_lesions = lesions[:3] if suspicious_view else lesions

    for idx, lesion in enumerate(render_lesions):
        if suspicious_view:
            line_color = (0, 0, 255)  # red
            target_layer = lesion_layer_red
        else:
            line_color = (110, 175, 250)  # blue
            target_layer = lesion_layer_blue
        cx, cy = lesion["center"]
        label_offset = lesion.get("radius", 8)
        if lesion.get("shape") == "ellipse":
            axes = lesion.get("axes", (0, 0))
            angle = lesion.get("angle", 0.0)
            cv2.ellipse(target_layer, (cx, cy), axes, angle, 0, 360, line_color, line_thickness, line_type)
            label_offset = max(axes[0], axes[1])
        else:
            radius = lesion["radius"]
            cv2.circle(target_layer, (cx, cy), radius, line_color, line_thickness, line_type)

        if suspicious_view:
            label_items.append((str(idx + 1), cx, cy, int(label_offset)))

    if body_mask is not None:
        lesion_layer_blue[body_mask == 0] = 0
        lesion_layer_red[body_mask == 0] = 0

    blue_pixels = np.any(lesion_layer_blue > 0, axis=2)
    if np.any(blue_pixels):
        alpha_blue = 0.78
        out_sel = out[blue_pixels].astype(np.float32)
        lesion_sel = lesion_layer_blue[blue_pixels].astype(np.float32)
        out[blue_pixels] = np.clip((1.0 - alpha_blue) * out_sel + alpha_blue * lesion_sel, 0, 255).astype(np.uint8)

    red_pixels = np.any(lesion_layer_red > 0, axis=2)
    if np.any(red_pixels):
        out[red_pixels] = lesion_layer_red[red_pixels]

    for text, cx, cy, offset in label_items:
        tx = max(0, cx + offset + 2)
        ty = max(10, cy - offset - 2)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_8)

    return out


st.title("Ugly Duckling - Lesion Detection (Streamlit)")
st.caption("Default: Rembg (U2-Net) body/background segmentation + circle-only lesion detection.")

with st.sidebar:
    st.header("Controls")
    seg_backend = st.selectbox(
        "Segmentation backend",
        ["Rembg (U2-Net Human)", "MediaPipe"],
        index=0,
    )
    seg_thresh = st.slider("Segmentation confidence", 0.1, 0.9, 0.50, 0.05)
    detect_upscale = st.slider("Detection upscale factor", 1.0, 4.0, 2.0, 0.25)
    tophat_kernel_size = st.select_slider("Black Top-Hat kernel", options=[2, 3, 4, 5, 6, 7, 8], value=5)
    tophat_gain = st.slider("Black Top-Hat gain", 1.0, 8.0, 5.0, 0.5)
    light_gray_threshold = st.slider("Light gray threshold", 120, 235, 190, 1)
    top = st.slider("Top boundary (%)", 0, 100, 0)
    bottom = st.slider("Bottom boundary (%)", 0, 100, 100)
    left = st.slider("Left boundary (%)", 0, 100, 0)
    right = st.slider("Right boundary (%)", 0, 100, 100)
    run = st.button("Detect Lesions")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(pil_img)
    h, w = image_rgb.shape[:2]
    x0, y0, x1, y1 = get_crop_bounds(w, h, top, bottom, left, right)

    if x1 <= x0 or y1 <= y0:
        st.error("Invalid boundaries. Ensure right > left and bottom > top.")
        st.stop()

    if run:
        body_mask = None
        if seg_backend == "Rembg (U2-Net Human)":
            with st.spinner("Running Rembg (U2-Net) body segmentation..."):
                body_mask = build_body_mask_rembg(image_rgb, seg_thresh)
        else:
            with st.spinner("Running MediaPipe body segmentation..."):
                body_mask = build_body_mask(image_rgb, seg_thresh)

        if body_mask is None or np.count_nonzero(body_mask) == 0:
            st.error("Background removal failed. Try lowering segmentation confidence.")
            st.stop()

        crop_mask = np.zeros((h, w), dtype=np.uint8)
        crop_mask[y0:y1, x0:x1] = 255

        # Preprocessing: always remove background before detection.
        effective_mask = cv2.bitwise_and(crop_mask, body_mask)

        masked = image_rgb.copy()
        masked[effective_mask == 0] = 0
        crop_rgb = masked[y0:y1, x0:x1]
        crop_fg_mask = effective_mask[y0:y1, x0:x1]

        lesions_all, _ = detect_lesions_circles(
            crop_rgb,
            crop_fg_mask,
            x0,
            y0,
            detect_upscale,
            False,
            tophat_kernel_size,
            tophat_gain,
            light_gray_threshold,
        )
        lesions_retinex, _ = detect_lesions_circles(
            crop_rgb,
            crop_fg_mask,
            x0,
            y0,
            detect_upscale,
            True,
            tophat_kernel_size,
            tophat_gain,
            light_gray_threshold,
        )
        overlay_all = draw_overlay(image_rgb, lesions_all, (x0, y0, x1, y1), body_mask, "All detected lesions")
        overlay_suspicious = draw_overlay(image_rgb, lesions_retinex, (x0, y0, x1, y1), body_mask, "Suspicious only")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("All Detected Lesions")
            st.image(overlay_all, width="stretch")

        with col2:
            st.subheader("Top 3 Suspicious Lesions")
            st.image(overlay_suspicious, width="stretch")
    else:
        st.image(image_rgb, caption="Uploaded image", width="stretch")
