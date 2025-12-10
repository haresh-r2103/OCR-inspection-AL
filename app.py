# app.py  (Compact, clean, user-friendly dashboard)
import os
import json
import random
import shutil
import time
from glob import glob
from datetime import datetime
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

from ultralytics import YOLO
from checker import scan_json_char_frequency, compare_to_allowed

# ============= PATHS (everything under ultralytics-main) =============
BASE = r"D:\ultralytics-main"
INPUT_DIR = os.path.join(BASE, "input")   # common folder for images + JSON
INPUT_JSON = INPUT_DIR
INPUT_IMG = INPUT_DIR
OUTPUT = os.path.join(BASE, "output")

# New split base: everything under "images"
SPLIT_BASE = os.path.join(BASE, "images")

# Deployed model location
LAST_PT = os.path.join(BASE, "last.pt")

# Training run name and weights directory (REAL YOLO TRAINING)
TRAIN_RUN_NAME = "Office_Senitize154"
TRAIN_WEIGHTS_DIR = os.path.join(BASE, "runs", "segment", TRAIN_RUN_NAME, "weights")
TRAIN_LAST_PT = os.path.join(TRAIN_WEIGHTS_DIR, "last.pt")

CLASSES = list("0123456789ABCDEFGHJKL MNP RSTVWXY".replace(" ", ""))

LOGO_URL = "https://pnghdpro.com/wp-content/themes/pnghdpro/download/social-media-and-brands/ashok-leyland-logo-hd.png"

# ============= PAGE SETTINGS =============
st.set_page_config(
    page_title="OCR INSPECTION TRAINING",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============= SESSION STATE (for validation gate) =============
if "validated_once" not in st.session_state:
    st.session_state["validated_once"] = False
if "last_char_counts" not in st.session_state:
    st.session_state["last_char_counts"] = None
if "last_unexpected" not in st.session_state:
    st.session_state["last_unexpected"] = None

# ============= ULTRA-COMPACT CSS + BRIGHT BUTTONS =============
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    
    button[kind="primary"], button[kind="secondary"] {
        padding: 0.5rem 1.2rem !important;
        height: 2.5rem !important;
        font-size: 0.95rem !important;
        min-height: 2.5rem !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #f97316 0%, #ea580c 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 0.6rem 1.4rem !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        cursor: pointer !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        min-height: 2.5rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ea580c 0%, #c2410c 100%) !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        transform: translateY(-1px);
    }

    .section-header {
        background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
        padding: 0.25rem 0.6rem !important;
        border-radius: 4px;
        color: white;
        margin-bottom: 0.4rem !important;
        font-size: 0.85rem !important;
        font-weight: 600;
    }
    
    .info-box {
        background-color: #f3f4ff;
        border: 1px solid #e0e7ff;
        padding: 0.6rem 0.8rem !important;
        border-radius: 6px;
        margin: 0.2rem 0 0.4rem 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.5;
        font-weight: 500;
    }

    .user-guide {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 3px solid #f59e0b;
        padding: 0.7rem 1rem !important;
        border-radius: 6px;
        margin: 0.4rem 0 0.5rem 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.5;
        color: #92400e;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .user-guide strong {
        color: #78350f;
        font-weight: 600;
    }

    .stMetric {
        padding: 0.1rem 0.2rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
    }
    
    .stColumn {
        padding: 0.2rem !important;
    }
    
    h1 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.3rem !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e293b !important;
    }

    .summary-container {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        padding: 1.2rem 1.5rem !important;
        border-radius: 8px;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #cbd5e1;
    }
    .summary-title {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1e293b !important;
        margin-bottom: 0.8rem !important;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #cbd5e1;
    }
    .summary-metric {
        background: white;
        padding: 0.8rem !important;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-left: 3px solid #4f46e5;
        transition: transform 0.2s;
    }
    .summary-metric:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============= HEADER =============
header_col1, header_col2 = st.columns([1.5, 4.5])
with header_col1:
    st.image(LOGO_URL, width=180)
with header_col2:
    st.markdown(
        "<h1 style='margin:0;padding-top:0.5rem;'>OCR INSPECTION TRAINING</h1>",
        unsafe_allow_html=True,
    )
    st.markdown("""
<p style='text-align:left;color:#444;font-size:1rem;margin-top:0;'>
    📁 <strong>User Guide:</strong>
    Store all files in 
    <code style="
        background:#fbbf24;
        padding:5px;
        border-radius:4px;
        color:#78350f;
        font-weight:600;
        display:inline;
    ">D:\\ultralytics-main\\input</code>
</p>
""", unsafe_allow_html=True)

    st.markdown(f"""
    <p style='margin:0.2rem 0 0 0;font-size:1rem;color:#475569;'>
        <b>Input Directory:</b> <a href="file:///D:/ultralytics-main" target="_blank" style="text-decoration:none;color:#2563eb;font-weight:500;">D:\\ultralytics-main</a>
    </p>
    <p style='margin:0.3rem 0 0 0;font-size:1rem;color:#64748b;font-style:italic;'>
        💡 First run <b>Validate</b> to check label characters and frequency before Convert / Split / Train.
    </p>
    """, unsafe_allow_html=True)

# ============= UTILITIES =============
def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p

def convert_points(points, w, h):
    yolo_points = []
    for x, y in points:
        yolo_points.append(float(x) / w)
        yolo_points.append(float(y) / h)
    return yolo_points

# ============= TABS (Validate first) =============
tab_validate, tab_convert, tab_split, tab_train = st.tabs(
    ["Validate", "Convert", "Split", "Train & Deploy"]
)

# ---------- TAB 1: VALIDATE ----------
with tab_validate:
    st.markdown("<div class='section-header'>Validate JSON Labels & Character Frequency</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        1️⃣ Checks that all label characters belong to the allowed set:<br>
        <code>0123456789ABCDEFGHJKLMNPRSTVWXY</code><br><br>
        2️⃣ Shows frequency of each character. If any character has <b>&lt; 50</b> samples, it is highlighted in <span style='color:#dc2626;font-weight:600;'>red</span>.
    </div>
    """, unsafe_allow_html=True)

    validate_btn = st.button("Validate Now", key="validate_btn")

    if validate_btn:
        with st.spinner("Checking JSON files and character frequencies..."):
            char_count, examples = scan_json_char_frequency(INPUT_JSON)
            unexpected, allowed_counts = compare_to_allowed(char_count, CLASSES)

            st.session_state["validated_once"] = True
            st.session_state["last_char_counts"] = allowed_counts
            st.session_state["last_unexpected"] = unexpected

            if not unexpected:
                st.success("✓ All labels contain only allowed characters.")
            else:
                st.error(f"✗ Found {len(unexpected)} invalid character(s).")
                with st.expander("Invalid characters and examples", expanded=True):
                    for ch, cnt in unexpected.items():
                        st.markdown(f"- **'{ch}'**: {cnt} occurrence(s)")
                        if ch in examples and examples[ch]:
                            st.caption(f"Examples: {', '.join(examples[ch][:3])}")

            # Character frequency view
            if allowed_counts:
                st.markdown("#### Character Frequency Overview")
                low_chars = {k: v for k, v in allowed_counts.items() if v < 50}
                ok_chars = {k: v for k, v in allowed_counts.items() if v >= 50}

                if low_chars:
                    st.markdown(
                        "<p style='color:#dc2626;font-weight:600;font-size:0.85rem;'>⚠ Characters with frequency &lt; 50 – need more samples:</p>",
                        unsafe_allow_html=True,
                    )
                    for ch, cnt in sorted(low_chars.items(), key=lambda x: x[0]):
                        st.markdown(
                            f"<span style='color:#dc2626;font-size:0.8rem;'>• '{ch}': {cnt}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        "<p style='color:#16a34a;font-weight:600;font-size:0.85rem;'>✅ All characters have ≥ 50 samples.</p>",
                        unsafe_allow_html=True,
                    )

                with st.expander("Full character distribution", expanded=False):
                    st.json({k: v for k, v in allowed_counts.items()})

    elif st.session_state["validated_once"] and st.session_state["last_char_counts"] is not None:
        allowed_counts = st.session_state["last_char_counts"]
        unexpected = st.session_state["last_unexpected"] or {}
        if not unexpected:
            st.success("✓ Validation already run: all labels within allowed character set.")
        else:
            st.error(f"✗ Previous validation found {len(unexpected)} invalid characters. Re-run after fixes.")

        low_chars = {k: v for k, v in allowed_counts.items() if v < 50}
        if low_chars:
            st.markdown(
                "<p style='color:#dc2626;font-weight:600;font-size:0.85rem;'>⚠ Some characters still have &lt; 50 samples:</p>",
                unsafe_allow_html=True,
            )
            for ch, cnt in sorted(low_chars.items(), key=lambda x: x[0]):
                st.markdown(
                    f"<span style='color:#dc2626;font-size:0.8rem;'>• '{ch}': {cnt}</span>",
                    unsafe_allow_html=True,
                )

# ---------- Helper: gate message ----------
def require_validation():
    if not st.session_state.get("validated_once", False):
        st.warning("Please run **Validate** first (see 'Validate' tab) before using this feature.")
        return False
    return True

# ---------- TAB 2: CONVERT ----------
with tab_convert:
    st.markdown("<div class='section-header'>Convert JSON → TXT Format for Training</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Converts LabelMe JSON to YOLO polygon format.<br>
    Output: <code>output/*.txt</code> and <code>output/classes.txt</code>
    <br><br><b>Note:</b> Run <b>Validate</b> first to ensure clean labels and good character frequency.
    </div>
    """, unsafe_allow_html=True)

    if not require_validation():
        st.stop()

    try:
        image_files = [
            f for f in os.listdir(INPUT_IMG)
            if os.path.splitext(f)[1].lower() in [".bmp", ".jpg", ".jpeg", ".png"]
        ]
    except:
        image_files = []
    try:
        json_files = glob(os.path.join(INPUT_JSON, "*.json"))
    except:
        json_files = []

    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    json_basenames = {os.path.splitext(os.path.basename(f))[0] for f in json_files}

    missing_json = sorted(list(image_basenames - json_basenames))
    missing_img = sorted(list(json_basenames - image_basenames))

    convert_btn = None
    if missing_json or missing_img:
        st.error("⚠ Filenames don't match. Fix before conversion:")
        if missing_json:
            with st.expander(f"Images without JSON ({len(missing_json)})", expanded=False):
                for name in missing_json[:10]:
                    st.text(f"• {name}")
                if len(missing_json) > 10:
                    st.caption(f"... and {len(missing_json) - 10} more")
        if missing_img:
            with st.expander(f"JSONs without image ({len(missing_img)})", expanded=False):
                for name in missing_img[:10]:
                    st.text(f"• {name}")
                if len(missing_img) > 10:
                    st.caption(f"... and {len(missing_img) - 10} more")
    else:
        st.success("✓ All filenames matched (1:1)")
        convert_btn = st.button("Convert", key="convert_btn")

    if convert_btn:
            with st.spinner("Converting..."):
                ensure(OUTPUT)
                files = json_files

                if not files:
                    st.warning("No JSON files found in input folder!")
                else:
                    bar = st.progress(0)
                    logs = []
                    classes_local = CLASSES.copy()

                    for i, jf in enumerate(files):
                        try:
                            with open(jf, "r", encoding="utf-8") as f:
                                data = json.load(f)
                        except Exception:
                            logs.append(f"❌ Bad JSON: {os.path.basename(jf)}")
                            continue

                        img_name = data.get("imagePath") or os.path.basename(jf).replace(".json", ".jpg")
                        img_path = os.path.join(INPUT_IMG, img_name)

                        if not os.path.exists(img_path):
                            logs.append(f"⚠️ Missing Image: {img_name}")
                            continue

                        with Image.open(img_path) as im:
                            w, h = im.size

                        yolo_lines = []

                        for shape in data.get("shapes", []):
                            lbl = shape.get("label", "")
                            if lbl not in classes_local:
                                classes_local.append(lbl)
                            cid = classes_local.index(lbl)

                            if shape.get("shape_type") != "polygon":
                                continue

                            pts = convert_points(shape.get("points", []), w, h)
                            yolo_lines.append(f"{cid} " + " ".join(f"{p:.6f}" for p in pts))

                        out = os.path.join(OUTPUT, os.path.basename(jf).replace(".json", ".txt"))
                        with open(out, "w", encoding="utf-8") as f:
                            f.write("\n".join(yolo_lines))

                        logs.append(f"✅ {os.path.basename(jf)} → {os.path.basename(out)}")
                        bar.progress(int((i + 1) / len(files) * 100))

                    with open(os.path.join(OUTPUT, "classes.txt"), "w", encoding="utf-8") as f:
                        f.write("\n".join(classes_local))

                    st.success(f"✓ Converted {len(files)} files")
                    if len(logs) > 0:
                        with st.expander("View logs", expanded=False):
                            st.code("\n".join(logs[-20:]))

# ---------- TAB 3: SPLIT ----------
with tab_split:
    st.markdown("<div class='section-header'>Split Dataset (80% Train / 20% Val)</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='info-box'>
    Splits dataset into train/val sets.<br>
    <b>Output structure:</b><br>
    <code>{SPLIT_BASE}\\images\\train</code><br>
    <code>{SPLIT_BASE}\\images\\val</code><br>
    <code>{SPLIT_BASE}\\Labels\\train</code><br>
    <code>{SPLIT_BASE}\\Labels\\val</code><br><br>
    <b>Note:</b> Run <b>Validate</b> and <b>Convert</b> before splitting.
    </div>
    """, unsafe_allow_html=True)

    if not require_validation():
        st.stop()

    split_btn = st.button("Split", key="split_btn")

    if split_btn:
        with st.spinner("Splitting dataset..."):
            img_root = ensure(os.path.join(SPLIT_BASE, "images"))
            lbl_root = ensure(os.path.join(SPLIT_BASE, "Labels"))

            train_img_dir = ensure(os.path.join(img_root, "train"))
            val_img_dir   = ensure(os.path.join(img_root, "val"))
            train_lbl_dir = ensure(os.path.join(lbl_root, "train"))
            val_lbl_dir   = ensure(os.path.join(lbl_root, "val"))

            try:
                all_imgs = [
                    f for f in os.listdir(INPUT_IMG)
                    if os.path.splitext(f)[1].lower() in [".bmp", ".jpg", ".jpeg", ".png"]
                ]
            except:
                all_imgs = []
                st.error("Cannot access input folder!")

            if not all_imgs:
                st.warning("No images found in input folder!")
            else:
                random.shuffle(all_imgs)
                split_point = int(len(all_imgs) * 0.8)
                train_files = all_imgs[:split_point]
                val_files = all_imgs[split_point:]

                def move_set(files, img_dst, lbl_dst):
                    for f in files:
                        shutil.copy2(os.path.join(INPUT_IMG, f), os.path.join(img_dst, f))
                        name_txt = os.path.splitext(f)[0] + ".txt"
                        lbl_src = os.path.join(OUTPUT, name_txt)
                        lbl_tgt = os.path.join(lbl_dst, name_txt)

                        if os.path.exists(lbl_src):
                            shutil.copy2(lbl_src, lbl_tgt)
                        else:
                            open(lbl_tgt, "w").close()

                move_set(train_files, train_img_dir, train_lbl_dir)
                move_set(val_files,   val_img_dir,   val_lbl_dir)

                st.success(f"✓ Split complete: {len(train_files)} train, {len(val_files)} val")

# ---------- TAB 4: TRAIN & DEPLOY ----------
with tab_train:
    st.markdown("<div class='section-header'>Train & Deploy Model</div>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='info-box'>
    <b>Train (YOLO Segmentation):</b> Trains model using <code>yolov8s-seg.pt</code> and <code>custom.yaml</code> → <code>{TRAIN_LAST_PT}</code><br>
    <b>Deploy:</b> Copies trained model → <code>{LAST_PT}</code><br><br>
    <b>Note:</b> Run <b>Validate</b> and <b>Convert</b> before training.
    </div>
    """, unsafe_allow_html=True)

    if not require_validation():
        st.stop()

    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        train_btn = st.button("Train", key="train_btn")
    with bcol2:
        deploy_btn = st.button("Deploy", key="deploy_btn")

    # REAL TRAINING (YOLO SEGMENTATION)
    if train_btn:
        with st.spinner("Training YOLO segmentation model..."):
            try:
                # Ensure runs directory structure exists (YOLO will create subfolders)
                ensure(os.path.join(BASE, "runs", "segment"))

                model = YOLO(os.path.join(BASE, "yolov8s-seg.pt"))
                results = model.train(
                    data=os.path.join(BASE, "ultralytics", "cfg", "datasets", "custom.yaml"),
                    imgsz=640,
                    epochs=100,
                    batch=4,
                    name=TRAIN_RUN_NAME,
                    task="segment",
                    device="cuda",   # change to "cpu" if no GPU
                    amp=False,
                    mixup=0.0,
                    copy_paste=0.0,
                    flipud=0.0,
                    fliplr=0.0,
                    perspective=0.0,
                    erasing=0.0,
                )

                # After training, YOLO saves weights under runs/segment/TRAIN_RUN_NAME/weights
                st.success(f"✓ Training complete. Weights saved under: {TRAIN_WEIGHTS_DIR}")
                if hasattr(results, "save_dir"):
                    st.caption(f"YOLO run directory: {results.save_dir}")
            except Exception as e:
                st.error(f"✗ Training failed: {e}")

    # DEPLOY: copy last.pt from YOLO run folder to BASE\last.pt
    if deploy_btn:
        if not os.path.exists(TRAIN_LAST_PT):
            st.error(f"✗ Model not found: {TRAIN_LAST_PT}")
        else:
            try:
                shutil.copy2(TRAIN_LAST_PT, LAST_PT)
                st.success(f"✓ Deployed to: {LAST_PT}")
                st.rerun()
            except Exception as e:
                st.error(f"✗ Deploy failed: {e}")

    # ============= SUMMARY SECTION =============
st.markdown("---")
st.markdown("""
<div class='summary-container'>
        <div class='summary-title'>📊 Application Summary</div>
</div>
""", unsafe_allow_html=True)

try:
    json_count = len(glob(os.path.join(INPUT_JSON, "*.json")))
except:
    json_count = 0
    
try:
    img_count = len([
        f for f in os.listdir(INPUT_IMG)
        if os.path.splitext(f)[1].lower() in [".bmp", ".jpg", ".jpeg", ".png"]
    ])
except:
    img_count = 0
    
try:
    output_count = len(glob(os.path.join(OUTPUT, "*.txt")))
    if os.path.exists(os.path.join(OUTPUT, "classes.txt")):
        output_count = max(0, output_count - 1)
except:
    output_count = 0
    
model_exists = os.path.exists(LAST_PT)

try:
    train_img_dir = os.path.join(SPLIT_BASE, "images", "train")
    val_img_dir   = os.path.join(SPLIT_BASE, "images", "val")
    train_count = len([f for f in os.listdir(train_img_dir) if os.path.splitext(f)[1].lower() in [".bmp", ".jpg", ".jpeg", ".png"]]) if os.path.exists(train_img_dir) else 0
    val_count   = len([f for f in os.listdir(val_img_dir)   if os.path.splitext(f)[1].lower() in [".bmp", ".jpg", ".jpeg", ".png"]]) if os.path.exists(val_img_dir)   else 0
except:
    train_count = 0
    val_count = 0

mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.markdown(f"""
    <div class='summary-metric'>
            <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>📄 JSON Files</div>
            <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{json_count}</div>
    </div>
        """, unsafe_allow_html=True)
    
with mc2:
        st.markdown(f"""
    <div class='summary-metric'>
            <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>🖼️ Images</div>
            <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{img_count}</div>
    </div>
        """, unsafe_allow_html=True)
    
with mc3:
    st.markdown(f"""
    <div class='summary-metric'>
        <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>📝 YOLO Labels</div>
        <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{output_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
with mc4:
    model_status = "✓ Ready" if model_exists else "✗ Not Deployed"
    model_color = "#16a34a" if model_exists else "#dc2626"
    st.markdown(f"""
    <div class='summary-metric'>
        <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>🤖 Model Status</div>
        <div style='font-size:1.5rem;font-weight:700;color:{model_color};'>{model_status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if train_count > 0 or val_count > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        mc5, mc6, mc7, mc8 = st.columns(4)
        with mc5:
            st.markdown(f"""
            <div class='summary-metric'>
                <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>🚂 Train Images</div>
                <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{train_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with mc6:
            st.markdown(f"""
            <div class='summary-metric'>
                <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>✅ Val Images</div>
                <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{val_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with mc7:
            total_split = train_count + val_count
            split_percent = f"{(train_count/total_split*100):.0f}% / {(val_count/total_split*100):.0f}%" if total_split > 0 else "0% / 0%"
            st.markdown(f"""
    <div class='summary-metric'>
                <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>📊 Split Ratio</div>
                <div style='font-size:1.2rem;font-weight:700;color:#1e293b;'>{split_percent}</div>
    </div>
            """, unsafe_allow_html=True)
        
        with mc8:
            total_split = train_count + val_count
            st.markdown(f"""
            <div class='summary-metric'>
                <div style='font-size:0.75rem;color:#64748b;margin-bottom:0.3rem;'>📦 Total Split</div>
                <div style='font-size:1.5rem;font-weight:700;color:#1e293b;'>{total_split}</div>
            </div>
            """, unsafe_allow_html=True)


# ============= FOOTER =============
st.markdown("""
<div style='text-align:center;color:#999;padding:0.3rem;font-size:0.65rem;margin-top:0.5rem;'>
    © Ashok Leyland Ltd. | OCR Inspection Dashboard
</div>
""", unsafe_allow_html=True)
