"""
GEOLOCALIZADOR OSINT - INTERFAZ PRINCIPAL
==========================================
Aplicación Streamlit para geolocalización de imágenes en México usando CLIP + OCR.
Sistema de memoria optimizado con carga lazy de recursos.

Requiere:
  - Modelo: model/modelo.pth (generado con training_pipeline.py --build-model)
  - Tesseract OCR (opcional): C:\Program Files\Tesseract-OCR

Uso:
    streamlit run Geolocalizador.py
"""

import os
import re
import math
import shutil
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium


# ----- Configuración robusta de Tesseract (ajusta si instalaste en otra ruta) -----
# Si tu instalación está en otra ruta, cambia estas dos variables:
DEFAULT_TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_TESSDATA_DIR = r"C:\Program Files\Tesseract-OCR\tessdata"

# Intenta preparar entorno Tesseract
try:
    import pytesseract
    if Path(DEFAULT_TESSERACT_EXE).exists():
        pytesseract.pytesseract.tesseract_cmd = DEFAULT_TESSERACT_EXE
        os.environ["TESSDATA_PREFIX"] = DEFAULT_TESSDATA_DIR
except Exception:
    pass

# OpenCV y utilidades para OCR
import cv2

# ---------------------- Utilidades generales ----------------------
def normalize_text(txt: str) -> str:
    """Minúsculas y sin acentos para comparaciones robustas."""
    txt = txt.lower()
    txt = unicodedata.normalize("NFD", txt)
    txt = "".join(ch for ch in txt if unicodedata.category(ch) != "Mn")
    return txt

def km2_to_radius_km(area_km2: float) -> float:
    # r = sqrt(area/pi)
    return math.sqrt(max(area_km2, 1e-6) / math.pi)

def dynamic_radius_km(base_area_km2: float, prob: float) -> float:
    """
    Radio dinámico (km) basado en probabilidad p∈[0,1].
    Base: área 60 km² -> ~4.37 km. Ajuste: 0.6x con p=1, hasta 3.0x con p=0.
    """
    r_base = km2_to_radius_km(base_area_km2)  # ~4.37 km si base=60
    factor = 0.6 + 2.4 * (1.0 - float(prob))
    return r_base * factor

def softmax_temp(x, t=1.0):
    x = np.asarray(x, dtype=np.float64)
    x = x / max(t, 1e-6)
    x -= x.max()
    e = np.exp(x)
    return e / e.sum()

# ---------------------- Carga de índice y modelo (optimizado) ----------------------
@st.cache_resource
def load_city_index(rel_path: str):
    """Carga índice del modelo con manejo optimizado de memoria"""
    base_dir = Path(__file__).resolve().parent
    real_path = (base_dir / rel_path).resolve()
    if not real_path.exists():
        raise FileNotFoundError(f"No se encontró el índice: {real_path}")
    
    # Carga con map_location para CPU (optimiza memoria GPU)
    payload = torch.load(real_path, map_location="cpu", weights_only=False)
    
    # Esperado del builder v2:
    #  - model_name
    #  - cities: [{name,state,lat,lon,tags}, ...]
    #  - city_embeds: tensor [N, D]
    #  - states: [str]
    #  - state_embeds: dict state->tensor[D]
    
    # Liberar memoria inmediatamente después de cargar
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return payload

@st.cache_resource
def load_clip(model_name: str):
    """Carga modelo CLIP con optimización de memoria"""
    from transformers import CLIPProcessor, CLIPModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cargar modelo en modo eval (ahorra memoria)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()  # Desactiva dropout y batch norm
    
    # No guardar gradientes (reduce memoria 2x)
    for param in model.parameters():
        param.requires_grad = False
    
    processor = CLIPProcessor.from_pretrained(model_name)
    
    return model, processor, device

# ---------------------- OCR ----------------------
def detect_tesseract_available() -> bool:
    # válido si hay binario en PATH o en la ruta por defecto
    if shutil.which("tesseract"):
        return True
    return Path(DEFAULT_TESSERACT_EXE).exists()

def extract_text_ocr(pil_img: Image.Image, prefer_langs=("eng", "spa")) -> str:
    """OCR robusto con fallback. No rompe la app si falta Tesseract."""
    try:
        import pytesseract
    except Exception:
        return ""
    try:
        # Preprocesado suave
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)

        # Selección de idiomas disponibles
        langs_avail = []
        try:
            langs_avail = pytesseract.get_languages(config="")
        except Exception:
            pass
        chosen = []
        for lg in prefer_langs:
            if (not langs_avail) or (lg in langs_avail):
                chosen.append(lg)
        lang_arg = "+".join(chosen) if chosen else "eng"

        txt = pytesseract.image_to_string(gray, lang=lang_arg)
        return txt
    except pytesseract.TesseractNotFoundError:
        return ""
    except Exception:
        return ""

# ---------------------- UI ----------------------
st.set_page_config(page_title="OSINT MX — Geolocalizador por Imagen", layout="wide")
st.title("🛰️ OSINT MX — Geolocalización de Imágenes (CLIP + OCR)")
st.caption("Sube una imagen; se estiman las **top-K** ciudades de México, con radio dinámico según confianza.")

# Parámetros (sidebar)
with st.sidebar:
    st.markdown("### Parámetros de inferencia")
    base_area_km2 = st.number_input("Área base (km²)", min_value=10.0, max_value=300.0, value=60.0, step=5.0)
    topk = st.slider("Top-K ciudades", min_value=1, max_value=10, value=3)
    temperature = st.slider("Temperatura (softmax)", min_value=0.1, max_value=2.0, value=0.7, step=0.05)
    state_backoff = st.slider("Backoff por estado", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    st.divider()
    st.markdown("### OCR (boost por coincidencias)")
    tesseract_ok = detect_tesseract_available()
    do_ocr = st.checkbox("Usar OCR (experimental)", value=tesseract_ok, disabled=not tesseract_ok)
    city_name_boost = st.slider("Boost por ciudad en OCR", 0.0, 0.50, 0.15, 0.01)
    state_name_boost = st.slider("Boost por estado en OCR", 0.0, 0.50, 0.05, 0.01)
    st.caption(("✅ Tesseract detectado." if tesseract_ok else
               "⚠️ Tesseract no detectado: el OCR se deshabilitará."))

# Carga índice y modelo
try:
    IDX = load_city_index("model/modelo.pth")
except Exception as e:
    st.error(f"No pude cargar el índice del modelo (.pth): {e}")
    st.stop()

MODEL_NAME = IDX["model_name"]
CITIES = IDX["cities"]  # [{name,state,lat,lon,tags}, ...]
CITY_EMBEDS_T = IDX["city_embeds"]  # tensor [N, D]
STATE_NAMES = IDX["states"]
STATE_EMBEDS_T = IDX["state_embeds"]  # dict state -> tensor[D]

CITY_EMBEDS = CITY_EMBEDS_T.numpy()  # [N, D]
STATE_EMBEDS = {k: v.numpy() for k, v in STATE_EMBEDS_T.items()}

MODEL, PROCESSOR, DEVICE = load_clip(MODEL_NAME)

# Uploader
uploaded = st.file_uploader("📷 Sube una imagen (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Sube una imagen para comenzar.")
    st.stop()

# Mostrar imagen
pil_img = Image.open(uploaded).convert("RGB")
st.image(pil_img, caption="Imagen cargada", use_container_width=True)

# OCR (opcional)
ocr_text_raw = ""
ocr_text_norm = ""
if do_ocr:
    with st.spinner("Ejecutando OCR…"):
        ocr_text_raw = extract_text_ocr(pil_img)
        ocr_text_norm = normalize_text(ocr_text_raw)
    with st.expander("🔎 Texto detectado (OCR)"):
        st.code(ocr_text_raw or "(sin texto)")

# Embedding de imagen (optimizado para memoria)
with st.spinner("Calculando similitud (CLIP)…"):
    inputs = PROCESSOR(images=pil_img, return_tensors="pt").to(DEVICE)
    
    # Inferencia sin gradientes (ahorra memoria)
    with torch.no_grad():
        z_img = MODEL.get_image_features(**inputs)  # [1, D]
        z_img = z_img / z_img.norm(dim=-1, keepdim=True)
        img_feat = z_img.cpu().numpy()[0]  # [D]
    
    # Liberar memoria GPU inmediatamente
    del inputs, z_img
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Similitud con ciudades
sims = CITY_EMBEDS @ img_feat  # coseno ya normalizado si CITY_EMBEDS está normalizado

# Boost por OCR (ciudad/estado) — robusto a acentos/uppercase
if do_ocr and ocr_text_norm:
    bonus = np.zeros_like(sims)
    for i, c in enumerate(CITIES):
        city_name = normalize_text(c["name"])
        state_name = normalize_text(c["state"])

        # Word-boundaries simples (añade espacios al inicio/fin para reducir falsos positivos)
        if re.search(rf"\b{re.escape(city_name)}\b", ocr_text_norm):
            bonus[i] += city_name_boost
        if re.search(rf"\b{re.escape(state_name)}\b", ocr_text_norm):
            bonus[i] += state_name_boost
    sims = sims + bonus

# Backoff por estado (mezcla score ciudad con score de estado)
if state_backoff > 0:
    # 1) máximo score por estado entre sus ciudades
    state_scores_citymax = defaultdict(list)
    for i, c in enumerate(CITIES):
        state_scores_citymax[c["state"]].append(sims[i])
    state_scores_citymax = {s: (max(v) if v else -1.0) for s, v in state_scores_citymax.items()}

    # 2) similitud directa con embedding del estado
    state_scores_embed = {}
    for s, emb in STATE_EMBEDS.items():
        state_scores_embed[s] = float(emb @ img_feat)

    # 3) mezcla de ambos
    state_mix = {}
    for s in STATE_NAMES:
        v1 = state_scores_citymax.get(s, -1.0)
        v2 = state_scores_embed.get(s, -1.0)
        state_mix[s] = 0.5 * v1 + 0.5 * v2

    # aplica mezcla a cada ciudad, según su estado
    sims = np.array([(1.0 - state_backoff) * sims[i] + state_backoff * state_mix[CITIES[i]["state"]]
                     for i in range(len(CITIES))])

# Probabilidades con temperatura
probs = softmax_temp(sims, t=temperature)
order = np.argsort(-probs)

# Top-K resultados
topk_idx = order[:topk]
results = []
for idx in topk_idx:
    c = CITIES[idx]
    p = float(probs[idx])
    results.append({
        "city": c["name"],
        "state": c["state"],
        "lat": float(c["lat"]),
        "lon": float(c["lon"]),
        "prob": p
    })

# Mostrar ranking
st.subheader("🏁 Predicción (Top-K)")
for r in results:
    st.write(f"**{r['city']}**, {r['state']} — Prob: **{r['prob']*100:.2f}%**")

# Mapa Leaflet
st.subheader("🗺️ Mapa de probables ubicaciones")
center = (results[0]["lat"], results[0]["lon"]) if results else (23.6345, -102.5528)  # centro de MX fallback
m = folium.Map(location=center, zoom_start=6, tiles="OpenStreetMap")

for r in results:
    radius_km = dynamic_radius_km(base_area_km2, r["prob"])
    radius_m = radius_km * 1000.0

    folium.Circle(
        location=[r["lat"], r["lon"]],
        radius=radius_m,
        color="blue",
        fill=True,
        fill_opacity=0.12,
        popup=f"{r['city']} — {r['prob']*100:.1f}% — radio ~{radius_km:.1f} km"
    ).add_to(m)
    folium.Marker(
        location=[r["lat"], r["lon"]],
        tooltip=f"{r['city']} ({r['prob']*100:.1f}%)"
    ).add_to(m)

st_folium(m, height=560, use_container_width=True)

st.caption("Consejo: ajusta Temperatura y Backoff por estado en la barra lateral para afinar la precisión.")
