# Geolocalization-OSINT (México)

Aplicación de **Streamlit** que estima la **ubicación probable de una fotografía** combinando señales visuales (modelo local), **texto detectado por OCR** (Tesseract) y capas geográficas en un **mapa interactivo** (Folium).  
Diseñada para **prácticas académicas de ciberinteligencia**: inspección de evidencias, generación de hipótesis y documentación reproducible.

## 1) Objetivo y alcance
- **Objetivo:** Proveer un flujo reproducible para **geolocalizar indicios visuales** en imágenes con foco en ciudades de **México**, integrando visión por computador, OCR y apoyo cartográfico.  
- **Alcance:** Uso **docente** y **experimental**. No sustituye peritajes; la salida es **hipótesis** a validar con fuentes adicionales.

## 2) Arquitectura y flujo
1. **Entrada:** imagen local (JPG/PNG/AVIF, etc.).  
2. **Procesamiento:**  
   - **Visión:** extracción de señales visuales para rankear lugares probables.  
   - **OCR (opcional):** extracción de texto (señalética, placas, anuncios) con `pytesseract`; normalización y búsqueda de **topónimos**.  
   - **Fusión heurística:** ponderación de coincidencias por estado/ciudad; *fallback* al centroide de México si la evidencia es débil.  
3. **Salida:**  
   - **Mapa Folium** con marcadores/áreas sugeridas.  
   - **Resumen textual** de pistas (OCR, coincidencias) y **coordenadas** aproximadas.

> **Modelo:** el código referencia `model/modelo.pth` mediante ruta relativa.

## 3) Tecnologías (enlaces oficiales)
- Streamlit: https://streamlit.io/
- streamlit-folium: https://github.com/randyzwitch/streamlit-folium
- Folium: https://python-visualization.github.io/folium/
- PyTorch: https://pytorch.org/
- Transformers (Hugging Face): https://huggingface.co/docs/transformers/index
- OpenCV (opencv-python-headless): https://opencv.org/
- Tesseract OCR: https://tesseract-ocr.github.io/
- Pillow: https://python-pillow.org/
- tqdm: https://tqdm.github.io/

## 4) Requisitos del sistema
- **SO**: Windows 10/11 (probado), compatible con macOS/Linux con ajustes menores.  
- **Python**: 3.10+ (recomendado).  
- **GPU**: no requerida (usa CPU por defecto; si hay CUDA, PyTorch la detecta).  
- **Tesseract (opcional para OCR)**: instalar binario del sistema y dejarlo en `PATH` (o ajustar ruta en el código).

## 5) Instalación (Windows, entorno aislado)
```bat
cd Geolocalization-OSINT\osint-geolocalizador
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### (Opcional) Instalar Tesseract
- Instala Tesseract (Windows).  
- Verifica:
```bat
tesseract --version
```
- Si no está en `PATH`, indica la ruta absoluta del ejecutable en el código.

## 6) Ejecución
Desde `osint-geolocalizador` con el entorno activado:
```bat
streamlit run Geolocalizador.py
```
La app abrirá en el navegador (por defecto `http://localhost:8501/`).

## 7) Uso (paso a paso)  
1. **(Opcional) Activar OCR** si la imagen contiene texto útil.  
2. **Ajustar parámetros** (idioma OCR, umbrales, etc., si están expuestos).  
3. **Ejecutar análisis** → se renderiza un **mapa Folium** con hipótesis de ubicación.  
4. **Interpretar resultados**: revisar el panel de evidencias (pistas OCR, coincidencias) y el mapa.

## 8) Resultados esperados / salidas
- **Mapa interactivo** con marcadores/regiones sugeridas.  
- **Coordenadas** (lat, lon) aproximadas.  
- **Pistas**: texto OCR normalizado y coincidencias con entidades geográficas.

## 9) Limitaciones y consideraciones éticas
- **Hipótesis, no veredicto**: la localización es **aproximada** y depende de la calidad de la imagen y del modelo.  
- **Cobertura geográfica**: optimizado para **México**; fuera de este ámbito puede degradarse.  
- **OCR** es sensible a borrosidad, ángulos y baja resolución.  
- **Ética y legalidad**: usar sólo con **material propio o autorizado** y con **finalidad académica**. Evita doxxing, acoso o violaciones de privacidad. Respeta leyes y políticas institucionales.

## 10) Estructura del repositorio
```
Geolocalization-OSINT/
├─ README.md
├─ model/
│  └─ modelo.pth                # modelo local (referenciado por el código)
├─ osint-geolocalizador/
│  ├─ Geolocalizador.py         # APP principal (Streamlit)
│  ├─ build_model.py            # script auxiliar (entrenamiento/ensamblado)
│  ├─ model/
│  │  └─ modelo.pth             # (posible duplicado; alinear ruta)
│  ├─ data/                     # (si aplica)
│  └─ requirements.txt
└─ photos/                      # imágenes de ejemplo (si se incluyen)
```
> **Nota rutas de modelo:** Mantén un **único** `modelo.pth` (en la raíz) y alinea la ruta en `Geolocalizador.py` para evitar confusiones.

---

## ANEXO TÉCNICO

### A1. Versiones y entorno
- Python 3.10+  
- Dependencias principales:
  - `streamlit`, `streamlit-folium`, `folium`, `Pillow`
  - `torch`, `transformers`, `tqdm`
  - `pytesseract`, `opencv-python-headless`

Instalación en limpio (Windows):
```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r osint-geolocalizador\requirements.txt
```

### A2. Ejecución local
```bat
cd osint-geolocalizador
.\.venv\Scripts\activate
streamlit run Geolocalizador.py
```

### A3. Variables y rutas importantes
- **Modelo:** `model/modelo.pth` (ruta relativa al **repo**). Si mueves el archivo, actualiza la ruta en `Geolocalizador.py`.  
- **Tesseract:** si no está en `PATH`, especifica la ruta del ejecutable (Windows suele ser `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`).

### A4. Estructura mínima para reproducir
```
Geolocalization-OSINT/
├─ model/modelo.pth
├─ osint-geolocalizador/Geolocalizador.py
└─ osint-geolocalizador/requirements.txt
```

### A5. Prueba rápida (smoke test)
1. Activar entorno y levantar app:
   ```bat
   streamlit run osint-geolocalizador\Geolocalizador.py
   ```
2. Cargar imagen de prueba (idealmente con texto/rasgos urbanos).  
3. Verificar render del mapa y que se muestren coordenadas y pistas OCR (si activaste OCR).

### A6. Errores comunes
- **`ModuleNotFoundError`**: instalar dependencias → `pip install -r requirements.txt`.  
- **CUDA no detectada**: es normal sin GPU; corre en CPU.  
- **`tesseract: not found`**: instalar Tesseract y añadir a `PATH`, o fijar la ruta absoluta.  
- **Mapa centrado en México con marcadores**: evidencia débil → usar imagen con más rasgos o activar OCR.

### Licencia
Uso **académico**.
