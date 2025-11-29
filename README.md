# 🛰️ GEOLOCALIZADOR OSINT - MÉXICO

Sistema completo de geolocalización de imágenes usando CLIP + OCR, 100% open source.

## 📋 Descripción

Herramienta OSINT para geolocalizar fotografías en México mediante:
- **CLIP (Vision Transformer)**: Modelo de IA para análisis visual
- **OCR (Tesseract)**: Extracción de texto en imágenes
- **Fine-tuning**: Mejora con datos anotados manualmente
- **Fuentes abiertas**: Wikimedia Commons, Wikipedia, Pexels

## 🎯 Modelo Pre-entrenado (Google Drive)

Si deseas usar el modelo ya entrenado sin realizar fine-tuning, descárgalo aquí:

**📦 [Descargar Modelo Fine-tuned](https://drive.google.com/drive/folders/1SMQZTZ1U_prWongTUwaCTURtpvYMaG8x?usp=sharing)**

Incluye:
- `modelo.pth` - Embeddings de 68 ciudades mexicanas
- `modelo_finetuned.pth` - Modelo CLIP entrenado con 100+ anotaciones
- `checkpoints/` - Checkpoints de entrenamiento por época

**Instrucciones:**
1. Descarga los archivos del Drive
2. Colócalos en la carpeta `model/` de este proyecto
3. Ejecuta `streamlit run Geolocalizador.py`

## 🚀 Inicio Rápido

### 1. Instalación

```powershell
# Clonar repositorio
git clone https://github.com/EGarpxMaster/Geolocalization-OSINT.git
cd Geolocalization-OSINT

# Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Uso Básico (Sin Fine-tuning)

```powershell
# Generar modelo base (si no existe model/modelo.pth)
python build_model.py

# Ejecutar interfaz OSINT
streamlit run Geolocalizador.py
```

Abre http://localhost:8501 y sube una imagen para geolocalizarla.

### 3. Workflow Completo (Con Fine-tuning)

#### **Paso 1: Minería de Imágenes**

```powershell
# Minar todas las ciudades (68 ciudades × 20 imágenes = 1,360)
python mining_pipeline.py --mode all --images 20

# Ver progreso
python mining_pipeline.py --check-progress

# Minar un estado específico
python mining_pipeline.py --mode state --state "Jalisco" --images 10
```

**Tiempo estimado**: 2 horas para dataset completo

#### **Paso 2: Anotación Manual**

```powershell
# Abrir herramienta de anotación
python training_pipeline.py --annotate
```

- Abre http://localhost:8501
- Categoriza imágenes (calidad, elementos, confianza)
- **Mínimo recomendado**: 100 imágenes anotadas
- Anotaciones guardadas en `data/mining/annotations.json`

#### **Paso 3: Fine-tuning**

```powershell
# Entrenar modelo (5 épocas, batch size 8)
python training_pipeline.py --train --epochs 5 --batch-size 8
```

**Parámetros opcionales**:
- `--min-quality 2`: Calidad mínima de imágenes (1-5)
- `--min-confidence 50`: Confianza mínima del anotador (0-100)
- `--learning-rate 1e-5`: Tasa de aprendizaje

**Tiempo estimado**: 15-30 minutos (CPU), 5-10 minutos (GPU)

#### **Paso 4: Regenerar Embeddings**

```powershell
# Generar embeddings con modelo mejorado
python training_pipeline.py --build-model
```

Esto actualiza `model/modelo.pth` con el modelo fine-tuned.

#### **Paso 5: Probar Mejoras**

```powershell
# Ejecutar interfaz con modelo mejorado
streamlit run Geolocalizador.py
```

**Mejora esperada**: 1-2% → 15-40% de confianza

## 📁 Estructura del Proyecto

```
Geolocalization-OSINT/
├── 📄 Archivos principales (ESENCIALES)
│   ├── Geolocalizador.py           # Interfaz OSINT principal
│   ├── mining_pipeline.py          # Minería de imágenes
│   ├── training_pipeline.py        # Anotación + Fine-tuning
│   ├── build_model.py              # Generador de embeddings base
│   ├── requirements.txt            # Dependencias Python
│   └── README.md                   # Esta documentación
│
├── 📊 Datos
│   ├── data/cities_mx.csv          # 68 ciudades de México
│   └── data/mining/                # Datos de minería
│       ├── images/                 # Imágenes descargadas
│       ├── metadata.json           # Metadata de imágenes
│       └── annotations.json        # Anotaciones manuales (Supabase)
│
├── 🤖 Modelos (Descargar desde Google Drive)
│   ├── model/modelo.pth            # Embeddings de ciudades
│   ├── model/modelo_finetuned.pth  # Modelo CLIP fine-tuned
│   └── model/checkpoints/          # Checkpoints de entrenamiento
│
├── 🔧 Scripts opcionales (Supabase)
│   ├── supabase_client.py          # Cliente de Supabase
│   ├── upload_annotations_to_supabase.py
│   ├── download_annotations_from_supabase.py
│   ├── fix_annotations_image_id.py
│   └── clean_orphan_annotations.py
│
└── 📸 Extras
    └── photos/                     # Fotos de prueba
```

### Archivos esenciales (mínimo para funcionar):
- `Geolocalizador.py` - Interfaz principal
- `build_model.py` - Generar modelo base
- `requirements.txt` - Instalar dependencias
- `data/cities_mx.csv` - Lista de ciudades
- `model/modelo.pth` - [Descargar del Drive](https://drive.google.com/drive/folders/1SMQZTZ1U_prWongTUwaCTURtpvYMaG8x?usp=sharing)

### Archivos opcionales (para fine-tuning):
- `mining_pipeline.py` - Solo si quieres minar más imágenes
- `training_pipeline.py` - Solo si quieres entrenar
- Scripts de Supabase - Solo si usas base de datos cloud

## 🔧 Configuración Avanzada

### Minería Personalizada

```powershell
# Minar ciudad específica
python mining_pipeline.py --mode city --city "Guadalajara" --images 30

# Ver estadísticas detalladas
python mining_pipeline.py --check-progress
```

**Nombres de archivos optimizados:**
El sistema genera nombres únicos automáticamente:
```
{fuente}_{ciudad}_{estado}_{índice}_{timestamp}.jpg
Ejemplo: wikimedia_Guadalajara_Jalisco_5_1732901234.jpg
```

Esto previene:
- ✅ Conflictos por duplicados
- ✅ Sobrescrituras accidentales
- ✅ Problemas con caracteres especiales (sanitizados automáticamente)

### Fine-tuning Personalizado

```powershell
# Entrenamiento intensivo (más épocas)
python training_pipeline.py --train --epochs 10 --batch-size 4 --learning-rate 5e-6

# Filtros más estrictos
python training_pipeline.py --train --min-quality 4 --min-confidence 80
```

### Optimización de Memoria

El sistema está optimizado para usar mínima memoria:

- **Carga lazy**: Recursos se cargan solo cuando se necesitan
- **Cache de Streamlit**: Modelo se carga 1 sola vez
- **Liberación explícita**: GPU memory se libera después de cada inferencia
- **Modo eval**: Desactiva gradientes en inferencia (reduce memoria 50%)

**Memoria requerida**:
- Inferencia básica: ~2 GB RAM, ~1 GB VRAM (GPU)
- Fine-tuning: ~8 GB RAM, ~4 GB VRAM (recomendado)

## 🌐 Fuentes de Datos (100% Gratuitas)

### 1. Wikimedia Commons
- **API**: Ilimitada, sin autenticación
- **Calidad**: Alta, imágenes de Wikipedia
- **Cobertura**: Excelente para monumentos y lugares turísticos

### 2. Wikipedia
- **API**: MediaWiki API, gratuita
- **Calidad**: Variable, pero contextualmente relevante
- **Cobertura**: Buena para artículos de ciudades

### 3. Pexels
- **API**: Gratuita con registro (2 min)
- **Límite**: 200 requests/hora
- **Calidad**: Profesional, fotos stock

**Clave API Pexels**: Ya incluida en `mining_pipeline.py`

## 📊 Resultados Esperados

### Antes del Fine-tuning
```
Taxco de Alarcón, Guerrero     — 1.66%
Cuernavaca, Morelos           — 1.60%
San Miguel de Allende, Gto    — 1.59%
```

### Después del Fine-tuning (100+ anotaciones)
```
Taxco de Alarcón, Guerrero     — 24.3%
Cuernavaca, Morelos           — 18.7%
San Miguel de Allende, Gto    — 15.2%
```

**Mejora típica**: 10-20x en confianza

## 🐛 Troubleshooting

### Problema: "KeyError: 'city_embeds'"
**Solución**: Regenera el modelo
```powershell
python build_model.py
```

### Problema: OCR no funciona
**Solución**: Instala Tesseract
```powershell
# Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
# Instalar en: C:\Program Files\Tesseract-OCR
```

### Problema: "CUDA out of memory"
**Solución**: Reduce batch size o usa CPU
```powershell
python training_pipeline.py --train --batch-size 4
```

### Problema: Minería muy lenta
**Solución**: Usa menos imágenes o un estado específico
```powershell
python mining_pipeline.py --mode state --state "CDMX" --images 10
```

### Problema: Pocas imágenes descargadas
**Causas comunes**:
- Pexels rate limit (200/hora) → Espera 1 hora
- Ciudad muy específica → Prueba ciudad más grande
- Problemas de red → Verifica conexión

## 🎯 Tips para Mejores Resultados

### Anotación
1. **Calidad > Cantidad**: 100 buenas anotaciones > 500 malas
2. **Prioriza elementos únicos**: Monumentos, arquitectura característica
3. **Sé consistente**: Usa los mismos criterios siempre
4. **Verifica la ciudad**: Solo marca "Sí" si estás seguro

### Fine-tuning
1. **Empieza pequeño**: 5 épocas, luego aumenta si mejora
2. **Monitorea val_loss**: Si sube, hay overfitting
3. **Usa checkpoints**: Guarda cada época para comparar
4. **Dataset balanceado**: Similar cantidad de imágenes por estado

### Inferencia
1. **Ajusta temperatura**: Menor = más confianza, Mayor = más diversidad
2. **Backoff por estado**: Útil para ciudades desconocidas
3. **OCR boost**: Aumenta si hay letreros visibles
4. **Prueba múltiples fotos**: Combina resultados mentalmente

## 📚 Arquitectura Técnica

### Modelo Base
- **CLIP**: `openai/clip-vit-large-patch14`
- **Dimensión**: 768D embeddings
- **Normalización**: Cosine similarity
- **Temperatura**: Softmax scaling (0.1-2.0)

### Fine-tuning
- **Loss**: Contrastive loss bidireccional (imagen→texto, texto→imagen)
- **Optimizador**: AdamW
- **Learning rate**: 1e-5 (default)
- **Data augmentation**: Multi-prompt per city (12 prompts)

### OCR Boost
- **Engine**: Tesseract 5.x
- **Idiomas**: spa+eng
- **Preprocesamiento**: Bilateral filter + grayscale
- **Boost**: +15% ciudad, +5% estado (configurable)

### Backoff por Estado
```python
score_final = (1 - α) * score_ciudad + α * score_estado
```
donde α = 0.25 (configurable)

## 🔐 Privacidad y OSINT

Este proyecto es **100% open source** y **no requiere APIs pagas**:
- ✅ Sin tracking
- ✅ Sin telemetría
- ✅ Datos procesados localmente
- ✅ Fuentes públicas y abiertas
- ✅ Compatible con investigación OSINT ética

## 📄 Licencia

MIT License - Uso libre para fines educativos y de investigación OSINT.

## 🙏 Créditos

- **CLIP**: OpenAI
- **Tesseract**: Google
- **Streamlit**: Streamlit Inc.
- **Wikimedia Commons**: Wikimedia Foundation
- **Pexels**: Pexels.com

## 📞 Soporte

Si encuentras errores o tienes sugerencias:
1. Revisa la sección **Troubleshooting**
2. Verifica que usaste los comandos correctos
3. Abre un issue en GitHub con detalles completos

---

**Versión**: 2.0 (Unificada y Optimizada)
**Última actualización**: Noviembre 2025
