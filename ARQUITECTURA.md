# 🏗️ Arquitectura del Sistema - Geolocalizador OSINT

## 📋 Tabla de Contenidos
1. [Visión General](#visión-general)
2. [Componentes Principales](#componentes-principales)
3. [Flujo de Datos](#flujo-de-datos)
4. [Base de Datos Supabase](#base-de-datos-supabase)
5. [Almacenamiento de Imágenes](#almacenamiento-de-imágenes)
6. [Pipeline de Entrenamiento](#pipeline-de-entrenamiento)
7. [Modelo de IA](#modelo-de-ia)

---

## 🎯 Visión General

Sistema de geolocalización basado en IA que identifica ciudades mexicanas a partir de fotografías. Utiliza CLIP (Contrastive Language-Image Pre-training) fine-tuneado con imágenes anotadas manualmente.

### Tecnologías Core
- **Backend**: Python 3.13
- **Base de Datos**: Supabase (PostgreSQL)
- **Almacenamiento**: Supabase Storage (bucket público)
- **Modelo IA**: OpenAI CLIP (ViT-Large-Patch14)
- **Framework UI**: Streamlit
- **Deep Learning**: PyTorch + Transformers

---

## 🧩 Componentes Principales

### 1. **Mining Pipeline** (`mining_pipeline.py`)
**Propósito**: Descarga automática de imágenes de ciudades mexicanas desde múltiples fuentes.

**Flujo de Trabajo**:
```
Ciudades CSV → Búsqueda en APIs → Descarga → Sanitización → Supabase Storage → Base de Datos
```

**Fuentes de Datos**:
- **Wikimedia Commons**: Fotos de monumentos y lugares emblemáticos
- **Wikipedia**: Imágenes de artículos de ciudades
- **Pexels**: Fotografías profesionales de stock

**Proceso de Minado**:
1. Lee `data/cities_mx.csv` (105 ciudades mexicanas)
2. Para cada ciudad:
   - Busca imágenes en las 3 fuentes
   - Descarga hasta N imágenes por fuente (configurable)
   - Sanitiza nombres de archivo (lowercase, sin acentos, underscores)
   - Detecta duplicados con hash MD5
   - Sube a Supabase Storage
   - Guarda metadata en base de datos

**Formato de Nombres de Archivo**:
```
{source}_{city}_{state}_{index}_{timestamp}.jpg

Ejemplo:
pexels_guadalajara_jalisco_0_1764047289.jpg
wikimedia_ciudad_de_mexico_cdmx_5_1764045693.jpg
```

**Metadata Generada**:
- Filename, ciudad, estado, coordenadas (lat/lon)
- Fuente, URL original, título, fotógrafo
- Dimensiones, tamaño, hash MD5
- URL de Supabase Storage

### 2. **Training Pipeline** (`training_pipeline.py`)
**Propósito**: Interfaz Streamlit para anotación manual y fine-tuning del modelo.

**Modos de Operación**:

#### 📝 **Anotación**
- Carga imágenes desde Supabase Storage (sin necesidad de archivos locales)
- Sistema de cola balanceada por estado (evita sesgo geográfico)
- Formulario de anotación con:
  - ✅ Verificación de ciudad correcta
  - ⭐ Calidad/relevancia (1-5 estrellas)
  - 🏷️ Etiquetas personalizadas (tags libres)
  - 👁️ Elementos detectados (landmarks, arquitectura, naturaleza, etc.)
  - 🎯 Confianza del anotador (0-100%)
  - 📝 Notas adicionales
- Opciones de eliminación de imágenes corruptas/irrelevantes
- Guarda en Supabase (`annotations` table) y CSV local (backup)

**Sistema de Balance**:
```python
# Algoritmo Round-Robin por estado
Jalisco: [img1, img2, img3, ...]
CDMX:    [img1, img2, img3, ...]
Yucatán: [img1, img2, ...]

Lista balanceada:
[Jalisco_img1, CDMX_img1, Yucatán_img1, 
 Jalisco_img2, CDMX_img2, Yucatán_img2, ...]
```

#### 🔬 **Fine-tuning**
- Entrena CLIP con imágenes anotadas (mínimo 50, recomendado 100+)
- Filtros configurables:
  - Calidad mínima (1-5)
  - Confianza mínima (0-100%)
- Arquitectura:
  - Loss: Contrastive Loss (temperatura 0.07)
  - Optimizador: AdamW
  - Split: 85% train / 15% validation
  - Early stopping en validation loss
- Guarda modelo fine-tuneado en `model/modelo_finetuned.pth`

#### 🏗️ **Regenerar Embeddings**
- Carga modelo fine-tuneado
- Genera embeddings para todas las ciudades
- Guarda modelo completo en `model/modelo.pth`:
  ```python
  {
    'city_embeds': Tensor[105, 768],  # Embeddings de ciudades
    'cities': List[Dict],              # Metadata de ciudades
    'model_name': str,                 # "openai/clip-vit-large-patch14"
    'states': List[str],               # Estados únicos
  }
  ```

#### 📊 **Estadísticas**
- Distribución de calidad de anotaciones
- Top ciudades anotadas
- Elementos más comunes detectados
- Etiquetas personalizadas populares

#### 🎯 **Evaluación**
- Prueba modelo con imágenes aleatorias anotadas
- Métricas:
  - Top-1 Accuracy (predicción exacta)
  - Top-3 Accuracy (top 3 predicciones)
  - Distancia promedio (error en km)
- Muestra predicciones con confianza y distancia real

### 3. **Geolocalizador** (`Geolocalizador.py`)
**Propósito**: Interfaz de usuario final para geolocalizar imágenes.

**Funcionalidades**:
- Sube una foto → Obtiene predicción de ciudad
- Muestra top 5 predicciones con porcentajes de confianza
- Visualización en mapa interactivo (Folium)
- Información detallada de la ciudad predicha

**Proceso Interno**:
1. Usuario sube imagen
2. CLIP procesa imagen → embedding de imagen
3. Compara con embeddings de 105 ciudades (cosine similarity)
4. Ordena por similitud
5. Retorna top 5 con porcentajes

### 4. **Supabase Client** (`supabase_client.py`)
**Propósito**: Wrapper de funciones para interactuar con Supabase.

**Funciones Principales**:
```python
# Lectura
get_all_images()              # Todas las imágenes
get_pending_images()          # Sin anotar
get_annotated_filenames()     # Nombres de anotadas
get_annotation_stats()        # Estadísticas agregadas

# Escritura
save_annotation(data)         # Guardar anotación
mark_deleted(filename, reason) # Marcar eliminada
```

### 5. **Upload to Storage** (`upload_to_supabase_storage.py`)
**Propósito**: Script de migración para subir imágenes locales a Supabase.

**Uso** (una sola vez):
```bash
python upload_to_supabase_storage.py
```
- Sube todas las imágenes de `data/mining/images/`
- Genera URLs públicas de Supabase Storage
- Actualiza `image_metadata.image_url` en BD

---

## 🔄 Flujo de Datos

### Flujo Completo del Sistema

```
1. MINADO DE DATOS
   ├─ cities_mx.csv (105 ciudades)
   ├─ APIs (Wikimedia, Wikipedia, Pexels)
   ├─ Descarga local temporal
   ├─ Supabase Storage upload
   └─ Base de datos (image_metadata)

2. ANOTACIÓN
   ├─ Streamlit carga desde Supabase
   ├─ Usuario anota imagen
   ├─ Guarda en annotations table
   └─ CSV backup local

3. ENTRENAMIENTO
   ├─ Lee annotations desde Supabase
   ├─ Fine-tune CLIP
   ├─ Genera modelo_finetuned.pth
   └─ Regenera embeddings → modelo.pth

4. PREDICCIÓN
   ├─ Usuario sube foto
   ├─ CLIP genera embedding
   ├─ Compara con ciudad_embeds
   └─ Retorna top 5 ciudades
```

### Flujo de Imagen Individual

```
API → Download → Sanitize → Storage → Database → Annotation → Training → Model
 │                   ↓                    ↓           ↓           ↓         ↓
Photo            filename.jpg         image_url   quality=4   Fine-tune  Predict
                                     (público)    tags=[...]   CLIP      City
```

---

## 🗄️ Base de Datos Supabase

### Esquema de Tablas

#### **image_metadata**
```sql
CREATE TABLE image_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL UNIQUE,
    image_url TEXT,                    -- URL pública de Supabase Storage
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    lat DECIMAL(10, 8),
    lon DECIMAL(11, 8),
    source TEXT,                       -- 'pexels', 'wikimedia', 'wikipedia'
    photo_id TEXT,
    url TEXT,                          -- URL original de la fuente
    title TEXT,
    photographer TEXT,
    width INTEGER,
    height INTEGER,
    size INTEGER,
    hash TEXT,                         -- MD5 para detectar duplicados
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### **annotations**
```sql
CREATE TABLE annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    image_id UUID REFERENCES image_metadata(id) UNIQUE,
    correct_city TEXT,                 -- Confirmación de ciudad
    quality INTEGER CHECK (quality BETWEEN 1 AND 5),
    confidence INTEGER CHECK (confidence BETWEEN 0 AND 100),
    
    -- Elementos detectados (9 banderas booleanas)
    has_landmarks BOOLEAN DEFAULT FALSE,
    has_architecture BOOLEAN DEFAULT FALSE,
    has_nature BOOLEAN DEFAULT FALSE,
    has_urban BOOLEAN DEFAULT FALSE,
    has_cultural BOOLEAN DEFAULT FALSE,
    has_religious BOOLEAN DEFAULT FALSE,
    has_modern BOOLEAN DEFAULT FALSE,
    has_historical BOOLEAN DEFAULT FALSE,
    has_coastal BOOLEAN DEFAULT FALSE,
    
    custom_tags TEXT[],               -- Array de PostgreSQL
    notes TEXT,
    annotated_by TEXT,                -- Nombre del anotador
    annotated_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### **deleted_images**
```sql
CREATE TABLE deleted_images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL UNIQUE,
    reason TEXT,                      -- Motivo de eliminación
    deleted_by TEXT,
    deleted_at TIMESTAMP DEFAULT NOW()
);
```

### Vistas Agregadas

#### **pending_images**
```sql
CREATE VIEW pending_images AS
SELECT im.*
FROM image_metadata im
LEFT JOIN annotations a ON im.id = a.image_id
WHERE a.id IS NULL;
```

#### **annotated_images**
```sql
CREATE VIEW annotated_images AS
SELECT 
    im.*,
    a.quality,
    a.confidence,
    a.custom_tags,
    a.annotated_by,
    a.annotated_at
FROM image_metadata im
INNER JOIN annotations a ON im.id = a.image_id;
```

#### **annotation_stats**
```sql
CREATE VIEW annotation_stats AS
SELECT 
    COUNT(DISTINCT im.id) as total_images,
    COUNT(DISTINCT a.id) as annotated_count,
    COUNT(DISTINCT im.state) as unique_states,
    COUNT(DISTINCT CASE WHEN a.id IS NULL THEN im.id END) as pending_count
FROM image_metadata im
LEFT JOIN annotations a ON im.id = a.image_id;
```

### Triggers

```sql
-- Auto-actualizar updated_at
CREATE TRIGGER update_image_metadata_updated_at
    BEFORE UPDATE ON image_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at
    BEFORE UPDATE ON annotations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

---

## 📦 Almacenamiento de Imágenes

### Supabase Storage

**Bucket**: `geolocalization-images` (público)

**Políticas RLS**:
```sql
-- Permitir lectura pública
CREATE POLICY "Lectura pública"
ON storage.objects FOR SELECT
TO public
USING (bucket_id = 'geolocalization-images');

-- Permitir subida pública
CREATE POLICY "Subida pública"
ON storage.objects FOR INSERT
TO public
WITH CHECK (bucket_id = 'geolocalization-images');
```

**Estructura de URLs**:
```
https://qlwzmjyztyfnhoxfjstd.supabase.co/storage/v1/object/public/geolocalization-images/{filename}

Ejemplo:
https://qlwzmjyztyfnhoxfjstd.supabase.co/storage/v1/object/public/geolocalization-images/pexels_guadalajara_jalisco_0_1764047289.jpg
```

**Ventajas**:
- ✅ Acceso público sin autenticación
- ✅ CDN integrado para carga rápida
- ✅ No requiere archivos locales en producción
- ✅ Deployment simplificado (Streamlit Cloud)
- ✅ URLs permanentes para referencias

---

## 🎓 Pipeline de Entrenamiento

### Arquitectura de Fine-tuning

```
Modelo Base: openai/clip-vit-large-patch14
├─ Vision Transformer (ViT-Large)
│  ├─ Patch size: 14x14
│  ├─ Hidden size: 1024
│  └─ Embedding dim: 768
└─ Text Transformer
   ├─ Vocabulary: 49408 tokens
   └─ Embedding dim: 768
```

### Dataset Personalizado

```python
class GeoDataset(Dataset):
    """
    Carga imágenes anotadas y genera pares imagen-texto
    
    Filtros:
    - min_quality: 1-5 (default: 2)
    - min_confidence: 0-100% (default: 50%)
    
    Texto generado:
    "A photo of {city}, {state}. Tags: {custom_tags}. 
     Elements: {detected_elements}"
    """
```

### Loss Function

```python
class ContrastiveLoss(nn.Module):
    """
    Pérdida contrastiva simétrica
    
    - Normaliza embeddings de imagen y texto
    - Calcula similitud coseno
    - Aplica temperatura (0.07)
    - Cross-entropy bidireccional
    """
```

### Proceso de Entrenamiento

1. **Preparación**:
   - Filtra anotaciones por calidad y confianza
   - Split 85% train / 15% validation
   - Batch size: 8 (configurable)

2. **Training Loop**:
   ```python
   for epoch in range(epochs):
       # Forward pass
       image_embeds = model.get_image_features(images)
       text_embeds = model.get_text_features(texts)
       
       # Contrastive loss
       loss = criterion(image_embeds, text_embeds)
       
       # Backward pass
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()
   ```

3. **Validation**:
   - Calcula loss en validation set
   - Early stopping si no mejora

4. **Guardado**:
   - Mejor modelo → `modelo_finetuned.pth`
   - Incluye solo `state_dict`

### Generación de Embeddings

```python
def build_model():
    """
    1. Carga modelo fine-tuneado
    2. Para cada ciudad:
       - Genera texto: "A photo of {city}, {state}"
       - Obtiene text embedding
       - Almacena en tensor
    3. Guarda modelo completo con embeddings
    """
```

---

## 🤖 Modelo de IA

### CLIP (Contrastive Language-Image Pre-training)

**Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

**Arquitectura**:
```
                    ┌─────────────┐
                    │   IMAGEN    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────┐
                    │  ViT Encoder    │
                    │  (Vision)       │
                    └──────┬──────────┘
                           │
                    ┌──────▼──────────┐
                    │ Image Embedding │ ──┐
                    │    (768 dim)    │   │
                    └─────────────────┘   │
                                          │ Cosine
                    ┌─────────────┐       │ Similarity
                    │    TEXTO    │       │
                    └──────┬──────┘       │
                           │              │
                    ┌──────▼──────────┐   │
                    │ Text Encoder    │   │
                    │ (Transformer)   │   │
                    └──────┬──────────┘   │
                           │              │
                    ┌──────▼──────────┐   │
                    │ Text Embedding  │ ──┘
                    │   (768 dim)     │
                    └─────────────────┘
```

### Proceso de Predicción

```python
def predict(image_path, model_path):
    # 1. Cargar modelo y embeddings
    model_data = torch.load(model_path)
    city_embeds = model_data['city_embeds']  # [105, 768]
    cities = model_data['cities']
    
    # 2. Procesar imagen
    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    
    # 3. Generar embedding de imagen
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_embed = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 4. Calcular similitudes
    similarities = (image_embed @ city_embeds.T).squeeze(0)
    
    # 5. Top 5 predicciones
    top5_indices = similarities.argsort(descending=True)[:5]
    top5_cities = [cities[i] for i in top5_indices]
    top5_scores = [similarities[i].item() for i in top5_indices]
    
    return top5_cities, top5_scores
```

### Mejoras del Fine-tuning

**Antes** (modelo base):
- Entrenado con textos genéricos de internet
- No conoce nombres de ciudades mexicanas
- No entiende características arquitectónicas locales

**Después** (fine-tuned):
- Aprende patrones visuales de ciudades mexicanas
- Asocia landmarks específicos con ciudades
- Entiende contexto cultural y arquitectónico
- Mejora significativa en top-1 accuracy

---

## 📊 Métricas de Evaluación

### Top-1 Accuracy
```
Predicción exacta de la ciudad correcta
Ejemplo: Predice "Guadalajara" y es "Guadalajara" ✅
```

### Top-3 Accuracy
```
Ciudad correcta está en las 3 primeras predicciones
Ejemplo: Top 3 = ["Zapopan", "Guadalajara", "Tlaquepaque"]
         Real = "Guadalajara" ✅
```

### Distancia Promedio
```
Distancia haversine entre predicción y realidad (en km)

Formula:
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1−a))
d = R × c  (R = 6371 km)
```

---

## 🚀 Deployment

### Streamlit Cloud

**Variables de Entorno** (`.streamlit/secrets.toml`):
```toml
SUPABASE_URL = "https://qlwzmjyztyfnhoxfjstd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

**Ventajas**:
- No requiere archivos locales de imágenes
- Todo se carga desde Supabase Storage
- Deployment automático desde GitHub
- Colaboración en tiempo real

### Flujo de Deployment

```
1. GitHub Push
   ├─ .env → ignorado (.gitignore)
   ├─ Código → main branch
   └─ modelo.pth → Git LFS

2. Streamlit Cloud
   ├─ Conecta a repo
   ├─ Configura secrets
   ├─ Auto-deploy
   └─ URL pública

3. Supabase
   ├─ Base de datos PostgreSQL
   ├─ Storage (imágenes)
   └─ Auto-sync con Streamlit
```

---

## 🔐 Seguridad

### Row Level Security (RLS)

**Tablas de Datos**: RLS deshabilitado para desarrollo
```sql
ALTER TABLE image_metadata DISABLE ROW LEVEL SECURITY;
ALTER TABLE annotations DISABLE ROW LEVEL SECURITY;
```

**Storage**: Políticas públicas para lectura/escritura
```sql
-- Producción: añadir autenticación
CREATE POLICY "Authenticated uploads"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'geolocalization-images');
```

### Variables Sensibles

**.gitignore**:
```
.env
.streamlit/secrets.toml
token.pickle
*.pth (excepto via Git LFS)
```

---

## 📈 Roadmap Futuro

### Mejoras Planificadas

1. **Más Datos**:
   - 500+ imágenes por ciudad
   - Diversidad de ángulos, clima, época del año
   - Aumentación de datos (rotación, crop, color)

2. **Modelo Mejorado**:
   - Ensemble de modelos (CLIP + ResNet + EfficientNet)
   - Atención espacial (destacar landmarks)
   - Transfer learning progresivo

3. **Features Adicionales**:
   - Predicción de estado (32 estados)
   - Detección de landmarks específicos
   - Estimación de coordenadas precisas
   - Reconocimiento de época histórica

4. **Optimización**:
   - Cuantización de modelo (reducir tamaño)
   - Caching de embeddings
   - API REST para integración externa

---

## 🛠️ Mantenimiento

### Scripts de Utilidad

**Migración inicial** (ejecutar una vez):
```bash
# 1. Subir imágenes locales a Supabase
python upload_to_supabase_storage.py

# 2. Migrar metadata y anotaciones
python migrate_to_supabase.py
```

**Sanitización de nombres**:
```bash
# Limpiar nombres de archivos (acentos, espacios)
python rename_images.py
```

**Validación de estructura**:
```bash
# Verificar integridad de datos
python validate_structure.py
```

### Backup

**Base de Datos**: Supabase hace backups automáticos diarios

**Imágenes**: Almacenadas en Supabase Storage (redundante)

**Modelo**: 
```bash
# Guardar en Git LFS
git lfs track "*.pth"
git add model/modelo.pth
git commit -m "Update model"
```

---

## 📚 Referencias

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [Supabase Docs](https://supabase.com/docs)
- [Streamlit Docs](https://docs.streamlit.io)
- [PyTorch Docs](https://pytorch.org/docs)
- [Transformers Docs](https://huggingface.co/docs/transformers)
