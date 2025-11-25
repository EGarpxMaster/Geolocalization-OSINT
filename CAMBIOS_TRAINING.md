# Cambios en el Sistema de Entrenamiento

## 📋 Resumen de Cambios

### 1. **Anotador Fijo: Emma**
- Todas las anotaciones ahora se registran con el anotador: **Emma**
- Campo `annotated_by` automáticamente configurado

### 2. **Formato CSV para Fine-tuning** 
- Las anotaciones ahora se guardan en **dos formatos**:
  - `data/mining/annotations.json` (formato original, para compatibilidad)
  - `data/mining/annotations.csv` (nuevo, optimizado para análisis)

#### Estructura del CSV:
```csv
filename,city,state,lat,lon,correct_city,quality,confidence,
landmarks,architecture,signs,nature,urban,beach,people,vehicles,text,
custom_tags,notes,annotated_at,annotated_by
```

**Ventajas del CSV:**
- ✅ Más fácil de importar en Excel/Google Sheets
- ✅ Compatible con pandas para análisis
- ✅ Más ligero que JSON
- ✅ Ideal para machine learning (scikit-learn, etc.)

### 3. **Nuevo Modo: 🎯 Evaluación**
Sistema completo de evaluación del modelo con métricas profesionales.

#### Características:
- **Selección de muestras**: Evalúa 10-100 imágenes aleatorias
- **Filtro de calidad**: Evalúa solo imágenes de alta calidad
- **Métricas calculadas**:
  - 🎯 **Precisión Total**: % de predicciones correctas
  - ✅ **Top-1 Accuracy**: Ciudad exacta en primera predicción
  - 🔝 **Top-3 Accuracy**: Ciudad correcta en top 3 predicciones
  - 📏 **Distancia Promedio**: Error geográfico en km

#### Visualizaciones:
- 📊 Tabla detallada de predicciones
- 🏙️ Precisión por ciudad (ranking)
- 📈 Histograma de distribución de errores
- ✅/❌ Indicadores visuales de aciertos

---

## 🚀 Cómo Usar

### Paso 1: Anotar Imágenes
```bash
streamlit run training_pipeline.py
```
1. Selecciona modo **"📝 Anotación"**
2. Anota al menos 50-100 imágenes
3. Las anotaciones se guardan automáticamente en JSON + CSV

### Paso 2: Fine-tuning del Modelo
1. Selecciona modo **"🔬 Fine-tuning"**
2. Configura épocas, batch size, learning rate
3. Ejecuta el entrenamiento (puede tardar 10-30 minutos)
4. El mejor modelo se guarda en `model/modelo_finetuned.pth`

### Paso 3: Regenerar Embeddings
1. Selecciona modo **"🏗️ Regenerar Modelo"**
2. Genera embeddings de todas las ciudades con el modelo mejorado
3. Modelo final guardado en `model/modelo.pth`

### Paso 4: Evaluar Precisión
1. Selecciona modo **"🎯 Evaluación"**
2. Configura número de muestras y calidad mínima
3. Ejecuta evaluación
4. Revisa métricas y análisis detallado

---

## 📊 Ejemplo de Salida de Evaluación

```
🎯 Precisión Total: 87.5%
✅ Top-1 Accuracy: 87.5%
🔝 Top-3 Accuracy: 95.0%
📏 Distancia Promedio: 12.3 km

Precisión por Ciudad:
- Cancún, Quintana Roo: 100% (5/5)
- Querétaro, Querétaro: 100% (5/5)
- Ciudad de México, CDMX: 80% (4/5)
- Guadalajara, Jalisco: 75% (3/4)
```

---

## 📁 Archivos Generados

```
data/mining/
├── annotations.json        # Anotaciones en JSON (original)
├── annotations.csv         # Anotaciones en CSV (nuevo)
├── annotations.backup.json # Backup automático
└── images/                 # Imágenes descargadas

model/
├── modelo.pth             # Modelo final con embeddings
├── modelo_finetuned.pth   # Modelo CLIP fine-tuned
└── checkpoints/           # Checkpoints de entrenamiento
```

---

## 🔧 Configuración Técnica

### CSV Export Function
La función `save_annotation_to_csv()` exporta automáticamente cada anotación con:
- Metadatos de la imagen (filename, ciudad, estado, coordenadas)
- Calidad y confianza de la anotación
- Elementos visuales detectados (9 categorías)
- Tags personalizados
- Notas de Emma
- Timestamp y anotador

### Evaluación con Haversine Distance
Calcula la distancia geográfica real entre:
- Ciudad predicha por el modelo
- Ciudad real de la imagen

Permite medir no solo "correcto/incorrecto" sino también **qué tan lejos** se equivocó el modelo.

---

## 💡 Recomendaciones

1. **Anotar al menos 100 imágenes** antes de hacer fine-tuning
2. **Usar calidad ≥ 3** para evaluación confiable
3. **Incluir variedad de ciudades** (no solo las populares)
4. **Agregar tags descriptivos** para mejorar el modelo
5. **Evaluar después de cada entrenamiento** para medir mejora

---

## 🐛 Troubleshooting

**Error: "No hay anotaciones en CSV"**
- Solución: Anota al menos 1 imagen en el modo "📝 Anotación"

**Error: "Dataset muy pequeño"**
- Solución: Necesitas mínimo 20 imágenes para fine-tuning

**Precisión muy baja (<50%)**
- Revisar calidad de las anotaciones
- Aumentar número de épocas de entrenamiento
- Anotar más imágenes variadas

---

## 👤 Anotador: Emma
Todas las anotaciones realizadas por: **Emma**
- Configuración automática en el código
- No requiere input manual
- Aparece en todos los registros CSV/JSON
