# 📦 Carpeta de Modelos

Esta carpeta contiene los modelos del sistema de geolocalización.

## 📥 Descargar Modelos Pre-entrenados

Si no quieres entrenar desde cero, descarga los modelos ya entrenados:

**[📂 Google Drive - Modelos](https://drive.google.com/drive/folders/1SMQZTZ1U_prWongTUwaCTURtpvYMaG8x?usp=sharing)**

### Archivos disponibles:

1. **`modelo.pth`** (Requerido)
   - Embeddings de 68 ciudades mexicanas
   - Generado con `build_model.py`
   - Peso: ~500 MB

2. **`modelo_finetuned.pth`** (Opcional - Recomendado)
   - Modelo CLIP fine-tuneado con anotaciones locales
   - Mejora la precisión 10-20x vs modelo base
   - Peso: ~1.5 GB

3. **`checkpoints/`** (Opcional)
   - Checkpoints de entrenamiento por época
   - Útil para comparar diferentes versiones

### Instalación:

```powershell
# 1. Descarga los archivos del Drive
# 2. Colócalos en esta carpeta (model/)
# 3. Verifica la estructura:

model/
├── modelo.pth              ← Archivo principal
├── modelo_finetuned.pth    ← Modelo mejorado
└── checkpoints/            ← (Opcional)
    ├── checkpoint_epoch1.pth
    ├── checkpoint_epoch2.pth
    └── ...
```

### Generar Modelos Localmente

Si prefieres generar los modelos tú mismo:

```powershell
# Generar embeddings base (modelo.pth)
python build_model.py

# Fine-tuning (requiere anotaciones)
python training_pipeline.py --train --epochs 5

# Regenerar embeddings con modelo fine-tuned
python training_pipeline.py --build-model
```

---

**Nota**: Los archivos `.pth` están excluidos del repositorio Git por su tamaño. Descárgalos del Drive o genera localmente.
