# 📸 Instrucciones para Minado de Imágenes

## 🔄 Sistema Actualizado con CSV

El sistema ahora guarda los metadatos en **CSV** automáticamente, además del JSON original. Esto hace más fácil trabajar con las imágenes.

---

## 🚀 Cómo Minar Imágenes

### Opción 1: Minar todas las ciudades (20 imágenes por ciudad)
```bash
python mining_pipeline.py --mode all --images 20
```

### Opción 2: Minar un estado específico
```bash
python mining_pipeline.py --mode state --state "Jalisco" --images 30
```

### Opción 3: Minar una ciudad específica
```bash
python mining_pipeline.py --mode city --city "Guadalajara" --images 50
```

### Verificar progreso
```bash
python mining_pipeline.py --check-progress
```

---

## 📁 Archivos Generados

Después de minar, se crearán automáticamente:

```
data/mining/
├── images/                    ← Imágenes descargadas
│   ├── pexels_Acapulco_Guerrero_0_1764040142.jpg
│   ├── pexels_Guadalajara_Jalisco_0_1764040150.jpg
│   └── ...
├── metadata.json             ← Metadata completo (original)
└── metadata.csv              ← Metadata en CSV (NUEVO - más fácil) ✨
```

---

## 📊 Formato del CSV

El archivo `metadata.csv` contiene:

| Columna | Descripción |
|---------|-------------|
| `filename` | Nombre del archivo de imagen |
| `source` | Fuente (pexels, wikimedia, wikipedia) |
| `photo_id` | ID único de la foto |
| `city` | Ciudad de la imagen |
| `state` | Estado de la imagen |
| `lat` | Latitud |
| `lon` | Longitud |
| `url` | URL original |
| `title` | Título/descripción |
| `photographer` | Fotógrafo/autor |
| `downloaded_at` | Fecha de descarga |
| `size` | Tamaño en bytes |
| `hash` | Hash MD5 (para deduplicación) |

**Ventajas del CSV:**
- ✅ Abre en Excel/Google Sheets
- ✅ Fácil de filtrar y ordenar
- ✅ Compatible con pandas
- ✅ Más ligero que JSON

---

## 🎓 Flujo de Trabajo Completo

### 1. Limpiar datos antiguos (opcional)
Si quieres empezar de cero:
```bash
Remove-Item "data\mining\images\*" -Force
Remove-Item "data\mining\metadata.*" -Force
Remove-Item "data\mining\annotations.*" -Force
```

### 2. Minar imágenes nuevas
```bash
python mining_pipeline.py --mode all --images 20
```

Esto descargará ~1,900 imágenes (95 ciudades × 20 imágenes).

**Tiempo estimado:** 30-60 minutos (depende de tu conexión)

### 3. Verificar que funcionó
```bash
# Ver cuántas imágenes se descargaron
Get-ChildItem "data\mining\images" | Measure-Object

# Ver el CSV generado
Get-Content "data\mining\metadata.csv" -Head 10
```

### 4. Anotar imágenes
```bash
streamlit run training_pipeline.py
```

Selecciona modo **"📝 Anotación"** y anota al menos 50-100 imágenes.

### 5. Entrenar modelo
1. Modo **"🔬 Fine-tuning"** → Entrenar con tus anotaciones
2. Modo **"🏗️ Regenerar Modelo"** → Crear embeddings mejorados
3. Modo **"🎯 Evaluación"** → Medir precisión

---

## 🔧 Solución de Problemas

### "❌ Imagen no encontrada"
**Causa:** El metadata apunta a archivos que no existen.

**Solución:** Volver a minar las imágenes:
```bash
# Limpiar todo
Remove-Item "data\mining\*" -Recurse -Force

# Minar de nuevo
python mining_pipeline.py --mode all --images 20
```

### "No hay imágenes para anotar"
**Causa:** No se ha ejecutado el minado.

**Solución:**
```bash
python mining_pipeline.py --mode all --images 20
```

### "Error cargando CSV"
**Causa:** El archivo CSV está corrupto o no existe.

**Solución:** Regenerar metadata ejecutando el minado de nuevo.

---

## 💡 Recomendaciones

1. **Primera vez:** Empieza con pocas imágenes para probar
   ```bash
   python mining_pipeline.py --mode state --state "Jalisco" --images 5
   ```

2. **Para producción:** Mina al menos 20-50 imágenes por ciudad
   ```bash
   python mining_pipeline.py --mode all --images 50
   ```

3. **Respaldo:** El sistema guarda JSON y CSV. Si uno falla, usa el otro.

4. **Monitoreo:** Usa `--check-progress` para ver el progreso en tiempo real.

---

## 📝 Notas Importantes

- Las imágenes se nombran automáticamente: `{fuente}_{ciudad}_{estado}_{index}_{timestamp}.jpg`
- El sistema evita duplicados usando hashes MD5
- Solo descarga imágenes entre 50KB y 10MB
- Valida que sean imágenes reales (no HTML ni errores)
- Usa rate limiting para no saturar las APIs

---

## ✅ Verificación Final

Después de minar, verifica que todo esté bien:

```powershell
# 1. Contar imágenes
(Get-ChildItem "data\mining\images" -Filter "*.jpg").Count

# 2. Ver tamaño total
(Get-ChildItem "data\mining\images" | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "GB"

# 3. Ver CSV
Import-Csv "data\mining\metadata.csv" | Select-Object -First 5 | Format-Table

# 4. Verificar que los filenames coincidan
$csvFiles = Import-Csv "data\mining\metadata.csv" | Select-Object -ExpandProperty filename
$actualFiles = Get-ChildItem "data\mining\images" -Filter "*.jpg" | Select-Object -ExpandProperty Name
Compare-Object $csvFiles $actualFiles
```

Si `Compare-Object` no muestra diferencias, ¡todo está perfecto! ✅

---

¡Ahora estás listo para minar imágenes correctamente! 🎉
