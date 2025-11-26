# Deployment en Streamlit Cloud

Este archivo contiene instrucciones paso a paso para deployar el sistema en Streamlit Cloud.

## 📋 Pre-requisitos

- [x] Cuenta de GitHub
- [x] Cuenta de Streamlit Cloud (gratis)
- [x] Cuenta de Supabase (gratis)
- [x] Código subido a GitHub

## 🚀 Pasos de Deployment

### 1. Preparar Repositorio GitHub

```bash
# Asegurarte que .gitignore incluye:
.env
.streamlit/secrets.toml
token.pickle
*.pth
__pycache__/
.venv/
data/mining/images/

# Commit y push
git add .
git commit -m "Preparar para deployment"
git push origin main
```

### 2. Configurar Streamlit Cloud

1. **Ir a**: https://share.streamlit.io/
2. **Sign in** con GitHub
3. **New app** → Seleccionar repo
4. **Configurar**:
   - **Repository**: `EGarpxMaster/Geolocalization-OSINT`
   - **Branch**: `main`
   - **Main file path**: `training_pipeline.py` o `Geolocalizador.py`
   - **App URL**: Elegir nombre personalizado

### 3. Configurar Secrets

En Streamlit Cloud → App settings → Secrets:

```toml
# .streamlit/secrets.toml
SUPABASE_URL = "https://qlwzmjyztyfnhoxfjstd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFsd3ptanl6dHlmbmhveGZqc3RkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQwNzI1OTYsImV4cCI6MjA3OTY0ODU5Nn0.oWv4Y_Zj2pr8DJ5AUkqJII0ajpP9KSXOlTFnFqLKm2o"
```

### 4. Verificar requirements.txt

Asegurar que incluye todas las dependencias:

```txt
streamlit
pillow
pandas
torch
transformers
tqdm
python-dotenv
supabase
requests
```

### 5. Deploy

Click en **Deploy** y esperar 2-5 minutos.

## 🎯 URLs de Producción

### App de Anotación
- **URL**: https://tu-app-annotation.streamlit.app
- **Archivo**: `training_pipeline.py`
- **Uso**: Equipo anota imágenes colaborativamente

### App de Geolocalización
- **URL**: https://tu-app-geolocator.streamlit.app
- **Archivo**: `Geolocalizador.py`
- **Uso**: Usuarios finales geolocalizan fotos

## ✅ Checklist Post-Deployment

- [ ] App carga correctamente
- [ ] Conexión a Supabase funcional
- [ ] Imágenes se cargan desde Storage
- [ ] Anotaciones se guardan en BD
- [ ] Modelo de predicción funciona
- [ ] Sin errores en logs

## 🔧 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'torch'"
**Solución**: Agregar `torch` a requirements.txt

### Error: "SUPABASE_URL not found"
**Solución**: Verificar secrets en configuración de Streamlit Cloud

### Error: "Connection timeout to Supabase"
**Solución**: Verificar que la URL de Supabase es correcta

### Error: "Image not loading"
**Solución**: Verificar que las políticas de Storage están configuradas

## 📊 Monitoreo

### Métricas de Streamlit Cloud
- Views por día
- Usuarios activos
- Tiempo de respuesta
- Errores

### Métricas de Supabase
- Queries por segundo
- Storage usado
- Bandwidth consumido

## 🔐 Seguridad

### Variables Sensibles
- ✅ Nunca commitear `.env`
- ✅ Usar secrets de Streamlit Cloud
- ✅ API keys en variables de entorno

### Políticas de Supabase
- ⚠️ RLS deshabilitado actualmente (desarrollo)
- 📌 Para producción: habilitar RLS y autenticación

## 📈 Escalabilidad

### Plan Gratuito (Actual)
- Usuarios: Ilimitados
- Horas: Ilimitadas
- Storage Streamlit: 1GB
- Storage Supabase: 500MB
- Bandwidth Supabase: 5GB/mes

### Upgrade Necesario Si:
- Storage > 500MB
- Bandwidth > 5GB/mes
- Necesitas más recursos de CPU

## 🎓 Recursos

- [Streamlit Cloud Docs](https://docs.streamlit.io/streamlit-community-cloud)
- [Supabase Docs](https://supabase.com/docs)
- [GitHub Actions para CI/CD](https://docs.github.com/en/actions)

## 📞 Soporte

Si hay problemas:
1. Revisar logs en Streamlit Cloud
2. Verificar configuración de Supabase
3. Revisar este archivo
4. Contactar soporte de Streamlit/Supabase
