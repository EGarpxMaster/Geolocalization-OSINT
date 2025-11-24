"""
PIPELINE DE ENTRENAMIENTO - GEOLOCALIZATION OSINT
===================================================
Sistema unificado para anotación manual y fine-tuning del modelo CLIP.

Uso:
    streamlit run training_pipeline.py
"""

import os
import json
import csv
from pathlib import Path
from datetime import datetime
import sys

# Verificar dependencias
try:
    import streamlit as st
    from PIL import Image
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import CLIPProcessor, CLIPModel
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Error: Falta instalar dependencias: {e}")
    print("💡 Ejecuta: pip install streamlit pillow pandas torch transformers tqdm")
    sys.exit(1)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MINING_DIR = DATA_DIR / "mining"
IMAGES_DIR = MINING_DIR / "images"
METADATA_FILE = MINING_DIR / "metadata.json"
ANNOTATIONS_FILE = MINING_DIR / "annotations.json"
CITIES_CSV = DATA_DIR / "cities_mx.csv"
MODEL_DIR = BASE_DIR / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Crear directorios
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# INTERFAZ PRINCIPAL STREAMLIT
# ============================================================================

def main_interface():
    """Interfaz principal unificada de Streamlit"""
    
    st.set_page_config(
        page_title="Pipeline de Entrenamiento OSINT",
        page_icon="🎓",
        layout="wide"
    )
    
    st.title("🎓 Pipeline de Entrenamiento - Geolocalizador OSINT")
    
    # Sidebar con selección de modo
    with st.sidebar:
        st.header("⚙️ Modo de Operación")
        mode = st.radio(
            "Selecciona el modo:",
            ["📝 Anotación", "🔬 Fine-tuning", "🏗️ Regenerar Modelo", "📊 Estadísticas"],
            index=0
        )
        
        st.divider()
        
        # Mostrar stats generales
        if METADATA_FILE.exists():
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            st.metric("Imágenes minadas", len(metadata.get('images', [])))
        
        if ANNOTATIONS_FILE.exists():
            with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            st.metric("Imágenes anotadas", len(annotations.get('images', [])))
    
    # Mostrar interfaz según modo
    if mode == "📝 Anotación":
        show_annotation_interface()
    elif mode == "🔬 Fine-tuning":
        show_training_interface()
    elif mode == "🏗️ Regenerar Modelo":
        show_build_model_interface()
    else:
        show_statistics_interface()

def show_annotation_interface():
    """Interfaz de anotación mejorada con etiquetas personalizadas"""
    
    st.header("📝 Anotación de Imágenes")
    st.markdown("Mejora el modelo agregando etiquetas descriptivas y verificando la calidad de las imágenes.")
    
    # Cargar datos
    if not METADATA_FILE.exists():
        st.error("❌ No hay imágenes descargadas. Ejecuta primero `mining_pipeline.py`")
        st.code("python mining_pipeline.py --mode all --images 20", language="bash")
        return
    
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Extraer imágenes - el metadata puede tener estructura simple o compleja
    images = []
    
    # Estructura simple: metadata['images'] es una lista directa
    if 'images' in metadata and isinstance(metadata['images'], list):
        images = metadata['images']
    
    # Estructura compleja: cities -> state -> city -> images
    elif 'cities' in metadata:
        for state_data in metadata['cities'].values():
            if isinstance(state_data, dict):
                for city_data in state_data.values():
                    if isinstance(city_data, dict) and 'images' in city_data:
                        images.extend(city_data['images'])
    
    # Normalizar filenames
    for img in images:
        if isinstance(img, dict):
            if 'filename' not in img and 'local_path' in img:
                img['filename'] = Path(img['local_path']).name
    
    if not images:
        st.warning("⚠️ No hay imágenes para anotar")
        return
    
    # Cargar anotaciones existentes
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        annotations = {'images': [], 'deleted_images': []}
    
    # Normalizar filenames en anotaciones
    for ann in annotations['images']:
        if 'filename' not in ann and 'local_path' in ann:
            ann['filename'] = Path(ann['local_path']).name
    
    annotated_files = {ann.get('filename', '') for ann in annotations['images']}
    deleted_files = set(annotations.get('deleted_images', []))
    pending_images = [img for img in images if img.get('filename', '') not in annotated_files and img.get('filename', '') not in deleted_files]
    
    # Estadísticas en cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📥 Total", len(images))
    col2.metric("✅ Anotadas", len(annotated_files))
    col3.metric("🗑️ Eliminadas", len(deleted_files))
    col4.metric("⏳ Pendientes", len(pending_images))
    
    if not pending_images:
        st.success("✅ ¡Todas las imágenes están procesadas!")
        st.balloons()
        
        if st.button("🔄 Revisar imágenes anotadas"):
            st.session_state.review_mode = True
            st.rerun()
        return
    
    # Inicializar estado
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
    
    if st.session_state.current_idx >= len(pending_images):
        st.session_state.current_idx = len(pending_images) - 1
    
    current_img = pending_images[st.session_state.current_idx]
    
    # Normalizar filename si viene de local_path
    if 'filename' not in current_img and 'local_path' in current_img:
        current_img['filename'] = Path(current_img['local_path']).name
    
    # Verificar que la imagen tenga el campo filename
    if 'filename' not in current_img:
        st.error(f"❌ Error: Imagen sin nombre de archivo en metadata")
        st.json(current_img)
        return
    
    # Determinar ruta de la imagen
    img_path = IMAGES_DIR / current_img['filename']
    
    # Si no existe, intentar con local_path
    if not img_path.exists() and 'local_path' in current_img:
        img_path = Path(current_img['local_path'])
        if not img_path.is_absolute():
            img_path = Path.cwd() / img_path
    
    st.divider()
    
    # Layout principal
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader(f"🖼️ Imagen {st.session_state.current_idx + 1} de {len(pending_images)}")
        
        if img_path.exists():
            image = Image.open(img_path)
            st.image(image, use_container_width=True)
            
            # Metadata de la imagen
            with st.expander("📋 Metadata de la imagen", expanded=True):
                # Mostrar ciudad prominentemente
                city_name = current_img.get('city_target', current_img.get('city', 'Desconocida'))
                state_name = current_img.get('state_target', current_img.get('state', ''))
                st.markdown(f"### 📍 {city_name}, {state_name}")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.write(f"**Archivo:** `{current_img.get('filename', 'Sin nombre')}`")
                    st.write(f"**Fuente:** {current_img.get('source', 'Desconocida')}")
                    st.write(f"**Tamaño:** {current_img.get('width', 0)} × {current_img.get('height', 0)} px")
                with col_m2:
                    st.write(f"**Coordenadas:** ({current_img.get('lat', 0):.4f}, {current_img.get('lon', 0):.4f})")
                    size_kb = current_img.get('size', 0) / 1024 if current_img.get('size') else 0
                    st.write(f"**Tamaño archivo:** {size_kb:.1f} KB")
                    st.write(f"**URL original:** [{current_img.get('title', 'Ver')}]({current_img.get('url', '#')})")
        else:
            st.error(f"❌ Imagen no encontrada: {current_img.get('filename', 'sin nombre')}")
            if st.button("Marcar como perdida y continuar"):
                annotations.setdefault('deleted_images', []).append(current_img.get('filename', ''))
                save_annotations(annotations)
                st.session_state.current_idx += 1
                st.rerun()
            return
    
    with col_right:
        st.subheader("✏️ Formulario de Anotación")
        
        # Ciudad esperada (usar city_target si existe, sino city)
        city_name = current_img.get('city_target', current_img.get('city', 'Desconocida'))
        state_name = current_img.get('state_target', current_img.get('state', ''))
        st.info(f"**Ciudad esperada:** {city_name}, {state_name}")
        
        # Verificación
        correct_city = st.radio(
            "¿La imagen corresponde a esta ciudad?",
            options=["✅ Sí", "❌ No", "🤔 No estoy seguro"],
            horizontal=True,
            key="city_verification"
        )
        
        # Calidad
        quality = st.select_slider(
            "Calidad de la imagen",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: ["⭐ Muy baja", "⭐⭐ Baja", "⭐⭐⭐ Media", "⭐⭐⭐⭐ Buena", "⭐⭐⭐⭐⭐ Excelente"][x-1]
        )
        
        # Etiquetas personalizadas
        st.markdown("**🏷️ Etiquetas personalizadas:**")
        st.caption("Agrega palabras clave que describan la imagen (separadas por comas)")
        custom_tags = st.text_input(
            "Etiquetas",
            placeholder="arquitectura colonial, plaza central, catedral, zócalo...",
            help="Estas etiquetas mejorarán la precisión del modelo"
        )
        
        # Elementos detectados
        st.markdown("**👁️ Elementos visibles:**")
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            has_landmarks = st.checkbox("🏛️ Monumentos")
            has_architecture = st.checkbox("🏘️ Arquitectura")
            has_signs = st.checkbox("🪧 Señales")
        
        with col_e2:
            has_nature = st.checkbox("🌳 Naturaleza")
            has_urban = st.checkbox("🏙️ Urbano")
            has_beach = st.checkbox("🏖️ Playa")
        
        with col_e3:
            has_people = st.checkbox("👥 Personas")
            has_vehicles = st.checkbox("🚗 Vehículos")
            has_text = st.checkbox("📝 Texto")
        
        # Confianza
        confidence = st.slider(
            "🎯 Tu confianza en esta anotación",
            min_value=0,
            max_value=100,
            value=80,
            help="0% = No estoy seguro, 100% = Muy seguro"
        )
        
        # Notas
        notes = st.text_area(
            "📝 Notas adicionales (opcional)",
            placeholder="Describe lo que ves: monumentos específicos, características únicas, clima, época del año...",
            height=100
        )
        
        st.divider()
        
        # Botones de acción
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        
        with col_b1:
            if st.button("⬅️ Anterior", use_container_width=True, disabled=(st.session_state.current_idx == 0)):
                st.session_state.current_idx -= 1
                st.rerun()
        
        with col_b2:
            if st.button("💾 Guardar", type="primary", use_container_width=True):
                # Guardar anotación (usar city_target/state_target si existen)
                annotation = {
                    'filename': current_img.get('filename', ''),
                    'city': current_img.get('city_target', current_img.get('city', '')),
                    'state': current_img.get('state_target', current_img.get('state', '')),
                    'lat': current_img.get('lat', 0),
                    'lon': current_img.get('lon', 0),
                    'source': current_img.get('source', ''),
                    'correct_city': correct_city.split()[0],  # ✅, ❌, o 🤔
                    'quality': quality,
                    'custom_tags': [tag.strip() for tag in custom_tags.split(',') if tag.strip()],
                    'elements': {
                        'landmarks': has_landmarks,
                        'architecture': has_architecture,
                        'signs': has_signs,
                        'nature': has_nature,
                        'urban': has_urban,
                        'beach': has_beach,
                        'people': has_people,
                        'vehicles': has_vehicles,
                        'text': has_text
                    },
                    'confidence': confidence,
                    'notes': notes,
                    'annotated_at': datetime.now().isoformat(),
                    'annotated_by': 'user'
                }
                
                annotations['images'].append(annotation)
                save_annotations(annotations)
                
                st.success("✅ Anotación guardada")
                
                # Siguiente imagen
                if st.session_state.current_idx < len(pending_images) - 1:
                    st.session_state.current_idx += 1
                    st.rerun()
                else:
                    st.balloons()
                    st.success("🎉 ¡Todas las imágenes anotadas!")
        
        with col_b3:
            if st.button("⏭️ Omitir", use_container_width=True):
                if st.session_state.current_idx < len(pending_images) - 1:
                    st.session_state.current_idx += 1
                    st.rerun()
        
        with col_b4:
            if st.button("🗑️ Eliminar", use_container_width=True, help="Eliminar imagen de baja calidad"):
                if st.checkbox("Confirmar eliminación", key="confirm_delete"):
                    # Agregar a lista de eliminadas
                    annotations.setdefault('deleted_images', []).append(current_img.get('filename', ''))
                    save_annotations(annotations)
                    
                    # Eliminar archivo físico
                    try:
                        img_path.unlink()
                        st.success(f"🗑️ Imagen eliminada: {current_img.get('filename', 'imagen')}")
                    except Exception as e:
                        st.warning(f"⚠️ No se pudo eliminar el archivo físico: {e}")
                    
                    st.session_state.current_idx = min(st.session_state.current_idx, len(pending_images) - 2)
                    st.rerun()
    
    # Barra de progreso
    progress = (st.session_state.current_idx + 1) / len(pending_images)
    st.progress(progress)
    st.caption(f"Progreso: {progress*100:.1f}% ({st.session_state.current_idx + 1}/{len(pending_images)})")

def save_annotations(annotations):
    """Guarda anotaciones con backup"""
    try:
        # Crear backup
        if ANNOTATIONS_FILE.exists():
            backup_file = ANNOTATIONS_FILE.with_suffix('.backup.json')
            ANNOTATIONS_FILE.rename(backup_file)
        
        # Guardar nuevo
        with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error guardando anotaciones: {e}")

def show_training_interface():
    """Interfaz para ejecutar el fine-tuning"""
    
    st.header("🔬 Fine-tuning del Modelo")
    
    # Verificar anotaciones
    if not ANNOTATIONS_FILE.exists():
        st.warning("⚠️ No hay anotaciones disponibles")
        st.info("Primero debes anotar imágenes en la sección **📝 Anotación**")
        return
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    total_annotations = len(annotations.get('images', []))
    
    if total_annotations < 50:
        st.warning(f"⚠️ Solo tienes {total_annotations} anotaciones")
        st.info("Se recomiendan al menos 100 anotaciones para mejores resultados, pero puedes intentar con 50+")
    
    st.success(f"✅ {total_annotations} imágenes anotadas disponibles")
    
    # Configuración de entrenamiento
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.number_input("Épocas", min_value=1, max_value=20, value=5)
        batch_size = st.select_slider("Batch Size", options=[2, 4, 8, 16], value=8)
        learning_rate = st.select_slider("Learning Rate", options=[1e-6, 5e-6, 1e-5, 5e-5], value=1e-5, format_func=lambda x: f"{x:.0e}")
    
    with col2:
        min_quality = st.slider("Calidad mínima", 1, 5, 2)
        min_confidence = st.slider("Confianza mínima (%)", 0, 100, 50)
    
    st.divider()
    
    # Estimación
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ **Dispositivo:** {device}")
    
    if device == "CPU":
        st.warning("⚠️ Entrenamiento en CPU será lento. Considera usar Google Colab con GPU.")
    
    estimated_time_min = epochs * (5 if device == "GPU" else 15)
    st.info(f"⏱️ **Tiempo estimado:** {estimated_time_min}-{estimated_time_min*2} minutos")
    
    # Botón de entrenamiento
    if st.button("🚀 Iniciar Fine-tuning", type="primary", use_container_width=True):
        with st.spinner("Entrenando modelo... Esto puede tardar varios minutos."):
            try:
                result = finetune_model(
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    min_quality=min_quality,
                    min_confidence=min_confidence
                )
                
                if result:
                    st.success("✅ Fine-tuning completado exitosamente!")
                    st.balloons()
                    st.info("💡 Siguiente paso: Ve a **🏗️ Regenerar Modelo** para actualizar los embeddings")
                else:
                    st.error("❌ Error durante el entrenamiento")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

def show_build_model_interface():
    """Interfaz para regenerar embeddings"""
    
    st.header("🏗️ Regenerar Embeddings del Modelo")
    
    finetuned_path = MODEL_DIR / "modelo_finetuned.pth"
    
    if not finetuned_path.exists():
        st.warning("⚠️ No existe modelo fine-tuned")
        st.info("Primero debes ejecutar el **🔬 Fine-tuning** del modelo")
        return
    
    st.success("✅ Modelo fine-tuned encontrado")
    
    st.markdown("""
    Este proceso:
    1. Carga el modelo CLIP fine-tuned
    2. Genera embeddings mejorados para todas las ciudades
    3. Guarda el modelo actualizado en `model/modelo.pth`
    
    **Tiempo estimado:** 3-5 minutos
    """)
    
    if st.button("🏗️ Regenerar Embeddings", type="primary", use_container_width=True):
        with st.spinner("Regenerando embeddings de ciudades..."):
            try:
                build_model()
                st.success("✅ Embeddings regenerados exitosamente!")
                st.balloons()
                st.info("💡 Ahora puedes probar el modelo mejorado ejecutando: `streamlit run Geolocalizador.py`")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.exception(e)

def show_statistics_interface():
    """Interfaz de estadísticas"""
    
    st.header("📊 Estadísticas del Dataset")
    
    # Cargar datos
    stats = {}
    
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        stats['total_images'] = len(metadata.get('images', []))
        stats['total_cities'] = len(metadata.get('cities', {}))
    
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        stats['annotated'] = len(annotations.get('images', []))
        stats['deleted'] = len(annotations.get('deleted_images', []))
    else:
        stats['annotated'] = 0
        stats['deleted'] = 0
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🖼️ Imágenes totales", stats.get('total_images', 0))
    col2.metric("✅ Anotadas", stats.get('annotated', 0))
    col3.metric("🗑️ Eliminadas", stats.get('deleted', 0))
    col4.metric("🏙️ Ciudades", stats.get('total_cities', 0))
    
    if not ANNOTATIONS_FILE.exists():
        st.info("No hay estadísticas de anotaciones disponibles")
        return
    
    # Gráficos
    st.divider()
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    if not annotations.get('images'):
        st.info("Aún no hay imágenes anotadas")
        return
    
    # Distribución de calidad
    st.subheader("📈 Distribución de Calidad")
    quality_dist = {}
    for ann in annotations['images']:
        q = ann.get('quality', 0)
        quality_dist[q] = quality_dist.get(q, 0) + 1
    
    quality_df = pd.DataFrame([
        {'Calidad': f"{'⭐' * k} ({k})", 'Cantidad': v}
        for k, v in sorted(quality_dist.items())
    ])
    st.bar_chart(quality_df.set_index('Calidad'))
    
    # Distribución por ciudad
    st.subheader("🗺️ Top 10 Ciudades Anotadas")
    city_dist = {}
    for ann in annotations['images']:
        city_key = f"{ann.get('city')}, {ann.get('state')}"
        city_dist[city_key] = city_dist.get(city_key, 0) + 1
    
    top_cities = sorted(city_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    city_df = pd.DataFrame(top_cities, columns=['Ciudad', 'Anotaciones'])
    st.dataframe(city_df, use_container_width=True, hide_index=True)
    
    # Elementos detectados
    st.subheader("👁️ Elementos Más Comunes")
    elements_count = {}
    for ann in annotations['images']:
        for element, value in ann.get('elements', {}).items():
            if value:
                elements_count[element] = elements_count.get(element, 0) + 1
    
    if elements_count:
        elements_df = pd.DataFrame([
            {'Elemento': k, 'Frecuencia': v}
            for k, v in sorted(elements_count.items(), key=lambda x: x[1], reverse=True)
        ])
        st.bar_chart(elements_df.set_index('Elemento'))
    
    # Etiquetas personalizadas más usadas
    st.subheader("🏷️ Etiquetas Personalizadas Populares")
    tags_count = {}
    for ann in annotations['images']:
        for tag in ann.get('custom_tags', []):
            if tag:
                tags_count[tag] = tags_count.get(tag, 0) + 1
    
    if tags_count:
        top_tags = sorted(tags_count.items(), key=lambda x: x[1], reverse=True)[:15]
        tags_df = pd.DataFrame(top_tags, columns=['Etiqueta', 'Uso'])
        st.dataframe(tags_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se han agregado etiquetas personalizadas aún")

# ============================================================================
# DATASET PERSONALIZADO
# ============================================================================

class GeoDataset(Dataset):
    """Dataset personalizado para fine-tuning de CLIP"""
    
    def __init__(self, annotations, processor, min_quality=2, min_confidence=50):
        self.processor = processor
        self.samples = []
        
        # Cargar ciudades
        cities = {}
        with open(CITIES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cities[f"{row['name']}_{row['state']}"] = row
        
        # Filtrar por calidad y confianza
        for ann in annotations['images']:
            if (ann.get('quality', 0) >= min_quality and 
                ann.get('confidence', 0) >= min_confidence and
                ann.get('correct_city') == 'Sí'):
                
                img_path = IMAGES_DIR / ann['filename']
                if img_path.exists():
                    city_key = f"{ann['city']}_{ann['state']}"
                    if city_key in cities:
                        self.samples.append({
                            'image_path': img_path,
                            'city': ann['city'],
                            'state': ann['state'],
                            'tags': cities[city_key].get('tags', '')
                        })
        
        print(f"✅ Dataset creado: {len(self.samples)} imágenes válidas")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Cargar imagen
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Crear prompts variados
        prompts = [
            f"a photo of {sample['city']}, {sample['state']}, Mexico",
            f"{sample['city']}, {sample['state']}",
            f"landscape of {sample['city']}, Mexico",
            f"{sample['city']} city view",
        ]
        
        # Agregar tags si existen
        if sample['tags']:
            tags = sample['tags'].split(',')
            for tag in tags[:3]:
                prompts.append(f"{tag.strip()} in {sample['city']}")
        
        # Procesar con CLIP
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }

# ============================================================================
# FINE-TUNING
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Pérdida contrastiva para CLIP"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_embeds, text_embeds):
        # Normalizar embeddings
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # Calcular similitud
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        
        # Labels (diagonal = positivos)
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Pérdida bidireccional
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Entrena una época"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Entrenando"):
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids.squeeze(1),
            attention_mask=attention_mask.squeeze(1)
        )
        
        # Calcular pérdida
        loss = criterion(outputs.image_embeds, outputs.text_embeds)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Valida el modelo"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validando"):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids.squeeze(1),
                attention_mask=attention_mask.squeeze(1)
            )
            
            loss = criterion(outputs.image_embeds, outputs.text_embeds)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def finetune_model(epochs=5, batch_size=8, learning_rate=1e-5, min_quality=2, min_confidence=50):
    """Fine-tuning del modelo CLIP con interfaz Streamlit"""
    
    # Verificar anotaciones
    if not ANNOTATIONS_FILE.exists():
        st.error("❌ No hay anotaciones")
        return False
    
    with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    # Crear contenedor para logs
    log_container = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_container.info(f"🖥️ Usando: {device}")
    
    # Cargar modelo y procesador
    status_text.text("📦 Cargando CLIP...")
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    progress_bar.progress(0.1)
    
    # Crear dataset
    status_text.text("📊 Creando dataset...")
    dataset = GeoDataset(annotations, processor, min_quality, min_confidence)
    
    if len(dataset) < 20:
        st.error(f"❌ Dataset muy pequeño: {len(dataset)} imágenes")
        st.info("💡 Ajusta los filtros o anota más imágenes")
        return False
    
    log_container.success(f"✅ Dataset creado: {len(dataset)} imágenes válidas")
    progress_bar.progress(0.2)
    
    # Split train/val
    val_size = int(len(dataset) * 0.15)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    log_container.info(f"📈 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # Configurar entrenamiento
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss()
    
    best_val_loss = float('inf')
    progress_bar.progress(0.3)
    
    # Contenedor para métricas
    metrics_placeholder = st.empty()
    
    # Entrenamiento
    for epoch in range(epochs):
        status_text.text(f"📍 Época {epoch+1}/{epochs}")
        
        # Simulación de progreso (en producción usarías callbacks reales)
        epoch_progress = 0.3 + (0.6 * (epoch + 1) / epochs)
        progress_bar.progress(epoch_progress)
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Mostrar métricas
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            col1.metric(f"Época {epoch+1} - Train Loss", f"{train_loss:.4f}")
            col2.metric(f"Época {epoch+1} - Val Loss", f"{val_loss:.4f}")
        
        # Guardar checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODEL_DIR / "modelo_finetuned.pth"
            torch.save(model.state_dict(), best_model_path)
            log_container.success(f"⭐ Mejor modelo en época {epoch+1} (val_loss: {val_loss:.4f})")
    
    progress_bar.progress(1.0)
    status_text.text("✅ Fine-tuning completado")
    
    return True

# ============================================================================
# REGENERAR EMBEDDINGS
# ============================================================================

def build_model():
    """Regenera embeddings de ciudades con el modelo fine-tuned (versión Streamlit)"""
    
    finetuned_path = MODEL_DIR / "modelo_finetuned.pth"
    
    # Crear contenedores
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status_text.text(f"🖥️ Usando: {device}")
    
    # Cargar modelo fine-tuned
    status_text.text("📦 Cargando modelo fine-tuned...")
    model_name = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.load_state_dict(torch.load(finetuned_path, map_location=device, weights_only=False))
    model.eval()
    progress_bar.progress(0.2)
    
    # Cargar ciudades
    status_text.text("📍 Cargando ciudades...")
    cities = []
    with open(CITIES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cities = list(reader)
    
    st.success(f"✅ {len(cities)} ciudades encontradas")
    progress_bar.progress(0.3)
    
    # Generar embeddings
    city_embeds = []
    status_text.text("🔄 Generando embeddings...")
    
    with torch.no_grad():
        for idx, city in enumerate(cities):
            # Actualizar progreso
            progress = 0.3 + (0.6 * (idx + 1) / len(cities))
            progress_bar.progress(progress)
            status_text.text(f"🔄 Procesando: {city['name']}, {city['state']} ({idx+1}/{len(cities)})")
            
            # Crear prompts múltiples
            prompts = [
                f"a photo of {city['name']}, {city['state']}, Mexico",
                f"{city['name']}, {city['state']}",
                f"landscape of {city['name']}, Mexico",
                f"tourist destination {city['name']}",
            ]
            
            # Agregar tags
            if city.get('tags'):
                tags = city['tags'].split(',')
                for tag in tags[:3]:
                    prompts.append(f"{tag.strip()} in {city['name']}")
            
            # Procesar textos
            inputs = processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**inputs)
            
            # Promediar embeddings
            avg_embed = text_features.mean(dim=0)
            city_embeds.append(avg_embed.cpu().numpy())
    
    # Guardar modelo completo
    status_text.text("💾 Guardando modelo...")
    output_path = MODEL_DIR / "modelo.pth"
    torch.save({
        'city_embeds': torch.tensor(city_embeds),
        'cities': cities,
        'model_name': model_name,
        'states': list(set(c['state'] for c in cities)),
        'state_embeds': {}  # Placeholder
    }, output_path)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Embeddings regenerados")
    
    st.success(f"📁 Modelo guardado en: {output_path}")
    st.info(f"🎯 {len(cities)} ciudades procesadas")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    main_interface()
