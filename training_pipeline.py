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
import requests
from io import BytesIO

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
    from dotenv import load_dotenv
    from supabase import create_client, Client
except ImportError as e:
    print(f"❌ Error: Falta instalar dependencias: {e}")
    print("💡 Ejecuta: pip install streamlit pillow pandas torch transformers tqdm python-dotenv supabase")
    sys.exit(1)

# Cargar variables de entorno y conectar a Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    USE_SUPABASE = True
else:
    USE_SUPABASE = False
    st.warning("⚠️ Supabase no configurado. Usando archivos locales.")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MINING_DIR = DATA_DIR / "mining"
IMAGES_DIR = MINING_DIR / "images"
METADATA_FILE = MINING_DIR / "metadata.json"
METADATA_CSV = MINING_DIR / "metadata.csv"
ANNOTATIONS_FILE = MINING_DIR / "annotations.json"
ANNOTATIONS_CSV = MINING_DIR / "annotations.csv"
CITIES_CSV = DATA_DIR / "cities_mx.csv"
MODEL_DIR = BASE_DIR / "model"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Crear directorios
MODEL_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def load_image_from_url(url):
    """Carga una imagen desde una URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error cargando imagen desde URL: {e}")
        return None

def load_metadata_from_supabase():
    """Carga metadata desde Supabase"""
    images = []
    try:
        result = supabase.table('image_metadata').select('*').execute()
        for row in result.data:
            images.append({
                'filename': row['filename'],
                'source': row.get('source', ''),
                'photo_id': row.get('photo_id', ''),
                'city': row['city'],
                'state': row['state'],
                'city_target': row['city'],
                'state_target': row['state'],
                'lat': float(row['lat']) if row.get('lat') else 0.0,
                'lon': float(row['lon']) if row.get('lon') else 0.0,
                'url': row.get('url', ''),
                'image_url': row.get('image_url', ''),  # URL de Supabase Storage
                'title': row.get('title', ''),
                'photographer': row.get('photographer', ''),
                'local_path': str(IMAGES_DIR / row['filename'])  # Fallback local
            })
    except Exception as e:
        st.error(f"Error cargando desde Supabase: {e}")
    return images

def load_metadata_from_csv():
    """Carga metadata desde CSV (fallback si no hay Supabase)"""
    images = []
    try:
        with open(METADATA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                images.append({
                    'filename': row['filename'],
                    'source': row['source'],
                    'photo_id': row.get('photo_id', ''),
                    'city': row['city'],
                    'state': row['state'],
                    'city_target': row['city'],
                    'state_target': row['state'],
                    'lat': float(row['lat']) if row['lat'] else 0.0,
                    'lon': float(row['lon']) if row['lon'] else 0.0,
                    'url': row.get('url', ''),
                    'image_url': row.get('image_url', ''),
                    'title': row.get('title', ''),
                    'photographer': row.get('photographer', ''),
                    'local_path': str(IMAGES_DIR / row['filename'])
                })
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
    return images

def load_metadata_from_json():
    """Carga metadata desde JSON (fallback)"""
    images = []
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Estructura simple: metadata['images']
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
    except Exception as e:
        st.error(f"Error cargando JSON: {e}")
    return images

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
            ["📝 Anotación", "🔬 Fine-tuning", "🏗️ Regenerar Modelo", "📊 Estadísticas", "🎯 Evaluación"],
            index=0
        )
        
        st.divider()
        
        # Mostrar stats generales
        if METADATA_CSV.exists():
            with open(METADATA_CSV, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                total_images = sum(1 for _ in reader)
            st.metric("Imágenes minadas", total_images)
        elif METADATA_FILE.exists():
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
    elif mode == "📊 Estadísticas":
        show_statistics_interface()
    else:
        show_evaluation_interface()

def create_balanced_list(images, annotated_files, deleted_files):
    """Crea lista balanceada: 1 imagen por estado, rotando"""
    from collections import defaultdict
    
    pending = [img for img in images if img.get('filename', '') not in annotated_files and img.get('filename', '') not in deleted_files]
    
    images_by_state = defaultdict(list)
    for img in pending:
        state = img.get('state', 'Unknown')
        images_by_state[state].append(img)
    
    # Crear lista balanceada: 1 de cada estado, luego repetir
    balanced = []
    max_per_state = max(len(imgs) for imgs in images_by_state.values()) if images_by_state else 0
    
    for round_num in range(max_per_state):
        for state in sorted(images_by_state.keys()):
            if round_num < len(images_by_state[state]):
                balanced.append(images_by_state[state][round_num])
    
    return balanced

def show_annotation_interface():
    """Interfaz de anotación mejorada con etiquetas personalizadas"""
    
    st.header("📝 Anotación de Imágenes - Modo Colaborativo")
    
    # ============================================================================
    # SELECTOR DE USUARIO (PERSISTENTE)
    # ============================================================================
    
    USERS = ["Orbe", "Isma", "Kary"]
    
    # Inicializar usuario si no existe
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    # Mostrar selector solo si no hay usuario seleccionado
    if st.session_state.current_user is None:
        st.info("👥 Selecciona tu usuario para comenzar")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            selected_user = st.selectbox(
                "Usuario:",
                options=USERS,
                key="user_selector"
            )
            
            if st.button("✅ Confirmar Usuario", type="primary", use_container_width=True):
                st.session_state.current_user = selected_user
                st.rerun()
        
        return
    
    # Mostrar usuario actual (bloqueado)
    st.success(f"👤 Usuario activo: **{st.session_state.current_user}**")
    st.caption("El usuario está bloqueado para esta sesión. Recarga la página para cambiar.")
    
    st.markdown("---")
    st.markdown("Sistema de asignación: **Round Robin por Estado** - Cada usuario trabaja en los mismos estados pero con diferentes imágenes.")
    
    # Cargar datos desde Supabase (preferido) o CSV/JSON (fallback)
    if USE_SUPABASE:
        images = load_metadata_from_supabase()
    elif METADATA_CSV.exists():
        images = load_metadata_from_csv()
    elif METADATA_FILE.exists():
        images = load_metadata_from_json()
    else:
        st.error("❌ No hay imágenes descargadas. Ejecuta primero `mining_pipeline.py`")
        st.code("python mining_pipeline.py --mode all --images 20", language="bash")
        return
    
    if not images:
        st.warning("⚠️ No hay imágenes para anotar")
        return
    
    # Cargar anotaciones desde CSV (principal) y JSON (secundario para compatibilidad)
    annotated_files = set()
    deleted_files = set()
    
    # Leer desde CSV primero (fuente de verdad)
    if ANNOTATIONS_CSV.exists():
        import csv
        with open(ANNOTATIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                annotated_files.add(row['filename'])
    
    # Leer imágenes eliminadas desde archivo de texto
    deleted_file = MINING_DIR / 'deleted_images.txt'
    if deleted_file.exists():
        with open(deleted_file, 'r', encoding='utf-8') as f:
            deleted_files = {line.strip() for line in f if line.strip()}
    
    # Cargar JSON para mantener estructura completa (se guarda en paralelo)
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        # Sincronizar deleted_images con el archivo de texto
        if deleted_files:
            annotations['deleted_images'] = list(deleted_files)
    else:
        annotations = {'images': [], 'deleted_images': list(deleted_files)}
    
    # Filtrar imágenes pendientes (no anotadas, no eliminadas)
    pending_images = [
        img for img in images 
        if img.get('filename') not in annotated_files 
        and img.get('filename') not in deleted_files
    ]
    
    # DISTRIBUCIÓN ROUND ROBIN:
    # 1. Ordenar todas las pendientes por estado (A-Z) y luego por ciudad
    pending_sorted = sorted(pending_images, key=lambda x: (x.get('state', ''), x.get('city', '')))
    
    # 2. Asignar a cada usuario según el índice: idx % 3
    user_index = USERS.index(st.session_state.current_user)
    user_queue = [img for idx, img in enumerate(pending_sorted) if idx % 3 == user_index]
    
    # Estadísticas en cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📥 Total", len(images))
    col2.metric("✅ Anotadas", len(annotated_files))
    col3.metric("🗑️ Eliminadas", len(deleted_files))
    col4.metric("⏳ Pendientes", len(pending_images))
    col5.metric(f"👤 Tu cola", len(user_queue))
    
    # Mostrar estados únicos en la cola del usuario
    unique_states = len(set(img.get('state', 'Unknown') for img in user_queue))
    st.caption(f"🗺️ Estados en tu cola: {unique_states}")
    
    if not user_queue:
        st.success(f"✅ ¡{st.session_state.current_user}, has completado todas tus imágenes asignadas!")
        st.balloons()
        
        if st.button("🔄 Cambiar de usuario"):
            st.session_state.current_user = None
            st.rerun()
        return
    
    # SIEMPRE trabajar con la primera imagen de la cola del usuario
    # (Esto garantiza que se mantenga el orden por estado)
    current_img = user_queue[0]
    
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
    
    # Verificar si la imagen actual ya está anotada
    current_filename = current_img.get('filename', '')
    already_annotated = any(
        ann.get('filename') == current_filename 
        for ann in annotations.get('images', [])
    )
    
    # Mostrar estado de anotación
    if already_annotated:
        st.info("✅ Esta imagen ya fue anotada anteriormente")
    
    # Layout principal
    col_left, col_right = st.columns([3, 2])
    
    with col_left:
        st.subheader(f"🖼️ Imagen 1 de {len(user_queue)}")
        st.caption(f"Total pendientes en el sistema: {len(pending_images)}")
        
        # Intentar cargar desde Supabase Storage primero
        image = None
        if USE_SUPABASE and current_img.get('image_url'):
            with st.spinner("Cargando imagen desde Supabase..."):
                image = load_image_from_url(current_img['image_url'])
        
        # Fallback: cargar desde archivo local
        if image is None and img_path.exists():
            try:
                image = Image.open(img_path)
            except Exception as e:
                st.error(f"❌ Imagen corrupta o inválida: {current_img.get('filename', 'sin nombre')}")
                st.caption(f"Error: {str(e)}")
                
                col_auto1, col_auto2 = st.columns(2)
                with col_auto1:
                    if st.button("🗑️ Eliminar automáticamente", type="primary", key='delete_corrupt'):
                        current_filename = current_img.get('filename', '')
                        
                        # Marcar como eliminada
                        annotations.setdefault('deleted_images', []).append(current_filename)
                        save_annotations(annotations)
                        
                        deleted_file = MINING_DIR / 'deleted_images.txt'
                        with open(deleted_file, 'a', encoding='utf-8') as f:
                            f.write(f"{current_filename}\n")
                        
                        # Eliminar archivo físico corrupto
                        try:
                            img_path.unlink()
                            print(f"✅ Imagen corrupta eliminada: {current_filename}")
                        except Exception as del_err:
                            print(f"⚠️ No se pudo eliminar archivo: {del_err}")
                        
                        # No avanzar índice - el rerun reconstruirá la lista balanceada
                        # Si había otra imagen del mismo estado, aparecerá en la misma posición
                        st.rerun()
                
                with col_auto2:
                    if st.button("⏭️ Omitir por ahora", key='skip_corrupt'):
                        st.rerun()
                return
        
        # Mostrar imagen si se cargó correctamente
        if image:
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
                    if current_img.get('image_url'):
                        st.caption("🌐 Cargada desde Supabase Storage")
                with col_m2:
                    st.write(f"**Coordenadas:** ({current_img.get('lat', 0):.4f}, {current_img.get('lon', 0):.4f})")
                    if current_img.get('url'):
                        st.write(f"**URL original:** [{current_img.get('title', 'Ver')}]({current_img.get('url', '#')})")
        else:
            st.error(f"❌ Imagen no encontrada: {current_img.get('filename', 'sin nombre')}")
            st.caption(f"No se pudo cargar desde Supabase ni desde archivo local")
            if st.button("Marcar como perdida y continuar", key='mark_missing'):
                annotations.setdefault('deleted_images', []).append(current_img.get('filename', ''))
                save_annotations(annotations)
                # No avanzar índice - reconstruir lista
                st.rerun()
            return
    
    with col_right:
        st.subheader("✏️ Formulario de Anotación")
        
        # Ciudad esperada (usar city_target si existe, sino city)
        city_name = current_img.get('city_target', current_img.get('city', 'Desconocida'))
        state_name = current_img.get('state_target', current_img.get('state', ''))
        st.info(f"**Ciudad esperada:** {city_name}, {state_name}")
        
        # Crear clave única basada en el filename para el formulario
        form_key = f"form_{current_img.get('filename', 'default')}"
        
        # Verificación
        correct_city = st.radio(
            "¿La imagen corresponde a esta ciudad?",
            options=["✅ Sí", "❌ No", "🤔 No estoy seguro"],
            horizontal=True,
            key=f"city_verification_{form_key}"
        )
        
        # Calidad
        quality = st.select_slider(
            "Relevancia de la imagen",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: ["⭐ Muy baja", "⭐⭐ Baja", "⭐⭐⭐ Media", "⭐⭐⭐⭐ Buena", "⭐⭐⭐⭐⭐ Excelente"][x-1],
            key=f"quality_{form_key}"
        )
        
        # Etiquetas personalizadas
        st.markdown("**🏷️ Etiquetas personalizadas:**")
        st.caption("Agrega palabras clave que describan la imagen (separadas por comas)")
        custom_tags = st.text_input(
            "Etiquetas",
            placeholder="arquitectura colonial, plaza central, catedral, zócalo...",
            help="Estas etiquetas mejorarán la precisión del modelo",
            key=f"custom_tags_{form_key}"
        )
        
        # Elementos detectados
        st.markdown("**👁️ Elementos visibles:**")
        col_e1, col_e2, col_e3 = st.columns(3)
        
        with col_e1:
            has_landmarks = st.checkbox("🏛️ Monumentos", key=f"landmarks_{form_key}")
            has_architecture = st.checkbox("🏘️ Arquitectura", key=f"architecture_{form_key}")
            has_signs = st.checkbox("🪧 Señales", key=f"signs_{form_key}")
        
        with col_e2:
            has_nature = st.checkbox("🌳 Naturaleza", key=f"nature_{form_key}")
            has_urban = st.checkbox("🏙️ Urbano", key=f"urban_{form_key}")
            has_beach = st.checkbox("🏖️ Playa", key=f"beach_{form_key}")
        
        with col_e3:
            has_people = st.checkbox("👥 Personas", key=f"people_{form_key}")
            has_vehicles = st.checkbox("🚗 Vehículos", key=f"vehicles_{form_key}")
            has_text = st.checkbox("📝 Texto", key=f"text_{form_key}")
        
        # Confianza
        confidence = st.slider(
            "🎯 Tu confianza en esta anotación",
            min_value=0,
            max_value=100,
            value=80,
            step=5,
            help="0% = No estoy seguro, 100% = Muy seguro",
            key=f"confidence_{form_key}"
        )
        
        # Notas
        notes = st.text_area(
            "📝 Notas adicionales (opcional)",
            placeholder="Describe lo que ves: monumentos específicos, características únicas, clima, época del año...",
            height=100,
            key=f"notes_{form_key}"
        )
        
        # Anotador fijo
        annotator = "Emma"
        
        st.divider()
        
        # Botones de acción
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            if st.button("💾 Guardar", type="primary", width='stretch'):
                # Verificar si esta imagen ya fue anotada
                current_filename = current_img.get('filename', '')
                already_annotated = any(
                    ann.get('filename') == current_filename 
                    for ann in annotations.get('images', [])
                )
                
                if already_annotated:
                    st.warning(f"⚠️ Esta imagen ya fue anotada anteriormente. Usa '⏭️ Omitir' para pasar a la siguiente.")
                else:
                    # Guardar anotación (usar city_target/state_target si existen)
                    annotation = {
                        'filename': current_filename,
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
                        'annotated_by': st.session_state.current_user  # ✅ Usuario actual
                    }
                    
                    # Guardar primero en CSV (fuente principal)
                    save_annotation_to_csv(annotation)
                    
                    # Guardar también en JSON (respaldo/compatibilidad)
                    annotations['images'].append(annotation)
                    save_annotations(annotations)
                    
                    st.success("✅ Anotación guardada (CSV ✓ + JSON ✓)")
                    
                    # Recargar interfaz para siguiente imagen en la cola
                    st.rerun()
        
        with col_b2:
            if st.button("⏭️ Omitir", width='stretch'):
                # Simplemente recargar para mostrar siguiente en cola
                st.rerun()
        
        with col_b3:
            if st.button("🗑️ Eliminar", width='stretch', help="Eliminar imagen de baja calidad"):
                current_filename = current_img.get('filename', '')
                
                # Agregar a lista de eliminadas en JSON
                annotations.setdefault('deleted_images', []).append(current_filename)
                save_annotations(annotations)
                
                # Guardar en archivo de texto para lectura rápida
                deleted_file = MINING_DIR / 'deleted_images.txt'
                with open(deleted_file, 'a', encoding='utf-8') as f:
                    f.write(f"{current_filename}\n")
                
                st.success(f"🗑️ Imagen eliminada")
                
                # Recargar interfaz para siguiente imagen
                st.rerun()
    
    # Navegación con teclado
    st.markdown("---")
    st.caption("💡 **Tip:** Usa las teclas ←/→ para navegar entre imágenes")

def save_annotations(annotations):
    """Guarda anotaciones con backup"""
    try:
        # Crear backup
        if ANNOTATIONS_FILE.exists():
            backup_file = ANNOTATIONS_FILE.with_suffix('.backup.json')
            import shutil
            shutil.copy2(ANNOTATIONS_FILE, backup_file)
        
        # Guardar nuevo
        with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Anotaciones JSON guardadas: {len(annotations.get('images', []))} anotaciones")
    except Exception as e:
        st.error(f"❌ Error guardando anotaciones JSON: {e}")
        import traceback
        traceback.print_exc()

def save_annotation_to_csv(annotation):
    """Guarda una anotación en formato CSV para fine-tuning"""
    import csv
    
    # Crear CSV si no existe
    file_exists = ANNOTATIONS_CSV.exists()
    
    try:
        with open(ANNOTATIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            
            # Escribir encabezado si es nuevo archivo
            if not file_exists:
                writer.writerow([
                    'filename', 'city', 'state', 'lat', 'lon',
                    'correct_city', 'quality', 'confidence',
                    'landmarks', 'architecture', 'signs', 'nature', 'urban', 'beach',
                    'people', 'vehicles', 'text', 'custom_tags', 'notes',
                    'annotated_at', 'annotated_by'
                ])
            
            # Escribir datos
            elements = annotation.get('elements', {})
            # Limpiar tags y notas de saltos de línea
            tags_str = ','.join(annotation.get('custom_tags', [])).replace('\n', ' ').replace('\r', ' ')
            notes_str = annotation.get('notes', '').replace('\n', ' ').replace('\r', ' ')
            
            writer.writerow([
                annotation.get('filename', ''),
                annotation.get('city', ''),
                annotation.get('state', ''),
                annotation.get('lat', 0.0),
                annotation.get('lon', 0.0),
                annotation.get('correct_city', ''),
                annotation.get('quality', 0),
                annotation.get('confidence', 0),
                elements.get('landmarks', False),
                elements.get('architecture', False),
                elements.get('signs', False),
                elements.get('nature', False),
                elements.get('urban', False),
                elements.get('beach', False),
                elements.get('people', False),
                elements.get('vehicles', False),
                elements.get('text', False),
                tags_str,
                notes_str,
                annotation.get('annotated_at', ''),
                annotation.get('annotated_by', 'Emma')
            ])
            
            print(f"✅ Anotación CSV guardada: {annotation.get('filename', '')}")
    except Exception as e:
        print(f"❌ Error guardando CSV: {e}")
        import traceback
        traceback.print_exc()


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
        min_quality = st.slider("Relevancia mínima", 1, 5, 2)
        min_confidence = st.slider("Confianza mínima (%)", 5, 100, 50)
    
    st.divider()
    
    # Estimación
    device = "GPU" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ **Dispositivo:** {device}")
    
    if device == "CPU":
        st.warning("⚠️ Entrenamiento en CPU será lento. Considera usar Google Colab con GPU.")
    
    estimated_time_min = epochs * (5 if device == "GPU" else 15)
    st.info(f"⏱️ **Tiempo estimado:** {estimated_time_min}-{estimated_time_min*2} minutos")
    
    # Botón de entrenamiento
    if st.button("🚀 Iniciar Fine-tuning", type="primary", width='stretch'):
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
    
    if st.button("🏗️ Regenerar Embeddings", type="primary", width='stretch'):
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
    
    if METADATA_CSV.exists():
        with open(METADATA_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            images = list(reader)
            stats['total_images'] = len(images)
            stats['total_cities'] = len(set(img['city'] for img in images if img.get('city')))
    elif METADATA_FILE.exists():
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
        {'Relevancia': f"{'⭐' * k} ({k})", 'Cantidad': v}
        for k, v in sorted(quality_dist.items())
    ])
    st.bar_chart(quality_df.set_index('Relevancia'))
    
    # Distribución por ciudad
    st.subheader("🗺️ Top 10 Ciudades Anotadas")
    city_dist = {}
    for ann in annotations['images']:
        city_key = f"{ann.get('city')}, {ann.get('state')}"
        city_dist[city_key] = city_dist.get(city_key, 0) + 1
    
    top_cities = sorted(city_dist.items(), key=lambda x: x[1], reverse=True)[:10]
    city_df = pd.DataFrame(top_cities, columns=['Ciudad', 'Anotaciones'])
    st.dataframe(city_df, width='stretch', hide_index=True)
    
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
        st.dataframe(tags_df, width='stretch', hide_index=True)
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
# EVALUACIÓN DE MODELO
# ============================================================================

def show_evaluation_interface():
    """Interfaz para evaluar el modelo con imágenes random"""
    
    st.header("🎯 Evaluación del Modelo")
    st.markdown("Evalúa la precisión del modelo con imágenes aleatorias.")
    
    # Verificar que exista modelo
    model_path = MODEL_DIR / "modelo.pth"
    if not model_path.exists():
        st.error("❌ No existe modelo entrenado")
        st.info("Primero debes ejecutar el **🔬 Fine-tuning** y **🏗️ Regenerar Modelo**")
        return
    
    # Verificar anotaciones CSV
    if not ANNOTATIONS_CSV.exists():
        st.error("❌ No hay anotaciones en CSV")
        st.info("Primero anota imágenes en el modo **📝 Anotación**")
        return
    
    # Leer CSV de anotaciones
    import csv
    annotations_data = []
    with open(ANNOTATIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        annotations_data = list(reader)
    
    if len(annotations_data) < 10:
        st.warning(f"⚠️ Solo hay {len(annotations_data)} anotaciones")
        st.info("Se recomienda tener al menos 20 imágenes anotadas para una evaluación significativa")
    
    st.success(f"✅ {len(annotations_data)} imágenes anotadas disponibles")
    
    # Configuración de evaluación
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    with col1:
        num_samples = st.slider("Número de imágenes a evaluar", 10, min(100, len(annotations_data)), min(20, len(annotations_data)))
    with col2:
        quality_filter = st.slider("Calidad mínima", 1, 5, 3)
    
    if st.button("🚀 Iniciar Evaluación", type="primary", width='stretch'):
        with st.spinner("Evaluando modelo..."):
            results = evaluate_model(annotations_data, num_samples, quality_filter)
            
            if results:
                st.divider()
                st.subheader("📊 Resultados de Evaluación")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🎯 Precisión Total", f"{results['accuracy']:.1f}%")
                col2.metric("✅ Top-1 Accuracy", f"{results['top1_accuracy']:.1f}%")
                col3.metric("🔝 Top-3 Accuracy", f"{results['top3_accuracy']:.1f}%")
                col4.metric("📏 Dist. Promedio", f"{results['avg_distance']:.1f} km")
                
                # Tabla de predicciones
                st.subheader("🔍 Predicciones Detalladas")
                
                df_results = pd.DataFrame(results['predictions'])
                df_results['correct'] = df_results.apply(
                    lambda row: '✅' if row['predicted_city'] == row['true_city'] else '❌',
                    axis=1
                )
                
                st.dataframe(
                    df_results[['filename', 'true_city', 'predicted_city', 'confidence', 'distance_km', 'correct']],
                    width='stretch',
                    hide_index=True
                )
                
                # Análisis por ciudad
                st.subheader("🏙️ Precisión por Ciudad")
                city_accuracy = {}
                for pred in results['predictions']:
                    city = pred['true_city']
                    if city not in city_accuracy:
                        city_accuracy[city] = {'correct': 0, 'total': 0}
                    city_accuracy[city]['total'] += 1
                    if pred['predicted_city'] == pred['true_city']:
                        city_accuracy[city]['correct'] += 1
                
                city_stats = []
                for city, stats in city_accuracy.items():
                    accuracy = (stats['correct'] / stats['total']) * 100
                    city_stats.append({
                        'Ciudad': city,
                        'Correctas': stats['correct'],
                        'Total': stats['total'],
                        'Precisión %': f"{accuracy:.1f}%"
                    })
                
                df_city = pd.DataFrame(city_stats).sort_values('Precisión %', ascending=False)
                st.dataframe(df_city, width='stretch', hide_index=True)
                
                # Gráfico de distribución de distancias
                st.subheader("📏 Distribución de Errores de Distancia")
                distances = [p['distance_km'] for p in results['predictions']]
                
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.hist(distances, bins=20, color='skyblue', edgecolor='black')
                ax.set_xlabel('Distancia del error (km)')
                ax.set_ylabel('Frecuencia')
                ax.set_title('Distribución de errores de distancia')
                st.pyplot(fig)

def evaluate_model(annotations_data, num_samples, quality_filter):
    """Evalúa el modelo con imágenes aleatorias"""
    import random
    from math import radians, cos, sin, asin, sqrt
    
    # Filtrar por calidad
    filtered_data = [
        ann for ann in annotations_data 
        if int(ann.get('quality', 0)) >= quality_filter
    ]
    
    if len(filtered_data) < num_samples:
        st.error(f"❌ Solo hay {len(filtered_data)} imágenes con calidad >= {quality_filter}")
        return None
    
    # Seleccionar muestras aleatorias
    samples = random.sample(filtered_data, num_samples)
    
    # Cargar modelo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = MODEL_DIR / "modelo.pth"
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    
    city_embeds = model_data['city_embeds'].to(device)
    cities = model_data['cities']
    model_name = model_data['model_name']
    
    # Cargar CLIP
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    
    # Cargar modelo fine-tuned si existe
    finetuned_path = MODEL_DIR / "modelo_finetuned.pth"
    if finetuned_path.exists():
        model.load_state_dict(torch.load(finetuned_path, map_location=device, weights_only=False))
    
    model.eval()
    
    # Función para calcular distancia haversine
    def haversine(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        km = 6371 * c
        return km
    
    # Evaluar cada muestra
    predictions = []
    correct_top1 = 0
    correct_top3 = 0
    total_distance = 0
    
    progress_bar = st.progress(0)
    
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            # Actualizar progreso
            progress_bar.progress((idx + 1) / len(samples))
            
            # Cargar imagen
            img_path = IMAGES_DIR / sample['filename']
            if not img_path.exists():
                continue
            
            image = Image.open(img_path).convert('RGB')
            
            # Procesar imagen
            inputs = processor(images=image, return_tensors="pt").to(device)
            image_features = model.get_image_features(**inputs)
            
            # Calcular similitudes
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            city_embeds_norm = city_embeds / city_embeds.norm(dim=-1, keepdim=True)
            similarities = torch.matmul(image_features, city_embeds_norm.t()).cpu().numpy()[0]
            
            # Top-3 predicciones
            top3_indices = similarities.argsort()[-3:][::-1]
            
            predicted_city = cities[top3_indices[0]]['name']
            predicted_state = cities[top3_indices[0]]['state']
            confidence = float(similarities[top3_indices[0]])
            
            # Ciudad verdadera
            true_city = sample['city']
            true_state = sample['state']
            true_lat = float(sample['lat'])
            true_lon = float(sample['lon'])
            
            # Calcular distancia del error
            pred_lat = float(cities[top3_indices[0]]['lat'])
            pred_lon = float(cities[top3_indices[0]]['lon'])
            distance = haversine(true_lon, true_lat, pred_lon, pred_lat)
            
            # Verificar si es correcto
            is_correct_top1 = (predicted_city == true_city)
            is_correct_top3 = any(cities[idx]['name'] == true_city for idx in top3_indices)
            
            if is_correct_top1:
                correct_top1 += 1
            if is_correct_top3:
                correct_top3 += 1
            
            total_distance += distance
            
            predictions.append({
                'filename': sample['filename'],
                'true_city': f"{true_city}, {true_state}",
                'predicted_city': f"{predicted_city}, {predicted_state}",
                'confidence': f"{confidence:.3f}",
                'distance_km': round(distance, 2),
                'correct_top1': is_correct_top1
            })
    
    progress_bar.progress(1.0)
    
    # Calcular métricas
    accuracy = (correct_top1 / len(predictions)) * 100
    top1_accuracy = (correct_top1 / len(predictions)) * 100
    top3_accuracy = (correct_top3 / len(predictions)) * 100
    avg_distance = total_distance / len(predictions)
    
    return {
        'accuracy': accuracy,
        'top1_accuracy': top1_accuracy,
        'top3_accuracy': top3_accuracy,
        'avg_distance': avg_distance,
        'predictions': predictions
    }

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    main_interface()
