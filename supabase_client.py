"""
Cliente de Supabase para el sistema de anotaciones
Reemplaza CSV/JSON local con PostgreSQL en la nube
"""

import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ Configura SUPABASE_URL y SUPABASE_KEY en .env")

# Cliente global
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_all_images() -> List[Dict]:
    """Obtiene todas las imágenes con metadata desde Supabase"""
    try:
        result = supabase.table('image_metadata').select('*').execute()
        return result.data
    except Exception as e:
        print(f"❌ Error obteniendo imágenes: {e}")
        return []


def get_pending_images() -> List[Dict]:
    """Obtiene imágenes pendientes de anotar (usa la vista)"""
    try:
        result = supabase.table('pending_images').select('*').execute()
        return result.data
    except Exception as e:
        print(f"❌ Error obteniendo pendientes: {e}")
        return []


def get_annotated_filenames() -> set:
    """Obtiene conjunto de nombres de archivos ya anotados"""
    try:
        result = supabase.table('annotations').select('filename').execute()
        return {row['filename'] for row in result.data}
    except Exception as e:
        print(f"❌ Error obteniendo anotados: {e}")
        return set()


def get_deleted_filenames() -> set:
    """Obtiene conjunto de nombres de archivos eliminados"""
    try:
        result = supabase.table('deleted_images').select('filename').execute()
        return {row['filename'] for row in result.data}
    except Exception as e:
        # Si falla (por ejemplo, RLS activo), retornar vacío
        return set()


def save_annotation(annotation_data: Dict) -> bool:
    """
    Guarda una anotación en Supabase
    
    Args:
        annotation_data: Dict con todos los campos de la anotación
        
    Returns:
        True si se guardó exitosamente
    """
    try:
        # Primero obtener el image_id desde el filename
        filename = annotation_data.get('filename')
        result = supabase.table('image_metadata').select('id').eq(
            'filename', filename
        ).execute()
        
        if not result.data:
            print(f"❌ Imagen no encontrada: {filename}")
            return False
        
        image_id = result.data[0]['id']
        
        # Preparar datos para inserción
        data = {
            'image_id': image_id,
            'filename': annotation_data.get('filename'),
            'city': annotation_data.get('city'),
            'state': annotation_data.get('state'),
            'lat': float(annotation_data.get('lat', 0)),
            'lon': float(annotation_data.get('lon', 0)),
            'correct_city': annotation_data.get('correct_city'),
            'quality': int(annotation_data.get('quality', 3)),
            'confidence': int(annotation_data.get('confidence', 80)),
            'has_landmarks': annotation_data.get('landmarks', False),
            'has_architecture': annotation_data.get('architecture', False),
            'has_signs': annotation_data.get('signs', False),
            'has_nature': annotation_data.get('nature', False),
            'has_urban': annotation_data.get('urban', False),
            'has_beach': annotation_data.get('beach', False),
            'has_people': annotation_data.get('people', False),
            'has_vehicles': annotation_data.get('vehicles', False),
            'has_text': annotation_data.get('text', False),
            'custom_tags': annotation_data.get('custom_tags', []),
            'notes': annotation_data.get('notes', ''),
            'annotated_by': annotation_data.get('annotated_by', 'Unknown'),
            'annotated_at': annotation_data.get('annotated_at', datetime.now().isoformat())
        }
        
        # Upsert (insertar o actualizar si ya existe)
        result = supabase.table('annotations').upsert(
            data,
            on_conflict='image_id'
        ).execute()
        
        return True
        
    except Exception as e:
        print(f"❌ Error guardando anotación: {e}")
        return False


def mark_deleted(filename: str, reason: str = "user_deleted", deleted_by: str = "Unknown") -> bool:
    """
    Marca una imagen como eliminada
    
    Args:
        filename: Nombre del archivo
        reason: Razón de eliminación
        deleted_by: Usuario que eliminó
        
    Returns:
        True si se marcó exitosamente
    """
    try:
        data = {
            'filename': filename,
            'reason': reason,
            'deleted_by': deleted_by
        }
        
        result = supabase.table('deleted_images').upsert(
            data,
            on_conflict='filename'
        ).execute()
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error marcando como eliminada (probablemente RLS activo): {filename}")
        # No es crítico, continuar
        return False


def get_annotation_stats() -> Dict:
    """Obtiene estadísticas de anotaciones desde la vista"""
    try:
        result = supabase.table('annotation_stats').select('*').execute()
        if result.data:
            return result.data[0]
        return {}
    except Exception as e:
        print(f"❌ Error obteniendo estadísticas: {e}")
        return {}


def get_annotations_by_user(username: str) -> List[Dict]:
    """Obtiene todas las anotaciones de un usuario específico"""
    try:
        result = supabase.table('annotations').select('*').eq(
            'annotated_by', username
        ).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error obteniendo anotaciones de {username}: {e}")
        return []


def test_connection() -> bool:
    """Prueba la conexión a Supabase"""
    try:
        result = supabase.table('image_metadata').select('id').limit(1).execute()
        print("✅ Conexión a Supabase exitosa")
        return True
    except Exception as e:
        print(f"❌ Error de conexión a Supabase: {e}")
        return False


if __name__ == "__main__":
    # Prueba rápida
    print("Probando conexión a Supabase...")
    test_connection()
    
    stats = get_annotation_stats()
    if stats:
        print(f"\n📊 Estadísticas:")
        print(f"   Total: {stats.get('total_images', 0)}")
        print(f"   Anotadas: {stats.get('annotated_images', 0)}")
        print(f"   Pendientes: {stats.get('pending_images', 0)}")
        print(f"   Estados: {stats.get('unique_states', 0)}")
