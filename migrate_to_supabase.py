"""
MIGRACIÓN DE DATOS LOCALES A SUPABASE
======================================
Migra metadata.csv y annotations.csv a Supabase para trabajo colaborativo

Requisitos:
    pip install supabase python-dotenv

Configuración:
    1. Crea cuenta en https://supabase.com
    2. Crea un nuevo proyecto
    3. Ve a Settings > API y copia:
       - Project URL
       - anon/public key
    4. Crea archivo .env con:
       SUPABASE_URL=https://xxxxx.supabase.co
       SUPABASE_KEY=tu_anon_key
"""

import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Configura SUPABASE_URL y SUPABASE_KEY en archivo .env")
    print("\nCrea archivo .env con:")
    print("SUPABASE_URL=https://xxxxx.supabase.co")
    print("SUPABASE_KEY=tu_anon_key")
    exit(1)

# Conectar a Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Rutas locales
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "mining"
METADATA_CSV = DATA_DIR / "metadata.csv"
ANNOTATIONS_CSV = DATA_DIR / "annotations.csv"
DELETED_TXT = DATA_DIR / "deleted_images.txt"

def migrate_metadata():
    """Migra metadata.csv a Supabase"""
    print("\n📊 Migrando metadatos...")
    
    if not METADATA_CSV.exists():
        print("⚠️  metadata.csv no encontrado")
        return
    
    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            data = {
                'filename': row['filename'],
                'source': row['source'],
                'photo_id': row.get('photo_id', ''),
                'city': row['city'],
                'state': row['state'],
                'lat': float(row['lat']) if row['lat'] else 0.0,
                'lon': float(row['lon']) if row['lon'] else 0.0,
                'url': row.get('url', ''),
                'title': row.get('title', ''),
                'photographer': row.get('photographer', ''),
                'size': int(row.get('size', 0)) if row.get('size') else 0,
                'hash': row.get('hash', ''),
                'downloaded_at': row.get('downloaded_at', ''),
                # image_url se agregará después (ver opciones de almacenamiento)
                'image_url': None
            }
            
            try:
                # Usar upsert para evitar duplicados
                result = supabase.table('image_metadata').upsert(
                    data,
                    on_conflict='filename'
                ).execute()
                count += 1
                
                if count % 10 == 0:
                    print(f"   Procesadas: {count} imágenes...")
                    
            except Exception as e:
                print(f"❌ Error en {row['filename']}: {e}")
    
    print(f"✅ Metadatos migrados: {count} imágenes")

def migrate_annotations():
    """Migra annotations.csv a Supabase"""
    print("\n✏️  Migrando anotaciones...")
    
    if not ANNOTATIONS_CSV.exists():
        print("⚠️  annotations.csv no encontrado")
        return
    
    with open(ANNOTATIONS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        count = 0
        
        for row in reader:
            # Primero obtener el image_id desde la tabla de metadata
            result = supabase.table('image_metadata').select('id').eq(
                'filename', row['filename']
            ).execute()
            
            if not result.data:
                print(f"⚠️  Imagen no encontrada en metadata: {row['filename']}")
                continue
            
            image_id = result.data[0]['id']
            
            # Convertir custom_tags de string a array
            tags_str = row.get('custom_tags', '')
            custom_tags = [tag.strip() for tag in tags_str.split(',') if tag.strip()]
            
            data = {
                'image_id': image_id,
                'filename': row['filename'],
                'city': row['city'],
                'state': row['state'],
                'lat': float(row['lat']) if row['lat'] else 0.0,
                'lon': float(row['lon']) if row['lon'] else 0.0,
                'correct_city': row['correct_city'],
                'quality': int(row['quality']) if row.get('quality') else 3,
                'confidence': int(row['confidence']) if row.get('confidence') else 80,
                'has_landmarks': row.get('landmarks', 'False') == 'True',
                'has_architecture': row.get('architecture', 'False') == 'True',
                'has_signs': row.get('signs', 'False') == 'True',
                'has_nature': row.get('nature', 'False') == 'True',
                'has_urban': row.get('urban', 'False') == 'True',
                'has_beach': row.get('beach', 'False') == 'True',
                'has_people': row.get('people', 'False') == 'True',
                'has_vehicles': row.get('vehicles', 'False') == 'True',
                'has_text': row.get('text', 'False') == 'True',
                'custom_tags': custom_tags,
                'notes': row.get('notes', ''),
                'annotated_by': row.get('annotated_by', 'Unknown'),
                'annotated_at': row.get('annotated_at', '')
            }
            
            try:
                result = supabase.table('annotations').upsert(
                    data,
                    on_conflict='image_id'
                ).execute()
                count += 1
                
                if count % 5 == 0:
                    print(f"   Procesadas: {count} anotaciones...")
                    
            except Exception as e:
                print(f"❌ Error en {row['filename']}: {e}")
    
    print(f"✅ Anotaciones migradas: {count} registros")

def migrate_deleted_images():
    """Migra deleted_images.txt a Supabase"""
    print("\n🗑️  Migrando imágenes eliminadas...")
    
    if not DELETED_TXT.exists():
        print("⚠️  deleted_images.txt no encontrado")
        return
    
    with open(DELETED_TXT, 'r', encoding='utf-8') as f:
        filenames = [line.strip() for line in f if line.strip()]
    
    if not filenames:
        print("⚠️  No hay imágenes eliminadas para migrar")
        return
    
    count = 0
    errors = 0
    for filename in filenames:
        data = {
            'filename': filename,
            'reason': 'migrated_from_local',
            'deleted_by': 'Migration Script'
        }
        
        try:
            result = supabase.table('deleted_images').upsert(
                data,
                on_conflict='filename'
            ).execute()
            count += 1
        except Exception as e:
            errors += 1
            if errors <= 3:  # Solo mostrar primeros 3 errores
                print(f"⚠️  Error en {filename}: RLS activo")
    
    if errors > 0:
        print(f"⚠️  {errors} imágenes no se pudieron migrar (RLS activo en deleted_images)")
        print(f"   Ejecuta en Supabase SQL: ALTER TABLE deleted_images DISABLE ROW LEVEL SECURITY;")
    
    print(f"✅ Imágenes eliminadas migradas: {count}/{len(filenames)} registros")

def verify_migration():
    """Verifica los datos migrados"""
    print("\n🔍 Verificando migración...")
    
    # Contar registros
    metadata_count = supabase.table('image_metadata').select('id', count='exact').execute()
    annotations_count = supabase.table('annotations').select('id', count='exact').execute()
    deleted_count = supabase.table('deleted_images').select('id', count='exact').execute()
    
    print(f"\n📊 RESUMEN:")
    print(f"   Metadatos: {metadata_count.count} imágenes")
    print(f"   Anotaciones: {annotations_count.count} registros")
    print(f"   Eliminadas: {deleted_count.count} registros")
    
    # Usar vista de estadísticas
    stats = supabase.table('annotation_stats').select('*').execute()
    if stats.data:
        s = stats.data[0]
        print(f"\n📈 ESTADÍSTICAS:")
        print(f"   Total: {s['total_images']}")
        print(f"   Anotadas: {s['annotated_images']}")
        print(f"   Eliminadas: {s['deleted_images']}")
        print(f"   Pendientes: {s['pending_images']}")
        print(f"   Estados únicos: {s['unique_states']}")
        print(f"   Anotadores: {s['unique_annotators']}")

if __name__ == "__main__":
    print("="*60)
    print("MIGRACIÓN DE DATOS A SUPABASE")
    print("="*60)
    
    # Ejecutar migraciones
    migrate_metadata()
    migrate_annotations()
    migrate_deleted_images()
    verify_migration()
    
    print("\n✅ ¡Migración completada!")
    print("\n📝 PRÓXIMOS PASOS:")
    print("1. Configura el almacenamiento de imágenes (ver opciones abajo)")
    print("2. Actualiza training_pipeline.py para usar Supabase")
    print("3. Configura variables de entorno en Streamlit Cloud")
