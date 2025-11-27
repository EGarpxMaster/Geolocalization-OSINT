"""
Subir anotaciones locales (JSON y CSV) a Supabase.
Esto sube las anotaciones que se hicieron localmente antes de la integración con Supabase.
"""

import os
import csv
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

ANNOTATIONS_CSV = Path('data/mining/annotations.csv')
ANNOTATIONS_JSON = Path('data/mining/annotations.json')

def upload_annotations():
    """Subir todas las anotaciones locales a Supabase"""
    
    print("📤 Subiendo anotaciones locales a Supabase...\n")
    
    # 1. Leer anotaciones desde JSON (fuente principal con más datos)
    local_annotations = []
    
    if ANNOTATIONS_JSON.exists():
        print(f"📄 Leyendo anotaciones desde {ANNOTATIONS_JSON}...")
        with open(ANNOTATIONS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
            local_annotations = data.get('images', [])
        print(f"✅ {len(local_annotations)} anotaciones en JSON\n")
    elif ANNOTATIONS_CSV.exists():
        print(f"📄 Leyendo anotaciones desde {ANNOTATIONS_CSV}...")
        with open(ANNOTATIONS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            local_annotations = list(reader)
        print(f"✅ {len(local_annotations)} anotaciones en CSV\n")
    else:
        print("❌ No se encontraron archivos de anotaciones")
        return
    
    # 3. Obtener anotaciones existentes en Supabase
    print("📊 Verificando anotaciones en Supabase...")
    existing = supabase.table('annotations').select('filename').execute()
    existing_filenames = {row['filename'] for row in existing.data}
    
    print(f"✅ {len(existing_filenames)} anotaciones ya en Supabase\n")
    
    # 4. Filtrar anotaciones nuevas
    new_annotations = [
        ann for ann in local_annotations 
        if ann['filename'] not in existing_filenames
    ]
    
    print(f"🆕 {len(new_annotations)} anotaciones nuevas para subir\n")
    
    if not new_annotations:
        print("✅ Todas las anotaciones ya están en Supabase!")
        return
    
    # 5. Preparar datos para Supabase (obteniendo image_id desde image_metadata)
    to_insert = []
    
    # Primero, obtener todos los image_id de una vez para eficiencia
    print("🔍 Obteniendo image_id desde image_metadata...")
    filenames = [ann['filename'] for ann in new_annotations]
    metadata_result = supabase.table('image_metadata').select('id, filename').in_('filename', filenames).execute()
    filename_to_id = {row['filename']: row['id'] for row in metadata_result.data}
    print(f"✅ Encontrados {len(filename_to_id)} image_id\n")
    
    for ann in new_annotations:
        # Buscar image_id
        image_id = filename_to_id.get(ann['filename'])
        
        if not image_id:
            print(f"⚠️  Saltando {ann['filename']} - no tiene image_id en image_metadata")
            continue
        
        # Detectar formato: JSON tiene 'elements' como dict, CSV tiene columnas booleanas
        if 'elements' in ann and isinstance(ann['elements'], dict):
            # Formato JSON - convertir dict de elements a columnas booleanas
            elements = ann['elements']
            has_landmarks = 'landmarks' in elements and bool(elements['landmarks'])
            has_architecture = 'architecture' in elements and bool(elements['architecture'])
            has_signs = 'signs' in elements and bool(elements['signs'])
            has_nature = 'nature' in elements and bool(elements['nature'])
            has_urban = 'urban' in elements and bool(elements['urban'])
            has_beach = 'beach' in elements and bool(elements['beach'])
            has_people = 'people' in elements and bool(elements['people'])
            has_vehicles = 'vehicles' in elements and bool(elements['vehicles'])
            has_text = 'text' in elements and bool(elements['text'])
            
            custom_tags = ann.get('custom_tags', [])
            if isinstance(custom_tags, str):
                custom_tags = [tag.strip() for tag in custom_tags.split(',') if tag.strip()]
        else:
            # Formato CSV - leer columnas booleanas directamente
            has_landmarks = ann.get('landmarks', '').lower() == 'true'
            has_architecture = ann.get('architecture', '').lower() == 'true'
            has_signs = ann.get('signs', '').lower() == 'true'
            has_nature = ann.get('nature', '').lower() == 'true'
            has_urban = ann.get('urban', '').lower() == 'true'
            has_beach = ann.get('beach', '').lower() == 'true'
            has_people = ann.get('people', '').lower() == 'true'
            has_vehicles = ann.get('vehicles', '').lower() == 'true'
            has_text = ann.get('text', '').lower() == 'true'
            custom_tags = [tag.strip() for tag in ann.get('custom_tags', '').split(',') if tag.strip()]
        
        # Convertir custom_tags a string separado por comas
        custom_tags_str = ','.join(custom_tags) if isinstance(custom_tags, list) else custom_tags
        
        to_insert.append({
            'image_id': image_id,
            'filename': ann['filename'],
            'city': ann.get('city', ''),
            'state': ann.get('state', ''),
            'lat': float(ann.get('lat', 0)),
            'lon': float(ann.get('lon', 0)),
            'correct_city': ann.get('correct_city', ''),
            'quality': int(ann.get('quality', 0)),
            'confidence': int(ann.get('confidence', 0)),
            'has_landmarks': has_landmarks,
            'has_architecture': has_architecture,
            'has_signs': has_signs,
            'has_nature': has_nature,
            'has_urban': has_urban,
            'has_beach': has_beach,
            'has_people': has_people,
            'has_vehicles': has_vehicles,
            'has_text': has_text,
            'custom_tags': custom_tags_str,
            'notes': ann.get('notes', ''),
            'annotated_at': ann.get('annotated_at', ''),
            'annotated_by': ann.get('annotated_by', 'Unknown')
        })
    
    # 6. Insertar en lotes
    print(f"💾 Insertando {len(to_insert)} anotaciones en Supabase...")
    
    batch_size = 100
    inserted_count = 0
    
    for i in range(0, len(to_insert), batch_size):
        batch = to_insert[i:i+batch_size]
        
        try:
            supabase.table('annotations').insert(batch).execute()
            inserted_count += len(batch)
            print(f"   ✓ Insertadas {inserted_count}/{len(to_insert)} anotaciones...")
        except Exception as e:
            print(f"   ❌ Error en lote {i//batch_size + 1}: {e}")
    
    print(f"\n✅ Subida completada!")
    
    # 7. Verificar estado final
    final = supabase.table('annotations').select('id', count='exact').execute()
    
    print(f"\n📊 Estado final:")
    print(f"   - Total de anotaciones en Supabase: {final.count}")
    print(f"   - Anotaciones locales: {len(local_annotations)}")
    
    if final.count >= len(local_annotations):
        print(f"\n🎉 ¡Todas las anotaciones locales están en Supabase!")
    else:
        diff = len(local_annotations) - final.count
        print(f"\n⚠️  Faltan {diff} anotaciones")

if __name__ == '__main__':
    upload_annotations()
