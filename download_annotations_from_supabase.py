"""
Descargar anotaciones desde Supabase a archivo CSV local.
Esto sincroniza las anotaciones hechas en Streamlit Cloud con tu entorno local.
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

ANNOTATIONS_CSV = Path('data/mining/annotations.csv')
MINING_DIR = Path('data/mining')

def download_annotations():
    """Descargar todas las anotaciones desde Supabase"""
    
    print("📥 Descargando anotaciones desde Supabase...\n")
    
    # 1. Obtener todas las anotaciones de Supabase
    result = supabase.table('annotations').select('*').execute()
    
    if not result.data:
        print("⚠️  No hay anotaciones en Supabase")
        return
    
    print(f"✅ {len(result.data)} anotaciones encontradas\n")
    
    # 2. Crear backup del CSV actual si existe
    if ANNOTATIONS_CSV.exists():
        backup_path = ANNOTATIONS_CSV.with_suffix('.backup.csv')
        import shutil
        shutil.copy2(ANNOTATIONS_CSV, backup_path)
        print(f"💾 Backup creado: {backup_path}")
    
    # 3. Escribir anotaciones al CSV
    print(f"💾 Escribiendo anotaciones a {ANNOTATIONS_CSV}...")
    
    MINING_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(ANNOTATIONS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        
        # Escribir encabezado
        writer.writerow([
            'filename', 'city', 'state', 'lat', 'lon',
            'correct_city', 'quality', 'confidence',
            'landmarks', 'architecture', 'signs', 'nature', 'urban', 'beach',
            'people', 'vehicles', 'text', 'custom_tags', 'notes',
            'annotated_at', 'annotated_by'
        ])
        
        # Escribir datos
        for ann in result.data:
            # Parsear elementos (están en formato JSON en Supabase)
            elements = ann.get('elements', {})
            if isinstance(elements, str):
                import json
                elements = json.loads(elements)
            
            # Limpiar tags y notas
            custom_tags = ann.get('custom_tags', [])
            if isinstance(custom_tags, list):
                tags_str = ','.join(custom_tags)
            else:
                tags_str = str(custom_tags) if custom_tags else ''
            
            notes_str = str(ann.get('notes', '')).replace('\n', ' ').replace('\r', ' ')
            
            writer.writerow([
                ann.get('filename', ''),
                ann.get('city', ''),
                ann.get('state', ''),
                ann.get('lat', 0.0),
                ann.get('lon', 0.0),
                ann.get('correct_city', ''),
                ann.get('quality', 0),
                ann.get('confidence', 0),
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
                ann.get('annotated_at', ''),
                ann.get('annotated_by', 'Unknown')
            ])
    
    print(f"✅ {len(result.data)} anotaciones descargadas\n")
    
    # 4. Descargar imágenes eliminadas
    deleted_result = supabase.table('deleted_images').select('filename').execute()
    
    if deleted_result.data:
        deleted_file = MINING_DIR / 'deleted_images.txt'
        with open(deleted_file, 'w', encoding='utf-8') as f:
            for row in deleted_result.data:
                f.write(f"{row['filename']}\n")
        
        print(f"🗑️  {len(deleted_result.data)} imágenes eliminadas sincronizadas")
    
    print(f"\n🎉 Sincronización completada!")
    print(f"   Ahora puedes entrenar el modelo localmente con las anotaciones actualizadas")

if __name__ == '__main__':
    download_annotations()
