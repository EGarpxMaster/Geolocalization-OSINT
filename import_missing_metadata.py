"""
Importar metadata de imágenes que están en Storage pero no en la BD.
Lee el archivo metadata.csv local y crea registros para las imágenes faltantes.
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

METADATA_CSV = Path('data/mining/metadata.csv')

def import_missing_metadata():
    """Importar metadata de imágenes faltantes desde CSV local"""
    
    print("🔍 Iniciando importación de metadata faltante...\n")
    
    # 1. Obtener todas las imágenes en Storage
    print("📦 Obteniendo lista de Storage...")
    storage_files = set()
    offset = 0
    limit = 1000
    
    while True:
        files = supabase.storage.from_('geolocalization-images').list(
            path='',
            options={'limit': limit, 'offset': offset}
        )
        
        if not files:
            break
        
        for f in files:
            if f['name'].endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
                storage_files.add(f['name'])
        
        if len(files) < limit:
            break
        
        offset += limit
    
    print(f"✅ {len(storage_files)} imágenes en Storage\n")
    
    # 2. Obtener filenames existentes en BD
    print("📊 Obteniendo registros de BD...")
    existing_records = supabase.table('image_metadata').select('filename').execute()
    existing_filenames = {record['filename'] for record in existing_records.data}
    
    print(f"✅ {len(existing_filenames)} registros en BD\n")
    
    # 3. Identificar imágenes sin metadata
    missing_filenames = storage_files - existing_filenames
    
    print(f"🔍 Imágenes sin metadata: {len(missing_filenames)}\n")
    
    if not missing_filenames:
        print("✅ Todas las imágenes tienen metadata!")
        return
    
    # 4. Cargar metadata desde CSV local
    if not METADATA_CSV.exists():
        print(f"❌ No se encontró {METADATA_CSV}")
        print("   Ejecuta mining_pipeline.py para generar el archivo metadata.csv")
        return
    
    print(f"📄 Leyendo metadata desde {METADATA_CSV}...")
    local_metadata = {}
    
    with open(METADATA_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_metadata[row['filename']] = row
    
    print(f"✅ {len(local_metadata)} registros en CSV local\n")
    
    # 5. Preparar registros a insertar
    to_insert = []
    not_found = []
    
    for filename in missing_filenames:
        if filename in local_metadata:
            metadata = local_metadata[filename]
            
            # Generar URL pública
            public_url = supabase.storage.from_('geolocalization-images').get_public_url(filename)
            
            to_insert.append({
                'filename': filename,
                'source': metadata.get('source', ''),
                'photo_id': metadata.get('photo_id', ''),
                'city': metadata.get('city', ''),
                'state': metadata.get('state', ''),
                'lat': float(metadata.get('lat', 0)),
                'lon': float(metadata.get('lon', 0)),
                'url': metadata.get('url', ''),
                'image_url': public_url,
                'title': metadata.get('title', ''),
                'photographer': metadata.get('photographer', '')
            })
        else:
            not_found.append(filename)
    
    print(f"📝 Registros a insertar: {len(to_insert)}")
    if not_found:
        print(f"⚠️  Sin metadata en CSV: {len(not_found)}")
        print("   Primeras 5:", not_found[:5])
    
    if not to_insert:
        print("\n❌ No hay registros para insertar")
        return
    
    # 6. Insertar en lotes
    print(f"\n💾 Insertando {len(to_insert)} registros...")
    
    batch_size = 100
    inserted_count = 0
    
    for i in range(0, len(to_insert), batch_size):
        batch = to_insert[i:i+batch_size]
        
        supabase.table('image_metadata').insert(batch).execute()
        
        inserted_count += len(batch)
        print(f"   ✓ Insertados {inserted_count}/{len(to_insert)} registros...")
    
    print(f"\n✅ Importación completada!")
    
    # 7. Verificar estado final
    final_count = supabase.table('image_metadata').select('id', count='exact').execute()
    with_url = supabase.table('image_metadata').select('id', count='exact').not_.is_('image_url', 'null').execute()
    
    print(f"\n📊 Estado final:")
    print(f"   - Total de registros en BD: {final_count.count}")
    print(f"   - Registros con URL: {with_url.count}")
    print(f"   - Imágenes en Storage: {len(storage_files)}")
    
    if final_count.count == len(storage_files):
        print(f"\n🎉 ¡Perfecta sincronización! BD = Storage")
    else:
        diff = abs(final_count.count - len(storage_files))
        print(f"\n⚠️  Diferencia: {diff} registros")

if __name__ == '__main__':
    import_missing_metadata()
