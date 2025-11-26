"""
Sincronizar URLs de Supabase Storage con la base de datos.
Este script actualiza el campo image_url en image_metadata para todas las imágenes
que existen en el bucket de Storage.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Inicializar cliente Supabase
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

def sync_storage_urls():
    """Sincronizar URLs de Storage con la base de datos"""
    
    # 1. Listar todas las imágenes en Storage (con paginación)
    print("📦 Obteniendo lista de imágenes en Storage...")
    all_files = []
    offset = 0
    limit = 1000
    
    while True:
        storage_files = supabase.storage.from_('geolocalization-images').list(
            path='',
            options={'limit': limit, 'offset': offset}
        )
        
        if not storage_files:
            break
        
        all_files.extend(storage_files)
        
        if len(storage_files) < limit:
            break
        
        offset += limit
    
    # Filtrar solo archivos de imagen (ignorar .emptyFolderPlaceholder)
    image_files = [f['name'] for f in all_files if f['name'].endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif'))]
    
    print(f"✅ Encontradas {len(image_files)} imágenes en Storage")
    
    # 2. Para cada imagen, generar su URL pública y actualizar BD
    updated_count = 0
    missing_count = 0
    
    for filename in image_files:
        # Generar URL pública
        public_url = supabase.storage.from_('geolocalization-images').get_public_url(filename)
        
        # Verificar si existe en la base de datos
        result = supabase.table('image_metadata').select('id').eq('filename', filename).execute()
        
        if result.data:
            # Actualizar URL
            supabase.table('image_metadata').update({
                'image_url': public_url
            }).eq('filename', filename).execute()
            
            updated_count += 1
            if updated_count % 10 == 0:
                print(f"  ✓ Actualizadas {updated_count} URLs...")
        else:
            missing_count += 1
            print(f"  ⚠️ Archivo sin registro en BD: {filename}")
    
    print(f"\n✅ Sincronización completada:")
    print(f"   - URLs actualizadas: {updated_count}")
    print(f"   - Archivos sin registro en BD: {missing_count}")
    
    # 3. Verificar cuántos registros tienen URL
    total_with_url = supabase.table('image_metadata').select('id', count='exact').not_.is_('image_url', 'null').execute()
    total_without_url = supabase.table('image_metadata').select('id', count='exact').is_('image_url', 'null').execute()
    
    print(f"\n📊 Estado de la base de datos:")
    print(f"   - Registros CON URL: {total_with_url.count}")
    print(f"   - Registros SIN URL: {total_without_url.count}")
    
    if total_without_url.count > 0:
        print(f"\n⚠️ IMPORTANTE: Hay {total_without_url.count} registros sin imagen en Storage.")
        print(f"   Estos registros no podrán cargar imágenes en la interfaz de anotación.")
        print(f"   Solución: Ejecutar mining_pipeline.py para descargar las imágenes faltantes.")

if __name__ == '__main__':
    print("🔄 Iniciando sincronización de URLs de Supabase Storage...\n")
    sync_storage_urls()
    print("\n✅ Proceso completado")
