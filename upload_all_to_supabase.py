"""
Subir TODAS las imágenes locales a Supabase Storage y actualizar URLs.
Este script procesa todas las imágenes en data/mining/images/ que no estén en Storage.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from tqdm import tqdm

load_dotenv()

# Inicializar cliente Supabase
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

IMAGES_DIR = Path('data/mining/images')

def upload_all_images():
    """Subir todas las imágenes locales a Supabase Storage"""
    
    # 1. Obtener lista de imágenes en Storage (con paginación)
    print("📦 Verificando imágenes existentes en Storage...")
    existing_files = set()
    offset = 0
    limit = 1000  # Máximo por página
    
    while True:
        storage_files = supabase.storage.from_('geolocalization-images').list(
            path='',
            options={'limit': limit, 'offset': offset}
        )
        
        if not storage_files:
            break
        
        for f in storage_files:
            existing_files.add(f['name'])
        
        if len(storage_files) < limit:
            break
        
        offset += limit
    
    print(f"   ✓ {len(existing_files)} archivos ya en Storage")
    
    # 2. Buscar todas las imágenes locales
    print("\n📂 Buscando imágenes locales...")
    if not IMAGES_DIR.exists():
        print(f"❌ Error: No se encontró el directorio {IMAGES_DIR}")
        return
    
    local_images = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.jpeg')) + \
                   list(IMAGES_DIR.glob('*.png')) + list(IMAGES_DIR.glob('*.webp')) + \
                   list(IMAGES_DIR.glob('*.avif'))
    
    print(f"   ✓ {len(local_images)} imágenes locales encontradas")
    
    # 3. Filtrar las que NO están en Storage
    to_upload = [img for img in local_images if img.name not in existing_files]
    print(f"   ✓ {len(to_upload)} imágenes pendientes de subir")
    
    if not to_upload:
        print("\n✅ Todas las imágenes ya están en Storage")
        return
    
    # 4. Subir imágenes faltantes
    print(f"\n🚀 Subiendo {len(to_upload)} imágenes a Supabase Storage...")
    uploaded = 0
    failed = 0
    
    for img_path in tqdm(to_upload, desc="Subiendo"):
        try:
            # Leer archivo
            with open(img_path, 'rb') as f:
                file_data = f.read()
            
            # Subir a Storage (con upsert para reemplazar si existe)
            try:
                supabase.storage.from_('geolocalization-images').upload(
                    path=img_path.name,
                    file=file_data,
                    file_options={"content-type": "image/jpeg", "upsert": "true"}
                )
            except Exception as upload_error:
                # Si falla, intentar actualizar en vez de crear
                if 'Duplicate' in str(upload_error) or '409' in str(upload_error):
                    supabase.storage.from_('geolocalization-images').update(
                        path=img_path.name,
                        file=file_data,
                        file_options={"content-type": "image/jpeg"}
                    )
                else:
                    raise
            
            # Generar URL pública
            public_url = supabase.storage.from_('geolocalization-images').get_public_url(img_path.name)
            
            # Actualizar URL en la base de datos
            supabase.table('image_metadata').update({
                'image_url': public_url
            }).eq('filename', img_path.name).execute()
            
            uploaded += 1
            
        except Exception as e:
            failed += 1
            if failed <= 5:  # Solo mostrar los primeros 5 errores
                print(f"\n❌ Error subiendo {img_path.name}: {e}")
    
    print(f"\n✅ Subida completada:")
    print(f"   - Imágenes subidas exitosamente: {uploaded}")
    print(f"   - Fallos: {failed}")
    
    # 5. Verificar estado final
    print("\n📊 Verificando estado final...")
    total_with_url = supabase.table('image_metadata').select('id', count='exact').not_.is_('image_url', 'null').execute()
    total_without_url = supabase.table('image_metadata').select('id', count='exact').is_('image_url', 'null').execute()
    
    print(f"   - Registros CON URL: {total_with_url.count}")
    print(f"   - Registros SIN URL: {total_without_url.count}")
    
    if total_without_url.count > 0:
        print(f"\n⚠️ Aún quedan {total_without_url.count} registros sin imagen.")
        print(f"   Estos pueden ser registros huérfanos sin archivo local.")

if __name__ == '__main__':
    print("🚀 Iniciando carga masiva de imágenes a Supabase Storage...\n")
    upload_all_images()
    print("\n✅ Proceso completado")
