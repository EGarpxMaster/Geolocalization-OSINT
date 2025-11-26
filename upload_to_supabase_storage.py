"""
SUBIR IMÁGENES A SUPABASE STORAGE
==================================
Sube imágenes locales al bucket de Supabase y actualiza las URLs en la BD
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm

# Cargar variables de entorno
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Configura SUPABASE_URL y SUPABASE_KEY en .env")
    exit(1)

# Conectar a Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configuración
BUCKET_NAME = "geolocalization-images"
IMAGES_DIR = Path("data/mining/images")

def upload_image_to_storage(file_path: Path) -> tuple[bool, str]:
    """
    Sube una imagen a Supabase Storage
    Retorna: (éxito, url_pública o error)
    """
    try:
        filename = file_path.name
        
        # Leer el archivo
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Subir a Supabase Storage
        supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            file_data,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        
        # Obtener URL pública
        url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        return True, url
        
    except Exception as e:
        return False, str(e)

def update_database_url(filename: str, url: str) -> bool:
    """Actualiza la URL en la tabla image_metadata"""
    try:
        supabase.table('image_metadata').update(
            {'image_url': url}
        ).eq('filename', filename).execute()
        return True
    except Exception as e:
        print(f"\n   ⚠️  Error actualizando BD para {filename}: {e}")
        return False

def main():
    print("="*70)
    print("SUBIR IMÁGENES A SUPABASE STORAGE")
    print("="*70)
    
    # Verificar directorio de imágenes
    if not IMAGES_DIR.exists():
        print(f"❌ No se encuentra el directorio: {IMAGES_DIR}")
        return
    
    # Obtener lista de imágenes
    image_files = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.jpeg"))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en {IMAGES_DIR}")
        return
    
    print(f"\n📁 Directorio: {IMAGES_DIR}")
    print(f"🖼️  Imágenes encontradas: {len(image_files)}")
    print(f"📦 Bucket de destino: {BUCKET_NAME}")
    
    # Confirmar
    response = input(f"\n¿Subir {len(image_files)} imágenes a Supabase? (s/n): ")
    if response.lower() != 's':
        print("❌ Operación cancelada")
        return
    
    # Subir imágenes
    print(f"\n🚀 Iniciando subida...")
    
    uploaded = 0
    failed = 0
    updated = 0
    
    for img_file in tqdm(image_files, desc="Subiendo imágenes"):
        # Subir a Storage
        success, result = upload_image_to_storage(img_file)
        
        if success:
            uploaded += 1
            url = result
            
            # Actualizar BD
            if update_database_url(img_file.name, url):
                updated += 1
        else:
            failed += 1
            tqdm.write(f"❌ Error en {img_file.name}: {result}")
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE SUBIDA")
    print("="*70)
    print(f"✅ Imágenes subidas a Storage: {uploaded}")
    print(f"✅ URLs actualizadas en BD: {updated}")
    print(f"❌ Errores: {failed}")
    print(f"📊 Total procesadas: {len(image_files)}")
    
    if uploaded > 0:
        # Verificar una URL
        print(f"\n🔍 Verificando URLs generadas...")
        try:
            result = supabase.table('image_metadata').select(
                'filename, image_url'
            ).not_.is_('image_url', 'null').limit(3).execute()
            
            print(f"\n📋 Ejemplos de URLs generadas:")
            for row in result.data:
                print(f"\n   Archivo: {row['filename']}")
                print(f"   URL: {row['image_url']}")
        except Exception as e:
            print(f"⚠️  Error verificando: {e}")
    
    print(f"\n✨ ¡Proceso completado!")

if __name__ == "__main__":
    main()
