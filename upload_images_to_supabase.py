"""
SUBIR IMÁGENES A SUPABASE STORAGE
==================================
Sube todas las imágenes locales a Supabase Storage y actualiza las URLs en la base de datos

Requisitos:
    1. Crear bucket en Supabase:
       - Ve a Storage en tu proyecto Supabase
       - Crea un bucket público llamado 'geolocalization-images'
       - Marca como público para acceso directo
    
    2. Ejecutar este script:
       python upload_images_to_supabase.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm
import mimetypes

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
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "data" / "mining" / "images"
BUCKET_NAME = "geolocalization-images"

def create_bucket_if_not_exists():
    """Crea el bucket si no existe"""
    try:
        # Intentar obtener el bucket
        buckets = supabase.storage.list_buckets()
        bucket_exists = any(b['name'] == BUCKET_NAME for b in buckets)
        
        if not bucket_exists:
            print(f"📦 Creando bucket '{BUCKET_NAME}'...")
            supabase.storage.create_bucket(
                BUCKET_NAME,
                options={"public": True}  # Bucket público
            )
            print(f"✅ Bucket '{BUCKET_NAME}' creado")
        else:
            print(f"✅ Bucket '{BUCKET_NAME}' ya existe")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Error con bucket: {e}")
        print(f"💡 Crea manualmente el bucket en Supabase Storage:")
        print(f"   1. Ve a Storage en tu proyecto")
        print(f"   2. Crea bucket '{BUCKET_NAME}'")
        print(f"   3. Márcalo como público")
        return False

def upload_image(file_path: Path) -> str:
    """
    Sube una imagen a Supabase Storage
    
    Returns:
        URL pública de la imagen
    """
    filename = file_path.name
    
    try:
        # Leer archivo
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Detectar tipo MIME
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = 'image/jpeg'
        
        # Subir archivo
        result = supabase.storage.from_(BUCKET_NAME).upload(
            filename,
            file_data,
            file_options={"content-type": mime_type, "upsert": "true"}
        )
        
        # Obtener URL pública
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        return public_url
        
    except KeyboardInterrupt:
        raise
    except Exception as e:
        # Solo mostrar primeros caracteres del error
        error_msg = str(e)[:100]
        return None

def update_image_url(filename: str, url: str) -> bool:
    """Actualiza la URL de la imagen en la base de datos"""
    try:
        result = supabase.table('image_metadata').update(
            {'image_url': url}
        ).eq('filename', filename).execute()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error actualizando URL de {filename}: {e}")
        return False

def main():
    print("="*60)
    print("SUBIR IMÁGENES A SUPABASE STORAGE")
    print("="*60)
    
    # Verificar directorio de imágenes
    if not IMAGES_DIR.exists():
        print(f"❌ Directorio no encontrado: {IMAGES_DIR}")
        return
    
    # Listar imágenes
    image_files = list(IMAGES_DIR.glob("*.jpg")) + \
                  list(IMAGES_DIR.glob("*.jpeg")) + \
                  list(IMAGES_DIR.glob("*.png")) + \
                  list(IMAGES_DIR.glob("*.avif"))
    
    if not image_files:
        print(f"❌ No se encontraron imágenes en {IMAGES_DIR}")
        return
    
    print(f"\n📁 Encontradas: {len(image_files)} imágenes")
    
    # Crear bucket
    if not create_bucket_if_not_exists():
        print("\n⚠️  Continúa solo si el bucket ya existe...")
        response = input("¿Continuar? (s/n): ")
        if response.lower() != 's':
            return
    
    # Confirmar subida
    print(f"\n⚠️  ADVERTENCIA: Esto subirá {len(image_files)} imágenes a Supabase")
    print(f"   Esto puede tomar varios minutos y consumir ancho de banda")
    
    response = input("\n¿Continuar con la subida? (s/n): ")
    if response.lower() != 's':
        print("❌ Cancelado por el usuario")
        return
    
    # Subir imágenes
    print(f"\n📤 Subiendo imágenes a Supabase Storage...")
    
    uploaded = 0
    failed = 0
    skipped = 0
    
    with tqdm(total=len(image_files), desc="Subiendo") as pbar:
        for img_file in image_files:
            filename = img_file.name
            
            # Verificar si ya existe en la BD con URL
            try:
                result = supabase.table('image_metadata').select(
                    'image_url'
                ).eq('filename', filename).execute()
                
                if result.data and result.data[0].get('image_url'):
                    # Ya tiene URL, omitir
                    skipped += 1
                    pbar.update(1)
                    continue
                    
            except:
                pass
            
            # Subir imagen
            url = upload_image(img_file)
            
            if url:
                # Actualizar BD
                if update_image_url(filename, url):
                    uploaded += 1
                else:
                    failed += 1
            else:
                failed += 1
            
            pbar.update(1)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE SUBIDA")
    print("="*60)
    print(f"✅ Subidas exitosas: {uploaded}")
    print(f"⏭️  Omitidas (ya existían): {skipped}")
    print(f"❌ Fallidas: {failed}")
    print(f"📊 Total procesadas: {len(image_files)}")
    
    if uploaded > 0:
        print(f"\n🎉 ¡Imágenes disponibles en Supabase Storage!")
        print(f"   URL del bucket: {SUPABASE_URL}/storage/v1/object/public/{BUCKET_NAME}/")
    
    # Verificar en BD
    try:
        result = supabase.table('image_metadata').select(
            'filename, image_url'
        ).not_.is_('image_url', 'null').execute()
        
        print(f"\n📈 Imágenes con URL en BD: {len(result.data)}/1020")
        
    except Exception as e:
        print(f"\n⚠️  No se pudo verificar BD: {e}")

if __name__ == "__main__":
    main()
