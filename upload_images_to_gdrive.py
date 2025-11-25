"""
SUBIR IMÁGENES A GOOGLE DRIVE
==============================
Sube todas las imágenes a Google Drive y genera URLs públicas

Requisitos:
    pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

Setup:
    1. Ve a https://console.cloud.google.com/
    2. Crea un proyecto nuevo o usa uno existente
    3. Habilita "Google Drive API"
    4. Ve a "Credentials" → "Create Credentials" → "OAuth client ID"
    5. Tipo: "Desktop app"
    6. Descarga el JSON como "credentials.json" en este directorio
    7. Ejecuta este script
"""

import os
import pickle
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client

# Cargar variables de entorno para Supabase
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Google Drive API scope
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Configuración
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "data" / "mining" / "images"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.pickle"

# ID de la carpeta en Google Drive (desde la URL)
# Extraído de: https://drive.google.com/drive/folders/1ptr7mBUcHWoBVtf5UD09PztPQGwaxMmx
DRIVE_FOLDER_ID = "1ptr7mBUcHWoBVtf5UD09PztPQGwaxMmx"

def authenticate_google_drive():
    """Autentica con Google Drive"""
    creds = None
    
    # Token guardado de sesiones previas
    if TOKEN_FILE.exists():
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    
    # Si no hay credenciales válidas, hacer login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print("❌ ERROR: No se encontró credentials.json")
                print("\n📝 Sigue estos pasos:")
                print("1. Ve a https://console.cloud.google.com/")
                print("2. Crea un proyecto o selecciona uno")
                print("3. Habilita 'Google Drive API'")
                print("4. Ve a 'Credentials' → 'Create Credentials' → 'OAuth client ID'")
                print("5. Tipo: 'Desktop app'")
                print("6. Descarga el JSON como 'credentials.json'")
                print("7. Colócalo en este directorio")
                return None
            
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Guardar credenciales para próximas ejecuciones
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)

def verify_drive_folder(service, folder_id):
    """Verifica que la carpeta existe y es accesible"""
    try:
        folder = service.files().get(
            fileId=folder_id,
            fields='id, name, mimeType'
        ).execute()
        
        print(f"✅ Carpeta encontrada: '{folder.get('name')}' (ID: {folder_id})")
        
        # Hacer carpeta pública si no lo está
        try:
            service.permissions().create(
                fileId=folder_id,
                body={'type': 'anyone', 'role': 'reader'}
            ).execute()
            print(f"✅ Carpeta configurada como pública")
        except Exception as e:
            # Probablemente ya es pública
            print(f"✅ Carpeta ya es pública")
        
        return folder_id
        
    except Exception as e:
        print(f"❌ Error verificando carpeta: {e}")
        print(f"⚠️  Verifica que la carpeta existe y tienes permisos")
        return None

def upload_image_to_drive(service, file_path: Path, folder_id: str) -> str:
    """Sube una imagen a Google Drive y retorna URL pública"""
    try:
        filename = file_path.name
        
        # Metadata del archivo
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        # Tipo MIME
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.avif': 'image/avif'
        }
        mime_type = mime_types.get(file_path.suffix.lower(), 'image/jpeg')
        
        # Subir archivo
        media = MediaFileUpload(
            str(file_path),
            mimetype=mime_type,
            resumable=True
        )
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink, webContentLink'
        ).execute()
        
        file_id = file.get('id')
        
        # Hacer archivo público
        service.permissions().create(
            fileId=file_id,
            body={'type': 'anyone', 'role': 'reader'}
        ).execute()
        
        # URL directa para visualización
        direct_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        
        return direct_url
        
    except Exception as e:
        print(f"\n❌ Error subiendo {filename}: {e}")
        return None

def update_supabase_url(filename: str, url: str) -> bool:
    """Actualiza la URL en Supabase"""
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = supabase.table('image_metadata').update(
            {'image_url': url}
        ).eq('filename', filename).execute()
        return True
    except Exception as e:
        return False

def main():
    print("="*60)
    print("SUBIR IMÁGENES A GOOGLE DRIVE")
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
    
    # Autenticar con Google Drive
    print("\n🔐 Autenticando con Google Drive...")
    service = authenticate_google_drive()
    
    if not service:
        return
    
    print("✅ Autenticación exitosa")
    
    # Crear carpeta
    folder_id = create_drive_folder(service)
    
    # Confirmar subida
    print(f"\n⚠️  ADVERTENCIA: Esto subirá {len(image_files)} imágenes a Google Drive")
    print(f"   Carpeta: {DRIVE_FOLDER_NAME}")
    print(f"   Esto puede tomar 30-60 minutos")
    
    print("✅ Autenticación exitosa")
    
    # Verificar carpeta existente
    print(f"\n📁 Verificando carpeta en Google Drive...")
    folder_id = verify_drive_folder(service, DRIVE_FOLDER_ID)
    
    if not folder_id:
        return
    
    # Confirmar subida
    print(f"\n⚠️  ADVERTENCIA: Esto subirá {len(image_files)} imágenes a Google Drive")
    print(f"   Carpeta ID: {DRIVE_FOLDER_ID}")
    print(f"   Esto puede tomar 30-60 minutos")
    failed = 0
    updated_db = 0
    
    with tqdm(total=len(image_files), desc="Subiendo") as pbar:
        for img_file in image_files:
            filename = img_file.name
            
            # Subir a Drive
            url = upload_image_to_drive(service, img_file, folder_id)
            
            if url:
                uploaded += 1
                
                # Actualizar Supabase si está configurado
                if SUPABASE_URL and SUPABASE_KEY:
                    if update_supabase_url(filename, url):
                        updated_db += 1
            else:
                failed += 1
            
            pbar.update(1)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE SUBIDA")
    print("="*60)
    print(f"✅ Subidas exitosas: {uploaded}")
    print(f"❌ Fallidas: {failed}")
    print(f"📊 Total procesadas: {len(image_files)}")
    
    if SUPABASE_URL and SUPABASE_KEY:
        print(f"💾 URLs actualizadas en Supabase: {updated_db}")
    
    if uploaded > 0:
        print(f"\n🎉 ¡Imágenes disponibles en Google Drive!")
        print(f"   Carpeta: https://drive.google.com/drive/folders/{folder_id}")

if __name__ == "__main__":
    main()
