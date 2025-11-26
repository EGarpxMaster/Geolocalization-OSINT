"""
RENOMBRAR IMÁGENES SIN ACENTOS
===============================
Renombra archivos removiendo acentos y caracteres especiales
para compatibilidad con Supabase Storage y URLs
"""

import os
from pathlib import Path
import unicodedata

BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "data" / "mining" / "images"
METADATA_CSV = BASE_DIR / "data" / "mining" / "metadata.csv"
ANNOTATIONS_CSV = BASE_DIR / "data" / "mining" / "annotations.csv"

def remove_accents(text):
    """Remueve acentos de un string"""
    # Normalizar a NFD (descomponer caracteres con acentos)
    nfd = unicodedata.normalize('NFD', text)
    # Filtrar solo caracteres ASCII
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')

def sanitize_filename(filename):
    """Sanitiza nombre de archivo para URLs"""
    name, ext = os.path.splitext(filename)
    
    # Remover acentos
    name = remove_accents(name)
    
    # Reemplazar espacios y caracteres especiales por guiones bajos
    name = name.replace(' ', '_')
    name = name.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
    name = name.replace('Á', 'A').replace('É', 'E').replace('Í', 'I').replace('Ó', 'O').replace('Ú', 'U')
    name = name.replace('ñ', 'n').replace('Ñ', 'N')
    
    # Remover TODOS los caracteres que no sean alfanuméricos, guiones o guiones bajos
    name = ''.join(c if c.isascii() and (c.isalnum() or c in ('_', '-')) else '_' for c in name)
    
    # Limpiar guiones bajos múltiples
    while '__' in name:
        name = name.replace('__', '_')
    
    # Remover guiones al inicio/final
    name = name.strip('_-')
    
    # Convertir a lowercase para evitar problemas
    name = name.lower()
    
    return name + ext.lower()

def rename_files():
    """Renombra todos los archivos de imágenes"""
    if not IMAGES_DIR.exists():
        print(f"❌ Directorio no encontrado: {IMAGES_DIR}")
        return
    
    image_files = list(IMAGES_DIR.glob("*.jpg")) + \
                  list(IMAGES_DIR.glob("*.jpeg")) + \
                  list(IMAGES_DIR.glob("*.png")) + \
                  list(IMAGES_DIR.glob("*.avif"))
    
    rename_map = {}
    
    print(f"📁 Encontrados: {len(image_files)} archivos")
    print("\nRenombrados:")
    
    for img_file in image_files:
        old_name = img_file.name
        new_name = sanitize_filename(old_name)
        
        if old_name != new_name:
            new_path = img_file.parent / new_name
            
            # Verificar que el nuevo nombre no existe
            if new_path.exists():
                print(f"⚠️  Ya existe: {new_name}")
                continue
            
            # Renombrar archivo
            img_file.rename(new_path)
            rename_map[old_name] = new_name
            print(f"  {old_name} → {new_name}")
    
    print(f"\n✅ Renombrados: {len(rename_map)} archivos")
    
    return rename_map

def update_csv_file(csv_path, rename_map):
    """Actualiza nombres en archivo CSV"""
    if not csv_path.exists():
        print(f"⚠️  No encontrado: {csv_path}")
        return
    
    import csv
    
    # Leer CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    
    # Actualizar filenames
    updated = 0
    for row in rows:
        if 'filename' in row and row['filename'] in rename_map:
            row['filename'] = rename_map[row['filename']]
            updated += 1
    
    # Escribir CSV actualizado
    if updated > 0:
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"✅ Actualizado {csv_path.name}: {updated} registros")

def main():
    print("="*60)
    print("RENOMBRAR IMÁGENES SIN ACENTOS")
    print("="*60)
    
    # Renombrar archivos
    rename_map = rename_files()
    
    if not rename_map:
        print("\n✅ No hay archivos para renombrar")
        return
    
    # Actualizar CSVs
    print("\n📝 Actualizando archivos CSV...")
    update_csv_file(METADATA_CSV, rename_map)
    update_csv_file(ANNOTATIONS_CSV, rename_map)
    
    print("\n✅ ¡Proceso completado!")
    print(f"\n📊 Resumen:")
    print(f"   Archivos renombrados: {len(rename_map)}")
    print(f"\n💡 Ahora puedes subir las imágenes sin problemas de acentos")

if __name__ == "__main__":
    main()
