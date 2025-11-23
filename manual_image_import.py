# manual_image_import.py
# Herramienta para importar imágenes descargadas manualmente
# Útil cuando quieres agregar tus propias fotos al dataset de entrenamiento

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
from PIL import Image
import hashlib

IMAGES_DIR = "data/mining/images"
METADATA_FILE = "data/mining/metadata.json"
MANUAL_IMPORT_DIR = "data/manual_imports"


class ManualImporter:
    def __init__(self):
        self.images_dir = Path(IMAGES_DIR)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.manual_dir = Path(MANUAL_IMPORT_DIR)
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = Path(METADATA_FILE)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"images": [], "last_update": None}
    
    def save_metadata(self):
        """Guarda metadatos"""
        self.metadata["last_update"] = datetime.now().isoformat()
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def import_image(self, image_path: str, city: str, state: str, lat: float = None, lon: float = None, tags: str = ""):
        """Importa una imagen manualmente al dataset"""
        img_path = Path(image_path)
        
        if not img_path.exists():
            print(f"❌ Imagen no encontrada: {image_path}")
            return False
        
        # Validar que sea una imagen
        try:
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"❌ Archivo no es una imagen válida: {e}")
            return False
        
        # Generar hash único
        with open(img_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:12]
        
        # Crear nombre de archivo
        city_slug = city.lower().replace(" ", "_")
        ext = img_path.suffix.lower()
        new_filename = f"manual_{city_slug}_{file_hash}{ext}"
        dest_path = self.images_dir / new_filename
        
        # Copiar imagen
        shutil.copy2(img_path, dest_path)
        
        # Agregar metadatos
        img_data = {
            "source": "manual_import",
            "url": f"file:///{img_path.as_posix()}",
            "photo_id": f"manual_{file_hash}",
            "title": img_path.stem,
            "tags": tags,
            "lat": lat if lat is not None else 0.0,
            "lon": lon if lon is not None else 0.0,
            "city_target": city,
            "state_target": state,
            "local_path": str(dest_path),
            "downloaded_at": datetime.now().isoformat(),
            "annotated": False,
            "annotation": None,
        }
        
        self.metadata["images"].append(img_data)
        self.save_metadata()
        
        print(f"✅ Imagen importada: {new_filename}")
        return True
    
    def import_folder(self, folder_path: str, city: str, state: str, lat: float = None, lon: float = None):
        """Importa todas las imágenes de una carpeta"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"❌ Carpeta no encontrada: {folder_path}")
            return
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
        image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"❌ No se encontraron imágenes en {folder_path}")
            return
        
        print(f"\n📁 Importando {len(image_files)} imágenes de {folder.name}...")
        
        imported = 0
        for img_file in image_files:
            if self.import_image(str(img_file), city, state, lat, lon):
                imported += 1
        
        print(f"\n✅ Importadas {imported}/{len(image_files)} imágenes")


def main():
    import argparse
    from csv import DictReader
    
    parser = argparse.ArgumentParser(description="Importar imágenes manualmente al dataset")
    parser.add_argument("--file", type=str, help="Ruta de la imagen a importar")
    parser.add_argument("--folder", type=str, help="Carpeta con imágenes a importar")
    parser.add_argument("--city", type=str, required=True, help="Nombre de la ciudad")
    parser.add_argument("--state", type=str, help="Estado (opcional, se infiere del CSV)")
    parser.add_argument("--lat", type=float, help="Latitud (opcional)")
    parser.add_argument("--lon", type=float, help="Longitud (opcional)")
    parser.add_argument("--tags", type=str, default="", help="Tags separados por comas")
    
    args = parser.parse_args()
    
    # Inferir estado y coordenadas del CSV si no se proporcionan
    if not args.state or args.lat is None or args.lon is None:
        cities_csv = Path("data/cities_mx.csv")
        if cities_csv.exists():
            with open(cities_csv, 'r', encoding='utf-8') as f:
                reader = DictReader(f)
                for row in reader:
                    if row["name"].lower() == args.city.lower():
                        args.state = args.state or row["state"]
                        args.lat = args.lat if args.lat is not None else float(row["lat"])
                        args.lon = args.lon if args.lon is not None else float(row["lon"])
                        break
    
    if not args.state:
        print(f"❌ No se pudo inferir el estado para '{args.city}'. Especifícalo con --state")
        return
    
    importer = ManualImporter()
    
    if args.file:
        importer.import_image(args.file, args.city, args.state, args.lat, args.lon, args.tags)
    elif args.folder:
        importer.import_folder(args.folder, args.city, args.state, args.lat, args.lon)
    else:
        print("❌ Debes especificar --file o --folder")
        print("\nEjemplos:")
        print("  # Importar una imagen")
        print('  python manual_image_import.py --file "foto.jpg" --city "Puebla"')
        print("\n  # Importar carpeta completa")
        print('  python manual_image_import.py --folder "mis_fotos/puebla" --city "Puebla"')


if __name__ == "__main__":
    main()
