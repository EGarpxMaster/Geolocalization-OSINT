"""
Script para generar metadata.csv a partir de las imágenes existentes
"""
import csv
import json
from pathlib import Path
from datetime import datetime

# Configuración
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "data" / "mining" / "images"
CITIES_CSV = BASE_DIR / "data" / "cities_mx.csv"
OUTPUT_CSV = BASE_DIR / "data" / "mining" / "metadata.csv"

def load_cities():
    """Carga las ciudades y sus coordenadas"""
    cities_dict = {}
    with open(CITIES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalizar nombre de ciudad (quitar acentos, mayúsculas)
            city_key = row['name'].lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            cities_dict[city_key] = {
                'name': row['name'],
                'state': row['state'],
                'lat': float(row['lat']),
                'lon': float(row['lon'])
            }
    return cities_dict

def parse_filename(filename):
    """
    Extrae información del filename
    Formato: pexels_Ciudad_Estado_index_timestamp.jpg
    """
    parts = filename.replace('.jpg', '').split('_')
    
    if len(parts) < 3:
        return None
    
    # El formato puede variar, intentar varios patrones
    if parts[0] == 'pexels':
        # pexels_Acapulco_Guerrero_0_1764040142.jpg
        city = parts[1]
        state = parts[2] if len(parts) > 2 else ''
        index = parts[3] if len(parts) > 3 else '0'
        timestamp = parts[4] if len(parts) > 4 else ''
        
        return {
            'city': city,
            'state': state,
            'index': index,
            'timestamp': timestamp
        }
    
    return None

def main():
    print("🔄 Generando metadata.csv desde imágenes existentes...")
    
    # Cargar ciudades
    cities = load_cities()
    print(f"✅ {len(cities)} ciudades cargadas")
    
    # Listar todas las imágenes
    image_files = list(IMAGES_DIR.glob("*.jpg"))
    print(f"📁 {len(image_files)} imágenes encontradas")
    
    # Crear CSV
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Escribir encabezado
        writer.writerow([
            'filename', 'source', 'city', 'state', 'lat', 'lon',
            'downloaded_at', 'exists'
        ])
        
        processed = 0
        matched = 0
        
        for img_path in image_files:
            filename = img_path.name
            
            # Parsear filename
            info = parse_filename(filename)
            
            if not info:
                print(f"⚠️  No se pudo parsear: {filename}")
                continue
            
            # Buscar ciudad en diccionario
            city_key = info['city'].lower().replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u')
            
            city_data = cities.get(city_key)
            
            if not city_data:
                # Intentar con el nombre completo
                city_key_full = f"{info['city']}_{info['state']}".lower()
                city_data = cities.get(city_key_full)
            
            if city_data:
                matched += 1
                writer.writerow([
                    filename,
                    'pexels',
                    city_data['name'],
                    city_data['state'],
                    city_data['lat'],
                    city_data['lon'],
                    datetime.now().isoformat(),
                    True
                ])
            else:
                # Escribir con información parcial
                writer.writerow([
                    filename,
                    'pexels',
                    info['city'],
                    info['state'],
                    0.0,
                    0.0,
                    datetime.now().isoformat(),
                    True
                ])
            
            processed += 1
            
            if processed % 100 == 0:
                print(f"   Procesadas: {processed}/{len(image_files)}")
    
    print(f"\n✅ Metadata CSV generado: {OUTPUT_CSV}")
    print(f"📊 Procesadas: {processed} imágenes")
    print(f"🎯 Coincidencias con ciudades: {matched}")
    print(f"⚠️  Sin coincidencia: {processed - matched}")

if __name__ == '__main__':
    main()
