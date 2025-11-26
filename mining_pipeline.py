"""
PIPELINE DE MINERÍA DE IMÁGENES - GEOLOCALIZATION OSINT
========================================================
Sistema unificado para minería de imágenes desde fuentes abiertas.
Incluye: Wikimedia Commons, Wikipedia, Pexels (100% gratuito)

Uso:
    python mining_pipeline.py --mode all --images 20
    python mining_pipeline.py --mode state --state "Jalisco" --images 10
    python mining_pipeline.py --check-progress
"""

import os
import json
import hashlib
import requests
import csv
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from urllib.parse import quote
from bs4 import BeautifulSoup
import argparse
from dotenv import load_dotenv
from supabase import create_client, Client

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MINING_DIR = DATA_DIR / "mining"
IMAGES_DIR = MINING_DIR / "images"
METADATA_FILE = MINING_DIR / "metadata.json"
METADATA_CSV = MINING_DIR / "metadata.csv"
CITIES_CSV = DATA_DIR / "cities_mx.csv"

# Crear directorios
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Configurar Supabase
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    USE_SUPABASE = True
    BUCKET_NAME = "geolocalization-images"
else:
    USE_SUPABASE = False
    print("⚠️ Supabase no configurado. Solo se guardarán archivos locales.")

# APIs gratuitas
PEXELS_API_KEY = "uaPqaLWpEWGCmQvywCUm7zTeWZrZJQFKkLdtRnxU4WhEXUS4zer3heNK"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# ============================================================================
# UTILIDADES
# ============================================================================

def sanitize_name(text):
    """Sanitiza nombres de archivo: lowercase, sin acentos, underscores"""
    # Normalizar unicode (descomponer acentos)
    text = unicodedata.normalize('NFD', text)
    # Filtrar solo ASCII
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    # Reemplazos manuales adicionales
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u',
        'ñ': 'n', 'Ñ': 'n', ' ': '_', '-': '_'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Lowercase y limpiar caracteres no válidos
    text = ''.join(c if c.isalnum() or c == '_' else '_' for c in text)
    # Limpiar múltiples underscores
    while '__' in text:
        text = text.replace('__', '_')
    return text.lower().strip('_')

# ============================================================================
# FUNCIONES DE MINERÍA
# ============================================================================

def load_metadata():
    """Carga metadata existente"""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"images": [], "cities": {}}

def save_metadata(metadata):
    """Guarda metadata en JSON y CSV"""
    try:
        # Guardar JSON (original)
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Guardar CSV (nuevo - más fácil de usar)
        save_metadata_csv(metadata)
    except Exception as e:
        print(f"⚠️  Error guardando metadata: {e}")

def save_metadata_csv(metadata):
    """Guarda metadata en formato CSV"""
    try:
        with open(METADATA_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Encabezado
            writer.writerow([
                'filename', 'source', 'photo_id', 'city', 'state', 
                'lat', 'lon', 'url', 'title', 'photographer',
                'downloaded_at', 'size', 'hash'
            ])
            
            # Escribir cada imagen
            for img in metadata.get('images', []):
                # Obtener filename del campo correcto
                filename = img.get('filename', '')
                if not filename and 'local_path' in img:
                    filename = Path(img['local_path']).name
                
                writer.writerow([
                    filename,
                    img.get('source', ''),
                    img.get('photo_id', ''),
                    img.get('city', ''),  # Cambiado de city_target a city
                    img.get('state', ''),  # Cambiado de state_target a state
                    img.get('lat', 0.0),
                    img.get('lon', 0.0),
                    img.get('url', ''),
                    img.get('title', ''),
                    img.get('photographer', ''),
                    img.get('downloaded_at', ''),
                    img.get('size', 0),
                    img.get('hash', '')
                ])
        
        print(f"✅ Metadata CSV guardado: {METADATA_CSV}")
    except Exception as e:
        print(f"⚠️  Error guardando CSV: {e}")

def get_image_hash(image_data):
    """Genera hash único para deduplicación"""
    return hashlib.md5(image_data).hexdigest()

def download_image(url, save_path, metadata_entry):
    """Descarga imagen con validación y sube a Supabase Storage"""
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        
        # Validar que es una imagen
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type.lower():
            return False
        
        # Validar tamaño (min 50KB, max 10MB)
        size = len(response.content)
        if size < 50000 or size > 10000000:
            return False
        
        # Guardar imagen localmente (temporal)
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        metadata_entry['hash'] = get_image_hash(response.content)
        metadata_entry['size'] = size
        metadata_entry['downloaded_at'] = datetime.now().isoformat()
        
        # Subir a Supabase Storage si está configurado
        if USE_SUPABASE:
            try:
                filename = save_path.name
                
                # Subir archivo
                with open(save_path, 'rb') as f:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        filename,
                        f.read(),
                        file_options={"content-type": "image/jpeg", "upsert": "true"}
                    )
                
                # Obtener URL pública
                image_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
                metadata_entry['image_url'] = image_url
                
                # Guardar metadata en base de datos
                supabase.table('image_metadata').upsert({
                    'filename': filename,
                    'image_url': image_url,
                    'city': metadata_entry.get('city'),
                    'state': metadata_entry.get('state'),
                    'lat': metadata_entry.get('lat'),
                    'lon': metadata_entry.get('lon'),
                    'source': metadata_entry.get('source'),
                    'photo_id': metadata_entry.get('photo_id'),
                    'url': metadata_entry.get('url'),
                    'title': metadata_entry.get('title'),
                    'photographer': metadata_entry.get('photographer'),
                    'width': metadata_entry.get('width', 0),
                    'height': metadata_entry.get('height', 0),
                    'size': size,
                    'hash': metadata_entry['hash']
                }, on_conflict='filename').execute()
                
            except Exception as e:
                print(f"    ⚠️  Error subiendo a Supabase: {e}")
                # Continuar aunque falle Supabase, archivo local guardado
        
        return True
    except Exception as e:
        print(f"    ❌ Error descargando: {e}")
        return False

# ============================================================================
# FUENTES DE DATOS GRATUITAS
# ============================================================================

def search_wikimedia_commons(query, limit=10):
    """Busca imágenes en Wikimedia Commons (100% gratuito, sin límites)"""
    results = []
    try:
        url = "https://commons.wikimedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'search',
            'gsrnamespace': 6,  # File namespace
            'gsrsearch': f'{query} Mexico',
            'gsrlimit': limit,
            'prop': 'imageinfo',
            'iiprop': 'url|size|extmetadata',
            'iiurlwidth': 1024
        }
        
        response = requests.get(url, params=params, timeout=10, headers=HEADERS)
        response.raise_for_status()
        
        if not response.text.strip():
            return results
        
        data = response.json()
        
        if 'query' in data and 'pages' in data['query']:
            for page in data['query']['pages'].values():
                if 'imageinfo' in page:
                    info = page['imageinfo'][0]
                    results.append({
                        'url': info.get('url'),
                        'thumbnail': info.get('thumburl', info.get('url')),
                        'source': 'wikimedia',
                        'title': page.get('title', ''),
                        'width': info.get('width', 0),
                        'height': info.get('height', 0)
                    })
    except Exception as e:
        print(f"    ⚠️  Wikimedia error: {e}")
    
    return results

def search_wikipedia_images(query, limit=10):
    """Busca imágenes en artículos de Wikipedia"""
    results = []
    try:
        # Buscar artículos relacionados
        search_url = f"https://es.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srsearch': f'{query} México',
            'srlimit': 3
        }
        
        response = requests.get(search_url, params=params, timeout=10, headers=HEADERS)
        response.raise_for_status()
        
        if not response.text.strip():
            return results
        
        data = response.json()
        
        if 'query' in data and 'search' in data['query']:
            for article in data['query']['search'][:2]:
                title = article['title']
                
                # Obtener imágenes del artículo
                img_params = {
                    'action': 'query',
                    'format': 'json',
                    'titles': title,
                    'prop': 'images',
                    'imlimit': limit
                }
                
                img_response = requests.get(search_url, params=img_params, timeout=10)
                img_data = img_response.json()
                
                if 'query' in img_data and 'pages' in img_data['query']:
                    for page in img_data['query']['pages'].values():
                        if 'images' in page:
                            for img in page['images'][:limit]:
                                img_title = img['title']
                                
                                # Obtener URL de la imagen
                                url_params = {
                                    'action': 'query',
                                    'format': 'json',
                                    'titles': img_title,
                                    'prop': 'imageinfo',
                                    'iiprop': 'url|size',
                                    'iiurlwidth': 1024
                                }
                                
                                url_response = requests.get(search_url, params=url_params, timeout=10)
                                url_data = url_response.json()
                                
                                if 'query' in url_data and 'pages' in url_data['query']:
                                    for img_page in url_data['query']['pages'].values():
                                        if 'imageinfo' in img_page:
                                            info = img_page['imageinfo'][0]
                                            results.append({
                                                'url': info.get('url'),
                                                'thumbnail': info.get('thumburl', info.get('url')),
                                                'source': 'wikipedia',
                                                'title': img_title,
                                                'width': info.get('width', 0),
                                                'height': info.get('height', 0)
                                            })
    except Exception as e:
        print(f"    ⚠️  Wikipedia error: {e}")
    
    return results[:limit]

def search_pexels(query, limit=10):
    """Busca imágenes en Pexels (API gratuita con registro)"""
    results = []
    try:
        url = "https://api.pexels.com/v1/search"
        headers = {'Authorization': PEXELS_API_KEY}
        params = {
            'query': f'{query} Mexico',
            'per_page': min(limit, 80),
            'orientation': 'landscape'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = response.json()
            for photo in data.get('photos', []):
                results.append({
                    'url': photo['src']['large2x'],
                    'thumbnail': photo['src']['medium'],
                    'source': 'pexels',
                    'title': f"Pexels {photo['id']}",
                    'photographer': photo.get('photographer', ''),
                    'width': photo.get('width', 0),
                    'height': photo.get('height', 0)
                })
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️  Pexels error: {e}")
    except Exception as e:
        print(f"    ⚠️  Pexels error inesperado: {e}")
    
    return results

# ============================================================================
# PIPELINE DE MINERÍA
# ============================================================================

def mine_city(city_name, state, lat, lon, images_per_source=20):
    """Mina imágenes de una ciudad desde todas las fuentes"""
    print(f"\n[{city_name}, {state}]")
    
    metadata = load_metadata()
    
    # Asegurar que exista la estructura base
    if 'cities' not in metadata:
        metadata['cities'] = {}
    if 'images' not in metadata:
        metadata['images'] = []
    
    # Sanitizar nombres para el formato de archivo
    city_sanitized = sanitize_name(city_name)
    state_sanitized = sanitize_name(state)
    
    city_key = f"{city_name}_{state}"
    
    if city_key not in metadata['cities']:
        metadata['cities'][city_key] = {
            'name': city_name,
            'state': state,
            'lat': lat,
            'lon': lon,
            'images': []
        }
    
    existing_hashes = {img.get('hash') for img in metadata['images']}
    new_images = 0
    
    # Buscar en cada fuente
    sources = [
        ('Wikimedia Commons', lambda: search_wikimedia_commons(city_name, images_per_source)),
        ('Wikipedia', lambda: search_wikipedia_images(city_name, images_per_source)),
        ('Pexels', lambda: search_pexels(city_name, images_per_source))
    ]
    
    for source_name, search_func in sources:
        print(f"  → Buscando en {source_name}...")
        results = search_func()
        
        for idx, img in enumerate(results):
            if new_images >= images_per_source:
                break
            
            # Crear nombre de archivo único con nombres sanitizados
            source_prefix = img['source']
            filename = f"{source_prefix}_{city_sanitized}_{state_sanitized}_{idx}_{int(time.time())}.jpg"
            save_path = IMAGES_DIR / filename
            
            print(f"    ⬇️  Descargando {source_prefix}/{img['title'][:30]}...", end=' ')
            
            metadata_entry = {
                'filename': filename,
                'city': city_name,
                'state': state,
                'lat': lat,
                'lon': lon,
                'source': img['source'],
                'photo_id': str(img.get('id', '')),
                'url': img['url'],
                'title': img.get('title', ''),
                'photographer': img.get('photographer', ''),
                'width': img.get('width', 0),
                'height': img.get('height', 0)
            }
            
            if download_image(img['url'], save_path, metadata_entry):
                # Verificar duplicados
                if metadata_entry['hash'] in existing_hashes:
                    print("⚠️  Duplicado")
                    save_path.unlink()
                    continue
                
                metadata['images'].append(metadata_entry)
                metadata['cities'][city_key]['images'].append(filename)
                existing_hashes.add(metadata_entry['hash'])
                new_images += 1
                print("✅")
            else:
                print("❌")
            
            time.sleep(0.5)  # Rate limiting
    
    save_metadata(metadata)
    print(f"  ✅ Total descargadas: {new_images} imágenes")
    return new_images

# ============================================================================
# FUNCIONES DE GESTIÓN
# ============================================================================

def load_cities():
    """Carga todas las ciudades del CSV"""
    cities = []
    with open(CITIES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cities.append({
                'name': row['name'],
                'state': row['state'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'tags': row.get('tags', '')
            })
    return cities

def get_cities_by_state(state_name=None):
    """Obtiene ciudades agrupadas por estado"""
    cities = load_cities()
    if state_name:
        return [c for c in cities if c['state'].lower() == state_name.lower()]
    
    states_dict = defaultdict(list)
    for city in cities:
        states_dict[city['state']].append(city)
    return dict(states_dict)

def mine_all_cities(images_per_city=20):
    """Mina todas las ciudades del CSV"""
    cities = load_cities()
    states_dict = get_cities_by_state()
    
    print("="*70)
    print("🗺️  MINERÍA COMPLETA - TODAS LAS CIUDADES DE MÉXICO")
    print("="*70)
    print(f"\n✅ {len(cities)} ciudades identificadas")
    print(f"🗺️  {len(states_dict)} estados cubiertos\n")
    
    # Mostrar distribución
    print("📊 DISTRIBUCIÓN POR ESTADO:")
    for state, cities_list in sorted(states_dict.items()):
        total_imgs = len(cities_list) * images_per_city
        print(f"  • {state}: {len(cities_list)} ciudad(es) × {images_per_city} = {total_imgs} imágenes")
    
    total_images = len(cities) * images_per_city
    print(f"\n🎯 TOTAL: {len(cities)} ciudades × {images_per_city} = {total_images} imágenes")
    print(f"⏱️  Tiempo estimado: {total_images * 0.045:.0f}-{total_images * 0.06:.0f} minutos\n")
    
    print("Iniciando en 3 segundos...")
    time.sleep(3)
    
    start_time = time.time()
    total_downloaded = 0
    
    for idx, city in enumerate(cities, 1):
        print(f"\n{'='*70}")
        print(f"[{idx}/{len(cities)}] {city['name']}, {city['state']}")
        print(f"{'='*70}")
        
        downloaded = mine_city(
            city['name'],
            city['state'],
            city['lat'],
            city['lon'],
            images_per_city
        )
        total_downloaded += downloaded
        
        # Auto-save cada 5 ciudades
        if idx % 5 == 0:
            print(f"\n💾 Checkpoint: {idx}/{len(cities)} ciudades completadas")
            print(f"📊 Progreso: {total_downloaded} imágenes descargadas")
    
    elapsed = time.time() - start_time
    
    # Estadísticas finales
    print("\n" + "="*70)
    print("✅ MINERÍA COMPLETADA")
    print("="*70)
    print(f"⏱️  Tiempo total: {elapsed/60:.1f} minutos")
    print(f"📊 Imágenes descargadas: {total_downloaded}")
    print(f"🗺️  Ciudades procesadas: {len(cities)}")
    print(f"📁 Ubicación: {IMAGES_DIR}")
    print("\n💡 Siguiente paso: python training_pipeline.py --annotate")

def check_progress():
    """Verifica el progreso de la minería"""
    metadata = load_metadata()
    
    print("\n" + "="*70)
    print("📊 PROGRESO DE MINERÍA")
    print("="*70)
    
    total_images = len(metadata['images'])
    cities_count = len(metadata['cities'])
    
    if total_images == 0:
        print("\n⚠️  No se han descargado imágenes aún")
        print("💡 Ejecuta: python mining_pipeline.py --mode all")
        return
    
    print(f"\n✅ Total de imágenes: {total_images}")
    print(f"🏙️  Ciudades cubiertas: {cities_count}")
    
    # Contar por estado
    states = defaultdict(int)
    for city_data in metadata['cities'].values():
        states[city_data['state']] += len(city_data['images'])
    
    print(f"🗺️  Estados cubiertos: {len(states)}/32")
    
    # Top ciudades
    print("\n📈 Top 10 ciudades:")
    sorted_cities = sorted(
        metadata['cities'].items(),
        key=lambda x: len(x[1]['images']),
        reverse=True
    )
    for idx, (city_key, city_data) in enumerate(sorted_cities[:10], 1):
        print(f"  {idx}. {city_data['name']}, {city_data['state']}: {len(city_data['images'])} imágenes")
    
    # Por fuente
    sources = defaultdict(int)
    for img in metadata['images']:
        sources[img['source']] += 1
    
    print("\n🌐 Por fuente:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images) * 100
        print(f"  • {source}: {count} ({percentage:.1f}%)")
    
    # Estimación de completitud
    all_cities = load_cities()
    progress = (cities_count / len(all_cities)) * 100
    print(f"\n🎯 Progreso general: {progress:.1f}% ({cities_count}/{len(all_cities)} ciudades)")
    
    print("\n" + "="*70)

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline de minería de imágenes para OSINT geolocalización'
    )
    parser.add_argument(
        '--mode',
        choices=['all', 'state', 'city', 'progress'],
        default='all',
        help='Modo de operación'
    )
    parser.add_argument('--state', help='Nombre del estado a minar')
    parser.add_argument('--city', help='Nombre de la ciudad a minar')
    parser.add_argument('--images', type=int, default=20, help='Imágenes por ciudad')
    parser.add_argument('--check-progress', action='store_true', help='Ver progreso')
    
    args = parser.parse_args()
    
    if args.check_progress or args.mode == 'progress':
        check_progress()
        return
    
    if args.mode == 'all':
        mine_all_cities(args.images)
    
    elif args.mode == 'state':
        if not args.state:
            print("❌ Debes especificar --state")
            return
        
        cities = get_cities_by_state(args.state)
        if not cities:
            print(f"❌ Estado '{args.state}' no encontrado")
            return
        
        print(f"Minando {len(cities)} ciudades de {args.state}...")
        for city in cities:
            mine_city(city['name'], city['state'], city['lat'], city['lon'], args.images)
    
    elif args.mode == 'city':
        if not args.city:
            print("❌ Debes especificar --city")
            return
        
        cities = load_cities()
        city_data = next((c for c in cities if c['name'].lower() == args.city.lower()), None)
        
        if not city_data:
            print(f"❌ Ciudad '{args.city}' no encontrada")
            return
        
        mine_city(city_data['name'], city_data['state'], city_data['lat'], city_data['lon'], args.images)

if __name__ == '__main__':
    main()
