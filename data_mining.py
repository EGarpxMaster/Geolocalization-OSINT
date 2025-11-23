# data_mining.py
# Script OPEN SOURCE para minería de imágenes geolocalizadas
# Usa métodos gratuitos: Wikimedia Commons, Google Street View estático, Pexels
# 100% gratuito, sin APIs de pago ni suscripciones

import os
import csv
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import hashlib
import urllib.parse
import random
from bs4 import BeautifulSoup

# Configuración
IMAGES_DIR = "data/mining/images"
METADATA_FILE = "data/mining/metadata.json"
CITIES_CSV = "data/cities_mx.csv"

# User agent para web scraping ético
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

class ImageMiner:
    def __init__(self):
        self.images_dir = Path(IMAGES_DIR)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = self.load_metadata()
        self.cities = self.load_cities()
        
    def load_metadata(self) -> Dict:
        """Carga metadatos existentes"""
        metadata_path = Path(METADATA_FILE)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"images": [], "last_update": None}
    
    def save_metadata(self):
        """Guarda metadatos"""
        self.metadata["last_update"] = datetime.now().isoformat()
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    def load_cities(self) -> List[Dict]:
        """Carga lista de ciudades objetivo"""
        cities = []
        with open(CITIES_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                cities.append({
                    "name": row["name"],
                    "state": row["state"],
                    "lat": float(row["lat"]),
                    "lon": float(row["lon"]),
                    "tags": row.get("tags", "").split("|")
                })
        return cities
    
    def search_wikimedia_commons(self, city: Dict, max_images: int = 20) -> List[Dict]:
        """Busca imágenes en Wikimedia Commons (100% gratis, open source)"""
        try:
            # API de Wikimedia Commons - sin autenticación
            query = f"{city['name']} {city['state']} Mexico"
            
            url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": max_images,
                "prop": "imageinfo",
                "iiprop": "url|extmetadata",
                "iiurlwidth": 800,
            }
            
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            pages = data.get("query", {}).get("pages", {})
            
            for page_id, page in pages.items():
                if "imageinfo" not in page:
                    continue
                
                img_info = page["imageinfo"][0]
                img_url = img_info.get("thumburl") or img_info.get("url")
                
                if not img_url or not img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                # Extraer metadatos
                metadata = img_info.get("extmetadata", {})
                title = page.get("title", "").replace("File:", "")
                description = metadata.get("ImageDescription", {}).get("value", "")
                
                results.append({
                    "source": "wikimedia",
                    "url": img_url,
                    "photo_id": f"wiki_{page_id}",
                    "title": title,
                    "description": description,
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "city_target": city["name"],
                    "state_target": city["state"],
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Error en Wikimedia: {e}")
            return []
    
    def search_pexels_free(self, city: Dict, max_images: int = 20) -> List[Dict]:
        """Busca en Pexels (API gratuita sin límites estrictos)"""
        try:
            # Pexels API es gratuita, solo necesitas crear cuenta (sin pago)
            # API key: https://www.pexels.com/api/
            api_key = os.getenv("PEXELS_API_KEY", "uaPqaLWpEWGCmQvywCUm7zTeWZrZJQFKkLdtRnxU4WhEXUS4zer3heNK")
            
            if not api_key:
                print("  ℹ️  PEXELS_API_KEY no configurada (opcional pero gratis)")
                return []
            
            query = f"{city['name']} Mexico city"
            url = "https://api.pexels.com/v1/search"
            
            headers = {
                "Authorization": api_key,
                "User-Agent": USER_AGENT
            }
            
            params = {
                "query": query,
                "per_page": min(max_images, 80),
                "orientation": "landscape",
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for photo in data.get("photos", []):
                results.append({
                    "source": "pexels",
                    "url": photo["src"]["large"],
                    "photo_id": f"pexels_{photo['id']}",
                    "title": photo.get("alt", ""),
                    "photographer": photo.get("photographer", ""),
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "city_target": city["name"],
                    "state_target": city["state"],
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Error en Pexels: {e}")
            return []
    
    def search_openstreetmap_images(self, city: Dict, max_images: int = 10) -> List[Dict]:
        """Genera referencias a OpenStreetMap (100% gratis y open source)"""
        # OpenStreetMap es open source y no requiere API key
        results = []
        
        try:
            # Diferentes zooms para capturar variedad
            zooms = [12, 13, 14, 15, 16]
            
            for i, zoom in enumerate(zooms[:max_images]):
                # URL de tile OpenStreetMap (gratis, sin límites para uso razonable)
                # Usamos el servicio de tiles estático
                osm_url = f"https://www.openstreetmap.org/export/embed.html?bbox={city['lon']-0.01},{city['lat']-0.01},{city['lon']+0.01},{city['lat']+0.01}&layer=mapnik"
                
                results.append({
                    "source": "openstreetmap",
                    "url": osm_url,
                    "photo_id": f"osm_{city['name']}_{zoom}",
                    "title": f"OpenStreetMap view of {city['name']} (zoom {zoom})",
                    "lat": city["lat"],
                    "lon": city["lon"],
                    "city_target": city["name"],
                    "state_target": city["state"],
                    "note": "OpenStreetMap reference"
                })
            
            return results
        
        except Exception as e:
            print(f"❌ Error generando refs OpenStreetMap: {e}")
            return []
    
    def search_geohack_images(self, city: Dict, max_images: int = 5) -> List[Dict]:
        """Busca imágenes relacionadas en Wikipedia/GeoHack (100% gratis)"""
        try:
            # GeoHack es un servicio de Wikimedia para coordenadas geográficas
            query = f"{city['name']} {city['state']} Mexico"
            
            # Buscar en Wikipedia en español
            wiki_api = "https://es.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": max_images,
            }
            
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(wiki_api, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            results = []
            search_results = data.get("query", {}).get("search", [])
            
            for result in search_results[:max_images]:
                page_title = result["title"]
                page_id = result["pageid"]
                
                # Obtener imágenes de la página
                img_params = {
                    "action": "query",
                    "format": "json",
                    "pageids": page_id,
                    "prop": "images",
                    "imlimit": 1,
                }
                
                img_response = requests.get(wiki_api, params=img_params, headers=headers, timeout=10)
                img_data = img_response.json()
                
                page_data = img_data.get("query", {}).get("pages", {}).get(str(page_id), {})
                images = page_data.get("images", [])
                
                if images:
                    img_title = images[0]["title"]
                    
                    # Obtener URL de la imagen
                    url_params = {
                        "action": "query",
                        "format": "json",
                        "titles": img_title,
                        "prop": "imageinfo",
                        "iiprop": "url",
                        "iiurlwidth": 800,
                    }
                    
                    url_response = requests.get("https://commons.wikimedia.org/w/api.php", 
                                               params=url_params, headers=headers, timeout=10)
                    url_data = url_response.json()
                    
                    pages = url_data.get("query", {}).get("pages", {})
                    for page in pages.values():
                        if "imageinfo" in page:
                            img_url = page["imageinfo"][0].get("thumburl") or page["imageinfo"][0].get("url")
                            
                            if img_url and img_url.lower().endswith(('.jpg', '.jpeg', '.png')):
                                results.append({
                                    "source": "wikipedia",
                                    "url": img_url,
                                    "photo_id": f"wiki_{page_id}",
                                    "title": page_title,
                                    "description": result.get("snippet", ""),
                                    "lat": city["lat"],
                                    "lon": city["lon"],
                                    "city_target": city["name"],
                                    "state_target": city["state"],
                                })
                                break
                
                time.sleep(0.5)  # Rate limiting
            
            return results
        
        except Exception as e:
            print(f"❌ Error en Wikipedia: {e}")
            return []
    
    def download_image(self, img_data: Dict) -> Optional[str]:
        """Descarga una imagen y retorna el path local"""
        try:
            response = requests.get(img_data["url"], timeout=15, stream=True)
            response.raise_for_status()
            
            # Generar nombre único
            url_hash = hashlib.md5(img_data["url"].encode()).hexdigest()[:12]
            source = img_data["source"]
            city_slug = img_data["city_target"].lower().replace(" ", "_")
            
            filename = f"{source}_{city_slug}_{url_hash}.jpg"
            filepath = self.images_dir / filename
            
            # Guardar imagen
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(filepath)
        
        except Exception as e:
            print(f"❌ Error descargando {img_data['url']}: {e}")
            return None
    
    def mine_city(self, city: Dict, images_per_source: int = 10):
        """Descarga imágenes para una ciudad específica usando fuentes GRATUITAS"""
        print(f"\n🔍 Minando: {city['name']}, {city['state']}")
        
        all_results = []
        
        # Wikimedia Commons (100% gratis, open source)
        print("  → Buscando en Wikimedia Commons (open source)...")
        wiki_results = self.search_wikimedia_commons(city, images_per_source)
        all_results.extend(wiki_results)
        time.sleep(1)  # rate limiting ético
        
        # Wikipedia con imágenes (100% gratis)
        print("  → Buscando en Wikipedia (artículos con imágenes)...")
        wikipedia_results = self.search_geohack_images(city, min(images_per_source, 5))
        all_results.extend(wikipedia_results)
        time.sleep(1)
        
        # Pexels (API gratuita)
        print("  → Buscando en Pexels (gratis)...")
        pexels_results = self.search_pexels_free(city, images_per_source)
        all_results.extend(pexels_results)
        time.sleep(1)
        
        # Descargar imágenes
        downloaded = 0
        for img_data in all_results:
            # Verificar si ya existe
            existing = [m for m in self.metadata["images"] 
                       if m.get("url") == img_data["url"]]
            if existing:
                continue
            
            print(f"  ⬇️  Descargando {img_data['source']}/{img_data['photo_id']}...", end=" ")
            local_path = self.download_image(img_data)
            
            if local_path:
                img_data["local_path"] = local_path
                img_data["downloaded_at"] = datetime.now().isoformat()
                img_data["annotated"] = False
                img_data["annotation"] = None
                self.metadata["images"].append(img_data)
                downloaded += 1
                print("✅")
            else:
                print("❌")
            
            time.sleep(0.5)  # rate limiting
        
        print(f"  ✅ Descargadas: {downloaded} imágenes nuevas")
        self.save_metadata()
    
    def mine_all_cities(self, images_per_source: int = 5, limit_cities: Optional[int] = None):
        """Descarga imágenes para todas las ciudades"""
        cities = self.cities[:limit_cities] if limit_cities else self.cities
        
        print(f"🚀 Iniciando minería para {len(cities)} ciudades")
        print(f"📊 Imágenes por fuente: {images_per_source}")
        
        for i, city in enumerate(cities, 1):
            print(f"\n[{i}/{len(cities)}]", end=" ")
            self.mine_city(city, images_per_source)
            
            # Guardar cada 5 ciudades
            if i % 5 == 0:
                self.save_metadata()
        
        self.save_metadata()
        print(f"\n\n✅ Minería completada. Total de imágenes: {len(self.metadata['images'])}")
        print(f"📁 Ubicación: {self.images_dir}")
    
    def stats(self):
        """Muestra estadísticas del dataset"""
        total = len(self.metadata["images"])
        annotated = sum(1 for img in self.metadata["images"] if img.get("annotated"))
        
        by_city = {}
        by_source = {}
        
        for img in self.metadata["images"]:
            city = img["city_target"]
            source = img["source"]
            by_city[city] = by_city.get(city, 0) + 1
            by_source[source] = by_source.get(source, 0) + 1
        
        print("\n📊 ESTADÍSTICAS DEL DATASET")
        print("=" * 50)
        print(f"Total de imágenes:    {total}")
        if total > 0:
            print(f"Imágenes anotadas:    {annotated} ({annotated/total*100:.1f}%)")
            print(f"Imágenes pendientes:  {total - annotated}")
        else:
            print("Imágenes anotadas:    0")
            print("Imágenes pendientes:  0")
            print("\n⚠️  No hay imágenes en el dataset todavía.")
            print("\n💡 Comienza minando imágenes:")
            print("   python data_mining.py --mode city --city 'Puebla' --images 10")
            print("   python data_mining.py --mode all --images 5 --limit 5")
            return
        
        print(f"\nPor fuente:")
        for source, count in sorted(by_source.items()):
            print(f"  {source:12} {count:4} imágenes")
        print(f"\nTop 10 ciudades:")
        for city, count in sorted(by_city.items(), key=lambda x: -x[1])[:10]:
            print(f"  {city:30} {count:3} imágenes")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="🆓 Minería GRATUITA de imágenes (Wikimedia, Pexels, Google Static Maps)"
    )
    parser.add_argument("--mode", choices=["city", "all", "stats"], default="stats",
                       help="Modo de operación")
    parser.add_argument("--city", type=str, help="Nombre de ciudad (para mode=city)")
    parser.add_argument("--images", type=int, default=5, help="Imágenes por fuente")
    parser.add_argument("--limit", type=int, help="Limitar número de ciudades")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🆓 SISTEMA DE MINERÍA OPEN SOURCE - 100% GRATUITO")
    print("="*70)
    print("Fuentes:")
    print("  ✅ Wikimedia Commons (sin límites, sin registro)")
    print("  ✅ Wikipedia (artículos con imágenes)")
    print("  ✅ Pexels (API gratuita - requiere key gratis)")
    print("\nPara Pexels (opcional pero recomendado):")
    print("  1. Registrarse en https://www.pexels.com/api/ (gratis, 2 min)")
    print("  2. Copiar API key")
    print("  3. Configurar: $env:PEXELS_API_KEY='tu_key'")
    print("="*70 + "\n")
    
    miner = ImageMiner()
    
    if args.mode == "stats":
        miner.stats()
    
    elif args.mode == "all":
        miner.mine_all_cities(images_per_source=args.images, limit_cities=args.limit)
    
    elif args.mode == "city":
        if not args.city:
            print("❌ Debes especificar --city con mode=city")
            return
        
        city_match = [c for c in miner.cities if c["name"].lower() == args.city.lower()]
        if not city_match:
            print(f"❌ Ciudad '{args.city}' no encontrada")
            print(f"Ciudades disponibles: {', '.join(c['name'] for c in miner.cities[:10])}...")
            return
        
        miner.mine_city(city_match[0], images_per_source=args.images)


if __name__ == "__main__":
    main()
