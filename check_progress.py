# check_progress.py
# Script rápido para monitorear el progreso de la minería de datos

import json
from pathlib import Path
from collections import Counter

METADATA_FILE = "data/mining/metadata.json"

def check_progress():
    """Muestra el progreso actual de la minería"""
    metadata_path = Path(METADATA_FILE)
    
    if not metadata_path.exists():
        print("❌ No hay datos de minería todavía.")
        print("💡 Ejecuta: python data_mining.py --mode all --images 10 --limit 32")
        return
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data.get("images", [])
    total = len(images)
    
    print("\n" + "="*60)
    print("📊 PROGRESO DE MINERÍA DE DATOS")
    print("="*60)
    print(f"\n✅ Total de imágenes descargadas: {total}")
    
    # Por ciudad
    by_city = Counter(img["city_target"] for img in images)
    cities_count = len(by_city)
    
    print(f"🏙️  Ciudades con imágenes: {cities_count}/32")
    
    # Por fuente
    by_source = Counter(img["source"] for img in images)
    print(f"\n📚 Por fuente:")
    for source, count in sorted(by_source.items()):
        print(f"   {source:20} {count:4} imágenes")
    
    # Top 10 ciudades
    print(f"\n🏆 Top 10 ciudades con más imágenes:")
    for city, count in by_city.most_common(10):
        print(f"   {city:30} {count:3} imágenes")
    
    # Estados cubiertos
    by_state = Counter(img["state_target"] for img in images)
    states_count = len(by_state)
    
    print(f"\n🗺️  Estados cubiertos: {states_count}/32")
    if states_count < 32:
        print(f"   Faltan: {32 - states_count} estados")
    
    # Progreso estimado
    if cities_count < 32:
        progress = (cities_count / 32) * 100
        print(f"\n⏳ Progreso estimado: {progress:.1f}%")
        print(f"   Ciudades restantes: {32 - cities_count}")
    else:
        print(f"\n🎉 ¡Minería de 32 ciudades completada!")
        print(f"\n💡 Siguiente paso: Anotación manual")
        print(f"   streamlit run annotation_tool.py")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    check_progress()
