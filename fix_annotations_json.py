"""Script para regenerar annotations.json desde annotations.csv"""
import json
import csv
from pathlib import Path

# Rutas
ann_csv = Path('data/mining/annotations.csv')
ann_json = Path('data/mining/annotations.json')

# Leer CSV
annotations = {'images': [], 'deleted_images': []}

if ann_csv.exists():
    with open(ann_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convertir strings de booleanos a bool
            elements = {
                'landmarks': row['landmarks'] == 'True',
                'architecture': row['architecture'] == 'True',
                'signs': row['signs'] == 'True',
                'nature': row['nature'] == 'True',
                'urban': row['urban'] == 'True',
                'beach': row['beach'] == 'True',
                'people': row['people'] == 'True',
                'vehicles': row['vehicles'] == 'True',
                'text': row['text'] == 'True'
            }
            
            # Convertir tags de string a lista
            custom_tags = [tag.strip() for tag in row['custom_tags'].split(',') if tag.strip()]
            
            annotation = {
                'filename': row['filename'],
                'city': row['city'],
                'state': row['state'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'source': row.get('source', ''),
                'correct_city': row['correct_city'],
                'quality': int(row['quality']),
                'custom_tags': custom_tags,
                'elements': elements,
                'confidence': int(row['confidence']),
                'notes': row['notes'],
                'annotated_at': row['annotated_at'],
                'annotated_by': row['annotated_by']
            }
            
            annotations['images'].append(annotation)

# Guardar JSON
with open(ann_json, 'w', encoding='utf-8') as f:
    json.dump(annotations, f, indent=2, ensure_ascii=False)

print(f'✅ JSON regenerado con {len(annotations["images"])} anotaciones')
print(f'📄 Archivo guardado en: {ann_json}')
