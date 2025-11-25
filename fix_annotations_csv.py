"""Script para regenerar annotations.csv desde annotations.json"""
import json
import csv
from pathlib import Path

# Rutas
ann_json = Path('data/mining/annotations.json')
ann_csv = Path('data/mining/annotations.csv')

# Cargar JSON
if ann_json.exists():
    with open(ann_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
else:
    data = {'images': []}

# Crear CSV
with open(ann_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    
    # Encabezado
    writer.writerow([
        'filename', 'city', 'state', 'lat', 'lon',
        'correct_city', 'quality', 'confidence',
        'landmarks', 'architecture', 'signs', 'nature', 'urban', 'beach',
        'people', 'vehicles', 'text', 'custom_tags', 'notes',
        'annotated_at', 'annotated_by'
    ])
    
    # Escribir cada anotación
    for ann in data.get('images', []):
        elements = ann.get('elements', {})
        tags_str = ','.join(ann.get('custom_tags', [])).replace('\n', ' ').replace('\r', ' ')
        notes_str = ann.get('notes', '').replace('\n', ' ').replace('\r', ' ')
        
        writer.writerow([
            ann.get('filename', ''),
            ann.get('city', ''),
            ann.get('state', ''),
            ann.get('lat', 0.0),
            ann.get('lon', 0.0),
            ann.get('correct_city', ''),
            ann.get('quality', 0),
            ann.get('confidence', 0),
            elements.get('landmarks', False),
            elements.get('architecture', False),
            elements.get('signs', False),
            elements.get('nature', False),
            elements.get('urban', False),
            elements.get('beach', False),
            elements.get('people', False),
            elements.get('vehicles', False),
            elements.get('text', False),
            tags_str,
            notes_str,
            ann.get('annotated_at', ''),
            ann.get('annotated_by', 'Emma')
        ])

print(f'✅ CSV regenerado con {len(data.get("images", []))} anotaciones')
print(f'📄 Archivo guardado en: {ann_csv}')
