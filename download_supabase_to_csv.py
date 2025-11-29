"""
Descargar todas las tablas de Supabase a archivos CSV para backup en GitHub.
"""

import os
import csv
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Directorio de salida
OUTPUT_DIR = Path('data/supabase_backup')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def download_table_to_csv(table_name, output_file):
    """Descargar tabla completa de Supabase a CSV"""
    
    print(f"📥 Descargando tabla '{table_name}'...")
    
    try:
        # Obtener todos los registros
        result = supabase.table(table_name).select('*').execute()
        data = result.data
        
        if not data:
            print(f"   ⚠️  Tabla vacía: {table_name}")
            return 0
        
        # Escribir a CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Usar las keys del primer registro como headers
            fieldnames = data[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(data)
        
        print(f"   ✅ {len(data)} registros → {output_file}")
        return len(data)
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return 0

def main():
    """Descargar todas las tablas importantes"""
    
    print("🗄️  DESCARGA DE BASE DE DATOS SUPABASE → CSV\n")
    
    tables = {
        'image_metadata': 'image_metadata.csv',
        'annotations': 'annotations.csv',
        'deleted_images': 'deleted_images.csv'
    }
    
    total_records = 0
    
    for table_name, filename in tables.items():
        output_file = OUTPUT_DIR / filename
        count = download_table_to_csv(table_name, output_file)
        total_records += count
    
    print(f"\n✅ Descarga completada!")
    print(f"   📊 Total de registros: {total_records}")
    print(f"   📂 Ubicación: {OUTPUT_DIR.absolute()}")
    print(f"\n💡 Tip: Ahora puedes hacer commit de estos CSVs a GitHub:")
    print(f"   git add {OUTPUT_DIR}")
    print(f"   git commit -m 'Backup de base de datos Supabase'")
    print(f"   git push origin main")

if __name__ == '__main__':
    main()
