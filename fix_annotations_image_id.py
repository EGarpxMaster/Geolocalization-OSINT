"""
Script para actualizar las anotaciones existentes en Supabase con el image_id correcto.
Las anotaciones antiguas solo tienen filename, necesitan el image_id de image_metadata.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def fix_annotations_image_id():
    """Actualizar todas las anotaciones con image_id desde image_metadata"""
    
    print("🔧 Reparando image_id en anotaciones...\n")
    
    # 1. Obtener todas las anotaciones
    print("📊 Cargando anotaciones...")
    result = supabase.table('annotations').select('id, filename, image_id').execute()
    annotations = result.data
    
    print(f"✅ {len(annotations)} anotaciones en total")
    
    # 2. Filtrar las que no tienen image_id o tienen NULL
    missing_id = [ann for ann in annotations if not ann.get('image_id')]
    print(f"⚠️  {len(missing_id)} anotaciones sin image_id\n")
    
    if not missing_id:
        print("✅ Todas las anotaciones ya tienen image_id!")
        return
    
    # 3. Obtener todos los image_id de image_metadata de una vez
    print("🔍 Obteniendo image_id desde image_metadata...")
    filenames = [ann['filename'] for ann in missing_id]
    metadata_result = supabase.table('image_metadata').select('id, filename').in_('filename', filenames).execute()
    filename_to_id = {row['filename']: row['id'] for row in metadata_result.data}
    
    print(f"✅ Encontrados {len(filename_to_id)} image_id\n")
    
    # 4. Actualizar cada anotación
    print(f"💾 Actualizando {len(missing_id)} anotaciones...")
    updated = 0
    skipped = 0
    
    for ann in missing_id:
        image_id = filename_to_id.get(ann['filename'])
        
        if not image_id:
            print(f"   ⚠️  Saltando {ann['filename']} - no encontrado en image_metadata")
            skipped += 1
            continue
        
        try:
            supabase.table('annotations').update({
                'image_id': image_id
            }).eq('id', ann['id']).execute()
            
            updated += 1
            if updated % 10 == 0:
                print(f"   ✓ Actualizadas {updated}/{len(missing_id)}...")
        except Exception as e:
            print(f"   ❌ Error actualizando {ann['filename']}: {e}")
    
    print(f"\n✅ Actualización completada!")
    print(f"   - Actualizadas: {updated}")
    print(f"   - Saltadas: {skipped}")
    
    # 5. Verificar estado final
    print("\n📊 Verificando estado final...")
    final_result = supabase.table('annotations').select('id, image_id').execute()
    final_annotations = final_result.data
    
    with_id = sum(1 for ann in final_annotations if ann.get('image_id'))
    without_id = len(final_annotations) - with_id
    
    print(f"   - Con image_id: {with_id}")
    print(f"   - Sin image_id: {without_id}")
    
    if without_id == 0:
        print("\n🎉 ¡Todas las anotaciones tienen image_id!")
    else:
        print(f"\n⚠️  Aún hay {without_id} anotaciones sin image_id")

if __name__ == '__main__':
    fix_annotations_image_id()
