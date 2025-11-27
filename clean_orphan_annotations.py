"""
Limpiar anotaciones huérfanas (sin image_id) que no tienen imagen correspondiente en image_metadata.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def clean_orphan_annotations():
    """Eliminar anotaciones que no tienen imagen en image_metadata"""
    
    print("🧹 Limpiando anotaciones huérfanas...\n")
    
    # 1. Obtener anotaciones sin image_id
    print("📊 Buscando anotaciones sin image_id...")
    result = supabase.table('annotations').select('id, filename').is_('image_id', 'null').execute()
    orphans = result.data
    
    print(f"⚠️  {len(orphans)} anotaciones sin image_id\n")
    
    if not orphans:
        print("✅ No hay anotaciones huérfanas!")
        return
    
    # 2. Verificar si existen en image_metadata
    print("🔍 Verificando existencia en image_metadata...")
    filenames = [ann['filename'] for ann in orphans]
    metadata_result = supabase.table('image_metadata').select('filename').in_('filename', filenames).execute()
    existing_filenames = {row['filename'] for row in metadata_result.data}
    
    print(f"✅ {len(existing_filenames)} sí existen en image_metadata")
    print(f"❌ {len(orphans) - len(existing_filenames)} NO existen (huérfanas reales)\n")
    
    # 3. Separar las que realmente son huérfanas
    truly_orphan = [ann for ann in orphans if ann['filename'] not in existing_filenames]
    
    if not truly_orphan:
        print("✅ Todas las anotaciones sin image_id tienen imagen en metadata")
        print("   (Ejecuta fix_annotations_image_id.py de nuevo para vincularlas)")
        return
    
    # 4. Preguntar confirmación
    print(f"⚠️  Se eliminarán {len(truly_orphan)} anotaciones huérfanas:")
    for i, ann in enumerate(truly_orphan[:10], 1):
        print(f"   {i}. {ann['filename']}")
    if len(truly_orphan) > 10:
        print(f"   ... y {len(truly_orphan) - 10} más")
    
    print("\n¿Continuar? (escribe 'SI' para confirmar)")
    confirm = input("> ").strip()
    
    if confirm != 'SI':
        print("❌ Cancelado")
        return
    
    # 5. Eliminar huérfanas
    print(f"\n🗑️  Eliminando {len(truly_orphan)} anotaciones huérfanas...")
    deleted = 0
    
    for ann in truly_orphan:
        try:
            supabase.table('annotations').delete().eq('id', ann['id']).execute()
            deleted += 1
            if deleted % 10 == 0:
                print(f"   ✓ Eliminadas {deleted}/{len(truly_orphan)}...")
        except Exception as e:
            print(f"   ❌ Error eliminando {ann['filename']}: {e}")
    
    print(f"\n✅ Limpieza completada!")
    print(f"   - Eliminadas: {deleted}")
    
    # 6. Estado final
    print("\n📊 Estado final de anotaciones:")
    final = supabase.table('annotations').select('id, image_id').execute()
    with_id = sum(1 for ann in final.data if ann.get('image_id'))
    without_id = len(final.data) - with_id
    
    print(f"   - Total: {len(final.data)}")
    print(f"   - Con image_id: {with_id}")
    print(f"   - Sin image_id: {without_id}")

if __name__ == '__main__':
    clean_orphan_annotations()
