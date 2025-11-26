"""
Limpiar registros huérfanos en la base de datos.
Elimina registros en image_metadata que NO tienen imagen correspondiente en Storage.
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

def cleanup_orphan_records():
    """Eliminar registros sin imagen en Storage"""
    
    print("🔍 Iniciando limpieza de registros huérfanos...\n")
    
    # 1. Obtener todas las imágenes en Storage (con paginación)
    print("📦 Obteniendo lista completa de Storage...")
    storage_files = set()
    offset = 0
    limit = 1000
    
    while True:
        files = supabase.storage.from_('geolocalization-images').list(
            path='',
            options={'limit': limit, 'offset': offset}
        )
        
        if not files:
            break
        
        for f in files:
            if f['name'].endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
                storage_files.add(f['name'])
        
        if len(files) < limit:
            break
        
        offset += limit
    
    print(f"✅ {len(storage_files)} imágenes encontradas en Storage\n")
    
    # 2. Obtener todos los registros de la BD
    print("📊 Obteniendo registros de la base de datos...")
    all_records = supabase.table('image_metadata').select('id, filename').execute()
    
    print(f"✅ {len(all_records.data)} registros en image_metadata\n")
    
    # 3. Identificar registros huérfanos (sin imagen en Storage)
    orphan_ids = []
    orphan_files = []
    
    for record in all_records.data:
        filename = record['filename']
        if filename not in storage_files:
            orphan_ids.append(record['id'])
            orphan_files.append(filename)
    
    print(f"🔍 Registros huérfanos encontrados: {len(orphan_ids)}\n")
    
    if not orphan_ids:
        print("✅ No hay registros huérfanos. Base de datos limpia!")
        return
    
    # Mostrar algunos ejemplos
    print("📝 Ejemplos de registros a eliminar:")
    for filename in orphan_files[:10]:
        print(f"   - {filename}")
    if len(orphan_files) > 10:
        print(f"   ... y {len(orphan_files) - 10} más\n")
    
    # 4. Confirmar eliminación
    print(f"\n⚠️  SE ELIMINARÁN {len(orphan_ids)} REGISTROS de la base de datos")
    confirm = input("¿Continuar? (si/no): ").strip().lower()
    
    if confirm not in ['si', 's', 'yes', 'y']:
        print("❌ Operación cancelada")
        return
    
    # 5. Eliminar registros en lotes
    print(f"\n🗑️  Eliminando {len(orphan_ids)} registros...")
    
    batch_size = 100
    deleted_count = 0
    
    for i in range(0, len(orphan_ids), batch_size):
        batch = orphan_ids[i:i+batch_size]
        
        # Eliminar usando .in_()
        supabase.table('image_metadata').delete().in_('id', batch).execute()
        
        deleted_count += len(batch)
        print(f"   ✓ Eliminados {deleted_count}/{len(orphan_ids)} registros...")
    
    print(f"\n✅ Limpieza completada!")
    
    # 6. Verificar estado final
    final_count = supabase.table('image_metadata').select('id', count='exact').execute()
    with_url = supabase.table('image_metadata').select('id', count='exact').not_.is_('image_url', 'null').execute()
    without_url = supabase.table('image_metadata').select('id', count='exact').is_('image_url', 'null').execute()
    
    print(f"\n📊 Estado final de la base de datos:")
    print(f"   - Total de registros: {final_count.count}")
    print(f"   - Con URL: {with_url.count}")
    print(f"   - Sin URL: {without_url.count}")
    
    if without_url.count > 0:
        print(f"\n💡 Ejecuta 'python sync_supabase_urls.py' para sincronizar las URLs faltantes")

if __name__ == '__main__':
    cleanup_orphan_records()
