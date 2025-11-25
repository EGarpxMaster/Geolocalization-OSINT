-- ============================================================================
-- FIX: DESHABILITAR RLS TEMPORALMENTE (PARA MIGRACIÓN)
-- ============================================================================
-- Ejecuta esto en el SQL Editor de Supabase

-- OPCIÓN 1: Deshabilitar RLS completamente (más fácil para desarrollo)
ALTER TABLE image_metadata DISABLE ROW LEVEL SECURITY;
ALTER TABLE annotations DISABLE ROW LEVEL SECURITY;
ALTER TABLE deleted_images DISABLE ROW LEVEL SECURITY;

-- Verificar que RLS está deshabilitado
SELECT tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('image_metadata', 'annotations', 'deleted_images');

-- Nota: Después de la migración, si quieres re-habilitar RLS:
-- ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE deleted_images ENABLE ROW LEVEL SECURITY;
