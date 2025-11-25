-- ============================================================================
-- FIX: PERMISOS DE STORAGE PARA SUPABASE
-- ============================================================================
-- Ejecuta esto en el SQL Editor de Supabase

-- OPCIÓN 1: Crear políticas públicas para el bucket
-- Reemplaza 'geolocalization-images' con el nombre de tu bucket si es diferente

-- Permitir lectura pública (cualquiera puede ver las imágenes)
INSERT INTO storage.buckets (id, name, public)
VALUES ('geolocalization-images', 'geolocalization-images', true)
ON CONFLICT (id) DO UPDATE SET public = true;

-- Permitir subida pública (para desarrollo)
CREATE POLICY "Public Access"
ON storage.objects FOR ALL
USING (bucket_id = 'geolocalization-images');

-- O si prefieres políticas más específicas:
-- Lectura pública
CREATE POLICY "Public Read Access"
ON storage.objects FOR SELECT
USING (bucket_id = 'geolocalization-images');

-- Subida pública
CREATE POLICY "Public Upload Access"
ON storage.objects FOR INSERT
WITH CHECK (bucket_id = 'geolocalization-images');

-- Actualización pública
CREATE POLICY "Public Update Access"
ON storage.objects FOR UPDATE
USING (bucket_id = 'geolocalization-images');

-- VERIFICAR POLÍTICAS
SELECT * FROM storage.buckets WHERE name = 'geolocalization-images';
