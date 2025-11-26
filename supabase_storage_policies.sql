-- POLÍTICAS DE SUPABASE STORAGE PARA BUCKET PÚBLICO
-- ===================================================
-- Permitir operaciones en el bucket geolocalization-images

-- 1. Permitir INSERT (subida de archivos)
CREATE POLICY "Permitir subida pública de imágenes"
ON storage.objects
FOR INSERT
TO public
WITH CHECK (bucket_id = 'geolocalization-images');

-- 2. Permitir SELECT (lectura de archivos)
CREATE POLICY "Permitir lectura pública de imágenes"
ON storage.objects
FOR SELECT
TO public
USING (bucket_id = 'geolocalization-images');

-- 3. Permitir UPDATE (actualización de archivos)
CREATE POLICY "Permitir actualización pública de imágenes"
ON storage.objects
FOR UPDATE
TO public
USING (bucket_id = 'geolocalization-images')
WITH CHECK (bucket_id = 'geolocalization-images');

-- 4. Permitir DELETE (eliminación de archivos)
CREATE POLICY "Permitir eliminación pública de imágenes"
ON storage.objects
FOR DELETE
TO public
USING (bucket_id = 'geolocalization-images');

-- NOTA: Ejecuta este SQL en el SQL Editor de Supabase
-- Ir a: Dashboard → SQL Editor → New query → Pegar y ejecutar
