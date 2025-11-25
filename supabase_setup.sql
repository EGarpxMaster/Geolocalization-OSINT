-- ============================================================================
-- SCHEMA SUPABASE PARA GEOLOCALIZATION-OSINT
-- ============================================================================
-- Ejecuta este script en el SQL Editor de Supabase
-- https://supabase.com/dashboard/project/YOUR_PROJECT/sql

-- ============================================================================
-- 1. TABLA DE METADATOS DE IMÁGENES
-- ============================================================================
CREATE TABLE IF NOT EXISTS image_metadata (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL, -- 'pexels', 'wikimedia', 'wikipedia'
    photo_id TEXT,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    lat DECIMAL(10, 7) NOT NULL,
    lon DECIMAL(10, 7) NOT NULL,
    url TEXT,
    title TEXT,
    photographer TEXT,
    image_url TEXT, -- URL pública de la imagen (Supabase Storage o externo)
    width INTEGER,
    height INTEGER,
    size INTEGER, -- tamaño en bytes
    hash TEXT,
    downloaded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Índices para búsquedas rápidas
CREATE INDEX idx_metadata_city ON image_metadata(city);
CREATE INDEX idx_metadata_state ON image_metadata(state);
CREATE INDEX idx_metadata_source ON image_metadata(source);
CREATE INDEX idx_metadata_filename ON image_metadata(filename);

-- ============================================================================
-- 2. TABLA DE ANOTACIONES
-- ============================================================================
CREATE TABLE IF NOT EXISTS annotations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    image_id UUID REFERENCES image_metadata(id) ON DELETE CASCADE,
    filename TEXT NOT NULL, -- Redundante pero útil para búsquedas
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    lat DECIMAL(10, 7) NOT NULL,
    lon DECIMAL(10, 7) NOT NULL,
    
    -- Verificación
    correct_city TEXT NOT NULL, -- '✅', '❌', '🤔'
    
    -- Calidad/Relevancia
    quality INTEGER CHECK (quality BETWEEN 1 AND 5),
    confidence INTEGER CHECK (confidence BETWEEN 0 AND 100),
    
    -- Elementos detectados
    has_landmarks BOOLEAN DEFAULT FALSE,
    has_architecture BOOLEAN DEFAULT FALSE,
    has_signs BOOLEAN DEFAULT FALSE,
    has_nature BOOLEAN DEFAULT FALSE,
    has_urban BOOLEAN DEFAULT FALSE,
    has_beach BOOLEAN DEFAULT FALSE,
    has_people BOOLEAN DEFAULT FALSE,
    has_vehicles BOOLEAN DEFAULT FALSE,
    has_text BOOLEAN DEFAULT FALSE,
    
    -- Tags y notas
    custom_tags TEXT[], -- Array de strings
    notes TEXT,
    
    -- Metadata de anotación
    annotated_by TEXT NOT NULL, -- Nombre del anotador
    annotated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraint: una sola anotación por imagen
    UNIQUE(image_id)
);

-- Índices
CREATE INDEX idx_annotations_filename ON annotations(filename);
CREATE INDEX idx_annotations_city ON annotations(city);
CREATE INDEX idx_annotations_annotated_by ON annotations(annotated_by);
CREATE INDEX idx_annotations_quality ON annotations(quality);

-- ============================================================================
-- 3. TABLA DE IMÁGENES ELIMINADAS
-- ============================================================================
CREATE TABLE IF NOT EXISTS deleted_images (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    reason TEXT, -- 'corrupted', 'low_quality', 'irrelevant', 'duplicate'
    deleted_by TEXT NOT NULL,
    deleted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================================================
-- 4. FUNCIONES Y TRIGGERS
-- ============================================================================

-- Función para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers para updated_at
CREATE TRIGGER update_image_metadata_updated_at BEFORE UPDATE ON image_metadata
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_annotations_updated_at BEFORE UPDATE ON annotations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 5. ROW LEVEL SECURITY (RLS) - OPCIONAL
-- ============================================================================
-- Habilitar RLS para acceso controlado
ALTER TABLE image_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE deleted_images ENABLE ROW LEVEL SECURITY;

-- Políticas: permitir lectura a todos, escritura autenticada
-- LECTURA PÚBLICA
CREATE POLICY "Enable read access for all users" ON image_metadata
    FOR SELECT USING (true);

CREATE POLICY "Enable read access for all users" ON annotations
    FOR SELECT USING (true);

-- ESCRITURA AUTENTICADA (o pública si prefieres)
-- Opción 1: Solo usuarios autenticados pueden insertar/actualizar
CREATE POLICY "Enable insert for authenticated users only" ON image_metadata
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Enable update for authenticated users only" ON image_metadata
    FOR UPDATE USING (auth.role() = 'authenticated');

CREATE POLICY "Enable insert for authenticated users only" ON annotations
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Enable update for authenticated users only" ON annotations
    FOR UPDATE USING (auth.role() = 'authenticated');

CREATE POLICY "Enable insert for authenticated users only" ON deleted_images
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Opción 2: Si quieres permitir escritura pública (DESCOMENTA ESTO Y COMENTA LO DE ARRIBA)
-- CREATE POLICY "Enable insert for all users" ON image_metadata FOR INSERT WITH CHECK (true);
-- CREATE POLICY "Enable update for all users" ON image_metadata FOR UPDATE USING (true);
-- CREATE POLICY "Enable insert for all users" ON annotations FOR INSERT WITH CHECK (true);
-- CREATE POLICY "Enable update for all users" ON annotations FOR UPDATE USING (true);
-- CREATE POLICY "Enable insert for all users" ON deleted_images FOR INSERT WITH CHECK (true);

-- ============================================================================
-- 6. VISTAS ÚTILES
-- ============================================================================

-- Vista: Imágenes con anotaciones (JOIN)
CREATE OR REPLACE VIEW annotated_images AS
SELECT 
    m.id,
    m.filename,
    m.city,
    m.state,
    m.lat,
    m.lon,
    m.source,
    m.image_url,
    a.correct_city,
    a.quality,
    a.confidence,
    a.custom_tags,
    a.notes,
    a.annotated_by,
    a.annotated_at
FROM image_metadata m
INNER JOIN annotations a ON m.id = a.image_id;

-- Vista: Imágenes pendientes de anotación
CREATE OR REPLACE VIEW pending_images AS
SELECT 
    m.id,
    m.filename,
    m.city,
    m.state,
    m.lat,
    m.lon,
    m.source,
    m.image_url,
    m.downloaded_at
FROM image_metadata m
LEFT JOIN annotations a ON m.id = a.image_id
LEFT JOIN deleted_images d ON m.filename = d.filename
WHERE a.id IS NULL AND d.id IS NULL;

-- Vista: Estadísticas generales
CREATE OR REPLACE VIEW annotation_stats AS
SELECT 
    COUNT(DISTINCT m.id) as total_images,
    COUNT(DISTINCT a.id) as annotated_images,
    COUNT(DISTINCT d.id) as deleted_images,
    COUNT(DISTINCT m.id) - COUNT(DISTINCT a.id) - COUNT(DISTINCT d.id) as pending_images,
    COUNT(DISTINCT m.state) as unique_states,
    COUNT(DISTINCT a.annotated_by) as unique_annotators
FROM image_metadata m
LEFT JOIN annotations a ON m.id = a.image_id
LEFT JOIN deleted_images d ON m.filename = d.filename;

-- ============================================================================
-- VERIFICACIÓN
-- ============================================================================
-- Verifica que todo se creó correctamente:
SELECT 
    schemaname,
    tablename,
    tableowner
FROM pg_tables 
WHERE schemaname = 'public' 
    AND tablename IN ('image_metadata', 'annotations', 'deleted_images')
ORDER BY tablename;
