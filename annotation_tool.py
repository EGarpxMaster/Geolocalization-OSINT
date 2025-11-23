# annotation_tool.py
# Herramienta interactiva de anotación manual con Streamlit
# Permite categorizar imágenes minadas para crear dataset de entrenamiento

import os
import json
import streamlit as st
from pathlib import Path
from PIL import Image
from datetime import datetime
import pandas as pd

# Configuración
METADATA_FILE = "data/mining/metadata.json"
ANNOTATIONS_FILE = "data/mining/annotations.json"

class AnnotationTool:
    def __init__(self):
        self.metadata_path = Path(METADATA_FILE)
        self.annotations_path = Path(ANNOTATIONS_FILE)
        
        if not self.metadata_path.exists():
            st.error(f"❌ No se encontró {METADATA_FILE}. Ejecuta primero data_mining.py")
            st.stop()
        
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # Cargar anotaciones existentes
        if self.annotations_path.exists():
            with open(self.annotations_path, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        else:
            self.annotations = {"annotations": [], "annotators": []}
    
    def save_annotations(self):
        """Guarda las anotaciones"""
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.annotations_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, indent=2, ensure_ascii=False)
    
    def get_unannotated_images(self):
        """Retorna imágenes que aún no han sido anotadas"""
        annotated_paths = {a["local_path"] for a in self.annotations["annotations"]}
        return [img for img in self.metadata["images"] 
                if img.get("local_path") and img["local_path"] not in annotated_paths]
    
    def get_annotation_stats(self):
        """Estadísticas de anotación"""
        total = len(self.metadata["images"])
        annotated = len(self.annotations["annotations"])
        pending = total - annotated
        
        return {
            "total": total,
            "annotated": annotated,
            "pending": pending,
            "progress": annotated / total * 100 if total > 0 else 0
        }


def main():
    st.set_page_config(page_title="Anotación de Imágenes", layout="wide")
    st.title("🏷️ Herramienta de Anotación Manual")
    st.caption("Categoriza imágenes para mejorar el modelo con fine-tuning")
    
    tool = AnnotationTool()
    
    # Sidebar con estadísticas
    with st.sidebar:
        st.markdown("### 📊 Progreso")
        stats = tool.get_annotation_stats()
        
        st.metric("Total de imágenes", stats["total"])
        st.metric("Anotadas", stats["annotated"])
        st.metric("Pendientes", stats["pending"])
        
        st.progress(stats["progress"] / 100)
        st.caption(f"{stats['progress']:.1f}% completado")
        
        st.divider()
        
        # Configuración del anotador
        st.markdown("### 👤 Anotador")
        annotator_name = st.text_input("Tu nombre", value="Anonymous")
        
        st.divider()
        
        # Instrucciones
        with st.expander("📖 Instrucciones"):
            st.markdown("""
            **Cómo anotar:**
            1. Verifica si la imagen corresponde a la ciudad indicada
            2. Si es incorrecta, especifica la ciudad correcta
            3. Evalúa la calidad (¿qué tan útil es para geolocalizar?)
            4. Marca elementos visibles (landmarks, arquitectura, etc.)
            5. Indica tu nivel de confianza
            
            **Fuentes de imágenes:**
            - `wikimedia`: Wikimedia Commons (open source)
            - `pexels`: Pexels (stock photos gratuitas)
            - `google_static`: Google Static Maps
            - `manual_import`: Imágenes que tú agregaste
            
            **Consejos:**
            - Prioriza calidad sobre cantidad
            - Sé consistente en tus criterios
            - Marca "No estoy seguro" si tienes dudas
            """)
        
        st.divider()
        
        # Filtros
        st.markdown("### 🔍 Filtros")
        available_sources = list(set(img.get("source", "unknown") for img in tool.metadata["images"]))
        filter_source = st.multiselect(
            "Fuente",
            options=available_sources,
            default=available_sources
        )
        
        # Botón para descargar anotaciones
        if st.button("💾 Descargar anotaciones"):
            st.download_button(
                label="⬇️ Descargar JSON",
                data=json.dumps(tool.annotations, indent=2, ensure_ascii=False),
                file_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Obtener imágenes pendientes
    unannotated = tool.get_unannotated_images()
    
    # Aplicar filtros
    if filter_source:
        unannotated = [img for img in unannotated if img["source"] in filter_source]
    
    if not unannotated:
        st.success("✅ ¡Todas las imágenes han sido anotadas!")
        st.balloons()
        
        # Mostrar resumen
        if tool.annotations["annotations"]:
            st.subheader("📈 Resumen de anotaciones")
            df = pd.DataFrame(tool.annotations["annotations"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Por ciudad correcta:**")
                city_counts = df["correct_city"].value_counts().head(10)
                st.dataframe(city_counts)
            
            with col2:
                st.markdown("**Por calidad:**")
                quality_counts = df["quality"].value_counts()
                st.dataframe(quality_counts)
        
        st.stop()
    
    # Mostrar imagen actual
    current_idx = st.session_state.get("current_idx", 0)
    
    if current_idx >= len(unannotated):
        current_idx = 0
        st.session_state.current_idx = 0
    
    img_data = unannotated[current_idx]
    
    st.markdown(f"### Imagen {current_idx + 1} de {len(unannotated)}")
    
    # Layout de dos columnas
    col_img, col_form = st.columns([1.2, 1])
    
    with col_img:
        try:
            img_path = Path(img_data["local_path"])
            if img_path.exists():
                img = Image.open(img_path)
                st.image(img, use_container_width=True)
            else:
                st.error(f"❌ Imagen no encontrada: {img_path}")
        except Exception as e:
            st.error(f"❌ Error cargando imagen: {e}")
        
        # Metadatos de la imagen
        with st.expander("ℹ️ Metadatos de la imagen"):
            st.json({
                "source": img_data.get("source", "unknown"),
                "photo_id": img_data.get("photo_id", ""),
                "title": img_data.get("title", ""),
                "tags": img_data.get("tags", ""),
                "city_target": img_data.get("city_target", ""),
                "state_target": img_data.get("state_target", ""),
                "url": img_data.get("url", ""),
                "description": img_data.get("description", ""),
            })
    
    with col_form:
        st.markdown("#### 📝 Anotación")
        
        with st.form(key="annotation_form"):
            st.markdown(f"**Ciudad objetivo:** {img_data['city_target']}, {img_data['state_target']}")
            
            # ¿Es correcta la ciudad?
            is_correct_city = st.radio(
                "¿La imagen corresponde a la ciudad indicada?",
                options=["Sí", "No", "No estoy seguro"],
                index=0
            )
            
            # Si no es correcta, ¿cuál es?
            correct_city = img_data["city_target"]
            correct_state = img_data["state_target"]
            
            if is_correct_city == "No":
                st.markdown("**¿Cuál es la ciudad correcta?**")
                col_c, col_s = st.columns(2)
                with col_c:
                    correct_city = st.text_input("Ciudad", value="")
                with col_s:
                    correct_state = st.text_input("Estado", value="")
            
            # Calidad de la imagen
            quality = st.select_slider(
                "Calidad de la imagen para geolocalización",
                options=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
                value="Media"
            )
            
            # Elementos visibles
            st.markdown("**Elementos visibles en la imagen:**")
            col1, col2 = st.columns(2)
            
            with col1:
                has_landmarks = st.checkbox("Monumentos/landmarks")
                has_architecture = st.checkbox("Arquitectura característica")
                has_signs = st.checkbox("Letreros/texto legible")
                has_nature = st.checkbox("Elementos naturales")
            
            with col2:
                has_people = st.checkbox("Personas")
                has_vehicles = st.checkbox("Vehículos")
                has_skyline = st.checkbox("Skyline/horizonte")
                has_street = st.checkbox("Vista de calle")
            
            # Notas adicionales
            notes = st.text_area("Notas adicionales (opcional)", height=100)
            
            # Confianza del anotador
            confidence = st.slider(
                "Tu confianza en esta anotación (%)",
                min_value=0, max_value=100, value=80, step=5
            )
            
            # Botones de acción
            col_skip, col_submit = st.columns([1, 2])
            
            with col_skip:
                skip_button = st.form_submit_button("⏭️ Saltar", use_container_width=True)
            
            with col_submit:
                submit_button = st.form_submit_button("✅ Guardar anotación", 
                                                      use_container_width=True, 
                                                      type="primary")
            
            if submit_button:
                # Crear anotación
                annotation = {
                    "local_path": img_data["local_path"],
                    "source": img_data["source"],
                    "photo_id": img_data["photo_id"],
                    "url": img_data["url"],
                    "predicted_city": img_data["city_target"],
                    "predicted_state": img_data["state_target"],
                    "is_correct": is_correct_city == "Sí",
                    "is_uncertain": is_correct_city == "No estoy seguro",
                    "correct_city": correct_city,
                    "correct_state": correct_state,
                    "quality": quality,
                    "elements": {
                        "landmarks": has_landmarks,
                        "architecture": has_architecture,
                        "signs": has_signs,
                        "nature": has_nature,
                        "people": has_people,
                        "vehicles": has_vehicles,
                        "skyline": has_skyline,
                        "street": has_street,
                    },
                    "notes": notes,
                    "annotator_confidence": confidence,
                    "annotator": annotator_name,
                    "annotated_at": datetime.now().isoformat(),
                }
                
                # Guardar
                tool.annotations["annotations"].append(annotation)
                if annotator_name not in tool.annotations["annotators"]:
                    tool.annotations["annotators"].append(annotator_name)
                
                tool.save_annotations()
                
                # Avanzar a la siguiente imagen
                st.session_state.current_idx = current_idx + 1
                st.success("✅ Anotación guardada")
                st.rerun()
            
            if skip_button:
                # Saltar a la siguiente
                st.session_state.current_idx = current_idx + 1
                st.rerun()
    
    # Navegación manual
    st.divider()
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.button("⬅️ Anterior", use_container_width=True):
            st.session_state.current_idx = max(0, current_idx - 1)
            st.rerun()
    
    with col_info:
        st.caption(f"Navegación: {current_idx + 1} / {len(unannotated)}")
    
    with col_next:
        if st.button("Siguiente ➡️", use_container_width=True):
            st.session_state.current_idx = min(len(unannotated) - 1, current_idx + 1)
            st.rerun()


if __name__ == "__main__":
    main()
