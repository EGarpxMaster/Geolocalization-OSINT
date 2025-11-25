"""
VALIDADOR DE ESTRUCTURA DEL PROYECTO
=====================================
Verifica que todo esté listo para entrenamiento y despliegue.
"""

import sys
from pathlib import Path
import json
import csv
import torch

def check_emoji(passed: bool, msg: str):
    """Imprime resultado con emoji"""
    emoji = "✅" if passed else "❌"
    print(f"{emoji} {msg}")
    return passed

def validate_structure():
    """Valida la estructura completa del proyecto"""
    
    print("\n" + "="*70)
    print("🔍 VALIDACIÓN DE ESTRUCTURA DEL PROYECTO")
    print("="*70 + "\n")
    
    all_checks = []
    
    # ========== ARCHIVOS PRINCIPALES ==========
    print("📁 ARCHIVOS PRINCIPALES:")
    all_checks.append(check_emoji(
        Path("Geolocalizador.py").exists(),
        "Interfaz principal de despliegue (Geolocalizador.py)"
    ))
    all_checks.append(check_emoji(
        Path("mining_pipeline.py").exists(),
        "Pipeline de minería de imágenes (mining_pipeline.py)"
    ))
    all_checks.append(check_emoji(
        Path("training_pipeline.py").exists(),
        "Pipeline de entrenamiento y anotación (training_pipeline.py)"
    ))
    all_checks.append(check_emoji(
        Path("build_model.py").exists(),
        "Script para construir modelo base (build_model.py)"
    ))
    all_checks.append(check_emoji(
        Path("requirements.txt").exists(),
        "Archivo de dependencias (requirements.txt)"
    ))
    
    # ========== DATOS ==========
    print("\n📊 ESTRUCTURA DE DATOS:")
    all_checks.append(check_emoji(
        Path("data").exists(),
        "Directorio de datos (data/)"
    ))
    all_checks.append(check_emoji(
        Path("data/cities_mx.csv").exists(),
        "Base de datos de ciudades (data/cities_mx.csv)"
    ))
    
    # Verificar CSV de ciudades
    if Path("data/cities_mx.csv").exists():
        with open("data/cities_mx.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            cities = list(reader)
            all_checks.append(check_emoji(
                len(cities) > 0,
                f"  └─ Ciudades cargadas: {len(cities)} ciudades"
            ))
            # Verificar columnas necesarias
            required_cols = ['name', 'state', 'lat', 'lon']
            has_cols = all(col in cities[0] for col in required_cols)
            all_checks.append(check_emoji(
                has_cols,
                f"  └─ Columnas requeridas: {', '.join(required_cols)}"
            ))
    
    all_checks.append(check_emoji(
        Path("data/mining").exists(),
        "Directorio de minería (data/mining/)"
    ))
    all_checks.append(check_emoji(
        Path("data/mining/images").exists(),
        "Directorio de imágenes (data/mining/images/)"
    ))
    
    # ========== MODELO ==========
    print("\n🤖 MODELO:")
    all_checks.append(check_emoji(
        Path("model").exists(),
        "Directorio del modelo (model/)"
    ))
    all_checks.append(check_emoji(
        Path("model/checkpoints").exists(),
        "Directorio de checkpoints (model/checkpoints/)"
    ))
    
    modelo_exists = Path("model/modelo.pth").exists()
    all_checks.append(check_emoji(
        modelo_exists,
        "Modelo base generado (model/modelo.pth)"
    ))
    
    if modelo_exists:
        try:
            modelo = torch.load("model/modelo.pth", map_location="cpu", weights_only=False)
            all_checks.append(check_emoji(
                'city_embeds' in modelo,
                "  └─ Contiene embeddings de ciudades"
            ))
            all_checks.append(check_emoji(
                'cities' in modelo,
                "  └─ Contiene lista de ciudades"
            ))
            all_checks.append(check_emoji(
                'model_name' in modelo,
                f"  └─ Modelo CLIP: {modelo.get('model_name', 'desconocido')}"
            ))
            
            num_cities = len(modelo.get('cities', []))
            all_checks.append(check_emoji(
                num_cities > 0,
                f"  └─ Número de ciudades en modelo: {num_cities}"
            ))
        except Exception as e:
            all_checks.append(check_emoji(
                False,
                f"  └─ Error cargando modelo: {e}"
            ))
    
    # ========== METADATA DE MINERÍA ==========
    print("\n⛏️ METADATA DE MINERÍA:")
    
    has_csv = Path("data/mining/metadata.csv").exists()
    has_json = Path("data/mining/metadata.json").exists()
    
    all_checks.append(check_emoji(
        has_csv or has_json,
        "Metadata de imágenes minadas (CSV o JSON)"
    ))
    
    if has_csv:
        with open("data/mining/metadata.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            images = list(reader)
            all_checks.append(check_emoji(
                len(images) > 0,
                f"  └─ CSV: {len(images)} imágenes registradas"
            ))
    
    if has_json:
        with open("data/mining/metadata.json", 'r', encoding='utf-8') as f:
            metadata = json.load(f)
            num_imgs = len(metadata.get('images', []))
            all_checks.append(check_emoji(
                num_imgs > 0,
                f"  └─ JSON: {num_imgs} imágenes registradas"
            ))
    
    # Verificar imágenes físicas
    images_path = Path("data/mining/images")
    if images_path.exists():
        image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.png"))
        all_checks.append(check_emoji(
            len(image_files) > 0,
            f"  └─ Imágenes descargadas: {len(image_files)} archivos"
        ))
    else:
        all_checks.append(check_emoji(
            False,
            "  └─ No hay imágenes descargadas aún"
        ))
    
    # ========== ANOTACIONES ==========
    print("\n✏️ ANOTACIONES:")
    
    has_annotations_csv = Path("data/mining/annotations.csv").exists()
    has_annotations_json = Path("data/mining/annotations.json").exists()
    
    all_checks.append(check_emoji(
        has_annotations_csv or has_annotations_json,
        "Archivo de anotaciones (CSV o JSON)"
    ))
    
    if has_annotations_csv:
        with open("data/mining/annotations.csv", 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            annotations = list(reader)
            all_checks.append(check_emoji(
                len(annotations) >= 50,
                f"  └─ CSV: {len(annotations)} anotaciones (mínimo 50 recomendado)"
            ))
    
    if has_annotations_json:
        with open("data/mining/annotations.json", 'r', encoding='utf-8') as f:
            ann_data = json.load(f)
            num_ann = len(ann_data.get('images', []))
            all_checks.append(check_emoji(
                num_ann >= 50,
                f"  └─ JSON: {num_ann} anotaciones (mínimo 50 recomendado)"
            ))
    
    # ========== MODELO FINE-TUNED ==========
    print("\n🎓 MODELO FINE-TUNED:")
    
    finetuned_exists = Path("model/modelo_finetuned.pth").exists()
    all_checks.append(check_emoji(
        finetuned_exists,
        "Modelo fine-tuned (model/modelo_finetuned.pth)"
    ))
    
    if finetuned_exists:
        try:
            # Verificar que sea un estado de modelo válido
            state_dict = torch.load("model/modelo_finetuned.pth", map_location="cpu", weights_only=False)
            all_checks.append(check_emoji(
                isinstance(state_dict, dict),
                "  └─ Archivo de modelo válido"
            ))
        except Exception as e:
            all_checks.append(check_emoji(
                False,
                f"  └─ Error cargando: {e}"
            ))
    
    # ========== DEPENDENCIAS ==========
    print("\n📦 DEPENDENCIAS:")
    
    try:
        import streamlit
        all_checks.append(check_emoji(True, f"Streamlit v{streamlit.__version__}"))
    except ImportError:
        all_checks.append(check_emoji(False, "Streamlit (NO INSTALADO)"))
    
    try:
        import torch
        all_checks.append(check_emoji(True, f"PyTorch v{torch.__version__}"))
    except ImportError:
        all_checks.append(check_emoji(False, "PyTorch (NO INSTALADO)"))
    
    try:
        import transformers
        all_checks.append(check_emoji(True, f"Transformers v{transformers.__version__}"))
    except ImportError:
        all_checks.append(check_emoji(False, "Transformers (NO INSTALADO)"))
    
    try:
        from PIL import Image
        all_checks.append(check_emoji(True, "Pillow (PIL)"))
    except ImportError:
        all_checks.append(check_emoji(False, "Pillow (NO INSTALADO)"))
    
    try:
        import pandas
        all_checks.append(check_emoji(True, f"Pandas v{pandas.__version__}"))
    except ImportError:
        all_checks.append(check_emoji(False, "Pandas (NO INSTALADO)"))
    
    try:
        import matplotlib
        all_checks.append(check_emoji(True, f"Matplotlib v{matplotlib.__version__}"))
    except ImportError:
        all_checks.append(check_emoji(False, "Matplotlib (NO INSTALADO)"))
    
    # ========== RESUMEN FINAL ==========
    print("\n" + "="*70)
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100 if total > 0 else 0
    
    print(f"📊 RESUMEN: {passed}/{total} verificaciones pasadas ({percentage:.1f}%)")
    print("="*70 + "\n")
    
    # ========== ESTADO DEL PROYECTO ==========
    print("🎯 ESTADO DEL PROYECTO:\n")
    
    if not has_csv and not has_json:
        print("🔴 FASE 1: MINERÍA DE IMÁGENES")
        print("   └─ Acción: Ejecutar 'python mining_pipeline.py --mode all --images 20'")
        print("   └─ Estado: Pendiente")
    elif len(image_files) == 0:
        print("🟡 FASE 1: MINERÍA EN PROGRESO")
        print("   └─ Metadata creado pero sin imágenes descargadas")
        print("   └─ Acción: Completar minado de imágenes")
    elif not has_annotations_csv and not has_annotations_json:
        print("🟡 FASE 2: ANOTACIÓN DE IMÁGENES")
        print("   └─ Acción: Ejecutar 'streamlit run training_pipeline.py'")
        print("   └─ Modo: 📝 Anotación (mínimo 50 imágenes)")
        print("   └─ Estado: Pendiente")
    elif not finetuned_exists:
        print("🟡 FASE 3: FINE-TUNING DEL MODELO")
        print("   └─ Acción: Usar modo 🔬 Fine-tuning en training_pipeline.py")
        print("   └─ Estado: Pendiente")
    elif finetuned_exists and modelo_exists:
        print("🟢 FASE 4: LISTO PARA DESPLIEGUE")
        print("   └─ Modelo base: ✅")
        print("   └─ Modelo fine-tuned: ✅")
        print("   └─ Acción: Ejecutar 'streamlit run Geolocalizador.py'")
        print("   └─ Estado: COMPLETO ✨")
    
    print("\n" + "="*70)
    
    return passed == total

if __name__ == "__main__":
    success = validate_structure()
    sys.exit(0 if success else 1)
