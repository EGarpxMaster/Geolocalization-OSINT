# finetune_model.py
# Script de fine-tuning del modelo CLIP con datos anotados
# Mejora la precisión usando imágenes reales categorizadas manualmente

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModel
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict
import argparse

# Configuración
ANNOTATIONS_FILE = "data/mining/annotations.json"
CITIES_CSV = "data/cities_mx.csv"
BASE_MODEL = "openai/clip-vit-large-patch14"
OUTPUT_MODEL = "model/modelo_finetuned.pth"
CHECKPOINT_DIR = "model/checkpoints"

class GeoDataset(Dataset):
    """Dataset personalizado para fine-tuning"""
    
    def __init__(self, annotations: List[Dict], processor, filter_quality=None, filter_confidence=None):
        self.annotations = annotations
        self.processor = processor
        
        # Filtrar por calidad
        if filter_quality:
            quality_levels = ["Muy baja", "Baja", "Media", "Alta", "Muy alta"]
            min_idx = quality_levels.index(filter_quality)
            self.annotations = [a for a in self.annotations 
                              if quality_levels.index(a.get("quality", "Media")) >= min_idx]
        
        # Filtrar por confianza del anotador
        if filter_confidence:
            self.annotations = [a for a in self.annotations 
                              if a.get("annotator_confidence", 0) >= filter_confidence]
        
        # Solo usar anotaciones correctas o corregidas
        self.annotations = [a for a in self.annotations 
                          if a.get("is_correct") or (not a.get("is_uncertain"))]
        
        print(f"📊 Dataset cargado: {len(self.annotations)} imágenes después de filtros")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        
        # Cargar imagen
        img_path = Path(annotation["local_path"])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            # Retornar imagen en blanco en caso de error
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        
        # Texto de la ciudad correcta
        city = annotation["correct_city"]
        state = annotation["correct_state"]
        
        # Crear prompts variados (data augmentation textual)
        prompts = [
            f"A photo from {city}, {state}, Mexico",
            f"Urban landscape in {city}, {state}",
            f"Street-level view of {city}, Mexico",
            f"Ciudad de {city}, {state}",
        ]
        
        # Procesamiento
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "city": city,
            "state": state,
        }


class ContrastiveLoss(nn.Module):
    """Loss contrastivo para CLIP fine-tuning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_features, text_features):
        # Normalizar
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Similitud coseno
        logits = (image_features @ text_features.T) / self.temperature
        
        # Labels: diagonal (imagen i debe coincidir con texto i)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Loss bidireccional (imagen→texto y texto→imagen)
        loss_i2t = self.criterion(logits, labels)
        loss_t2i = self.criterion(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


def collate_fn(batch):
    """Custom collate para manejar múltiples prompts por imagen"""
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    
    # Aplanar todos los prompts
    input_ids = torch.cat([item["input_ids"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Entrena por una época"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_loss=True
        )
        
        # CLIP ya devuelve un loss contrastivo
        loss = outputs.loss
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validación"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_loss=True
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def finetune(
    annotations_path: str,
    output_path: str,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    min_quality: str = "Media",
    min_confidence: int = 60,
    validation_split: float = 0.15,
):
    """Ejecuta el fine-tuning completo"""
    
    # Cargar anotaciones
    print(f"📂 Cargando anotaciones desde {annotations_path}...")
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    annotations = data["annotations"]
    print(f"✅ {len(annotations)} anotaciones totales")
    
    # Cargar modelo y procesador
    print(f"🤖 Cargando modelo base: {BASE_MODEL}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Dispositivo: {device}")
    
    model = CLIPModel.from_pretrained(BASE_MODEL).to(device)
    processor = CLIPProcessor.from_pretrained(BASE_MODEL)
    
    # Crear dataset
    full_dataset = GeoDataset(
        annotations, 
        processor, 
        filter_quality=min_quality,
        filter_confidence=min_confidence
    )
    
    if len(full_dataset) < 10:
        print(f"❌ Dataset muy pequeño ({len(full_dataset)} imágenes). Necesitas más anotaciones.")
        return
    
    # Split train/val
    val_size = int(len(full_dataset) * validation_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"📊 Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0  # Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss()
    
    # Crear directorio de checkpoints
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Iniciando fine-tuning ({epochs} épocas)...\n")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch}/{epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        print(f"✅ Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"✅ Val Loss: {val_loss:.4f}")
        
        # Guardar checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, checkpoint_path)
        print(f"💾 Checkpoint guardado: {checkpoint_path}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, best_model_path)
            print(f"⭐ Mejor modelo guardado: {best_model_path}")
    
    # Guardar modelo final en formato compatible con build_model.py
    print(f"\n{'='*60}")
    print("💾 Guardando modelo fine-tuneado...")
    
    # Cargar el mejor modelo
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    
    # Guardar solo los pesos necesarios
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": BASE_MODEL,
        "finetuned": True,
        "training_info": {
            "epochs": epochs,
            "best_val_loss": best_val_loss,
            "num_training_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
        }
    }, output_path)
    
    print(f"✅ Modelo guardado: {output_path}")
    print(f"\n🎉 Fine-tuning completado exitosamente!")
    print(f"📊 Mejor Val Loss: {best_val_loss:.4f}")
    print(f"\n💡 Siguiente paso: ejecuta build_model.py con el modelo fine-tuneado")


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning del modelo CLIP")
    parser.add_argument("--annotations", type=str, default=ANNOTATIONS_FILE,
                       help="Path al archivo de anotaciones JSON")
    parser.add_argument("--output", type=str, default=OUTPUT_MODEL,
                       help="Path de salida para el modelo fine-tuneado")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--min-quality", type=str, default="Media",
                       choices=["Muy baja", "Baja", "Media", "Alta", "Muy alta"],
                       help="Calidad mínima de imágenes a usar")
    parser.add_argument("--min-confidence", type=int, default=60,
                       help="Confianza mínima del anotador (0-100)")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Proporción del dataset para validación")
    
    args = parser.parse_args()
    
    finetune(
        annotations_path=args.annotations,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        min_quality=args.min_quality,
        min_confidence=args.min_confidence,
        validation_split=args.val_split,
    )


if __name__ == "__main__":
    main()
