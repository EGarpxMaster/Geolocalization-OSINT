# build_model.py (v2)
import os, csv, math
from collections import defaultdict
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

CSV_PATH = "data/cities_mx.csv"
OUT_PATH = "model/modelo.pth"

# Puedes probar modelos más fuertes:
# "openai/clip-vit-large-patch14"
# "laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K"  (si usas OpenCLIP vía transformers)
MODEL_NAME = "openai/clip-vit-large-patch14"

# Plantillas bilingües; {city},{state} y {country} se sustituyen
BASE_PROMPTS = [
    "A street-level photo from {city}, {state}, {country}",
    "Urban downtown in {city}, {state}, {country}",
    "City skyline of {city}, {state}, {country}",
    "Historic colonial streets of {city}, {state}, {country}",
    "Touristic landmarks in {city}, {state}, {country}",
    "Centro histórico de {city}, {state}, {country}",
    "Zona urbana en {city}, {state}, {country}",
    "Paisaje urbano de {city}, {state}, {country}",
]

# Prompts activados por tags
TAG_PROMPTS = {
    "beach": [
        "Beachfront and resorts in {city}, {state}, {country}",
        "Playa y zona hotelera en {city}, {state}, {country}",
        "Coastal skyline of {city}, {state}, {country}",
    ],
    "skyline": [
        "Modern skyline in {city}, {state}, {country}",
    ],
    "colonial": [
        "Colonial architecture in {city}, {state}, {country}",
        "Arquitectura colonial en {city}, {state}, {country}",
    ],
    "border": [
        "Border city streets in {city}, {state}, {country}",
    ],
    "desert": [
        "Semi-arid landscape around {city}, {state}, {country}",
    ],
    "port": [
        "Port and harbor area in {city}, {state}, {country}",
    ],
    "island": [
        "Island town streets in {city}, {state}, {country}",
    ],
    "bay": [
        "Bayfront in {city}, {state}, {country}",
    ],
    "mountains": [
        "City near mountains in {city}, {state}, {country}",
    ],
}

def read_cities(csv_path):
    cities = []
    states_set = set()
    with open(csv_path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"].strip()
            state = row["state"].strip()
            lat = float(row["lat"])
            lon = float(row["lon"])
            tags = [t.strip().lower() for t in row.get("tags","").split("|") if t.strip()]
            cities.append({"name": name, "state": state, "lat": lat, "lon": lon, "tags": tags})
            states_set.add(state)
    states = sorted(states_set)
    return cities, states

def make_prompts_for_city(city):
    country = "Mexico"
    base = [p.format(city=city["name"], state=city["state"], country=country) for p in BASE_PROMPTS]
    extra = []
    for t in city["tags"]:
        if t in TAG_PROMPTS:
            extra += [p.format(city=city["name"], state=city["state"], country=country) for p in TAG_PROMPTS[t]]
    # limitar para velocidad: quitar duplicados y acotar
    prompts = list(dict.fromkeys(base + extra))
    return prompts[:12]  # tope razonable

def make_prompts_for_state(state):
    country = "Mexico"
    return [
        f"Street-level photo from {state}, {country}",
        f"Urban areas in {state}, {country}",
        f"Ciudad en {state}, {country}",
    ]

def encode_texts(model, processor, device, texts, batch=24):
    embs = []
    for i in tqdm(range(0, len(texts), batch), desc="Encoding texts"):
        batch_text = texts[i:i+batch]
        inputs = processor(text=batch_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            z = model.get_text_features(**inputs)
            z = z / z.norm(dim=-1, keepdim=True)
        embs.append(z.cpu())
    return torch.cat(embs, dim=0)

def build_and_save():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)

    cities, states = read_cities(CSV_PATH)

    # --- embeddings por ciudad (promedio multi-prompt) ---
    city_prompts = []
    city_prompt_slices = []  # por ciudad: (start, end)
    for c in cities:
        ps = make_prompts_for_city(c)
        city_prompts.extend(ps)
        city_prompt_slices.append((len(city_prompts)-len(ps), len(city_prompts)))

    city_emb_all = encode_texts(model, processor, device, city_prompts, batch=24)
    city_emb_avg = []
    for (a, b) in city_prompt_slices:
        avg = city_emb_all[a:b].mean(dim=0)
        avg = avg / avg.norm()
        city_emb_avg.append(avg.unsqueeze(0))
    city_emb_avg = torch.cat(city_emb_avg, dim=0)  # [N_cities, D]

    # --- embeddings por estado (para backoff) ---
    state_prompts = []
    state_idx = {}
    for s in states:
        ps = make_prompts_for_state(s)
        start = len(state_prompts)
        state_prompts.extend(ps)
        state_idx[s] = (start, len(state_prompts))

    state_emb_all = encode_texts(model, processor, device, state_prompts, batch=24)
    state_emb = {}
    for s, (a, b) in state_idx.items():
        avg = state_emb_all[a:b].mean(dim=0)
        avg = avg / avg.norm()
        state_emb[s] = avg

    payload = {
        "model_name": MODEL_NAME,
        "cities": cities,                 # [{name,state,lat,lon,tags}, ...]
        "city_embeds": city_emb_avg,      # [N, D]
        "states": states,
        "state_embeds": state_emb,        # dict state -> [D]
        "csv_path": CSV_PATH,
        "prompt_set": {
            "base": BASE_PROMPTS,
            "tags": TAG_PROMPTS
        }
    }
    torch.save(payload, OUT_PATH)
    print(f"✅ Guardado {OUT_PATH} | ciudades={len(cities)} estados={len(states)} dim={city_emb_avg.shape[1]}")

if __name__ == "__main__":
    build_and_save()
