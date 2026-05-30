import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


RELEVANT_COLUMNS = {
    "Brand": "brand",
    "Name": "gpu_model",
    "Memory__Memory Size": "vram_gb",
    "Render Config__SM Count": "sm_count",
    "Render Config__Tensor Cores": "tensor_cores",
    "Theoretical Performance__FP32 (float)": "fp32_tflops",
    "Memory__Bandwidth": "memory_bandwidth",
    "Board Design__TDP": "tdp_watts",
    "Graphics Processor__Architecture": "architecture",
    "Graphics Card__Launch Price": "launch_price_usd"
}

df = pd.read_csv("gpu_1986-2026.csv")

df = df[list(RELEVANT_COLUMNS.keys())].rename(columns=RELEVANT_COLUMNS)

def extract_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    digits = "".join(c for c in str(x) if c.isdigit() or c == ".")
    return float(digits) if digits else np.nan

for col in ["vram_gb", "tdp_watts", "memory_bandwidth", "fp32_tflops"]:
    df[col] = df[col].apply(extract_number)

df = df.fillna({
    "tdp_watts": df["tdp_watts"].median(),
    "fp32_tflops": df["fp32_tflops"].median(),
    "vram_gb": df["vram_gb"].median()
})

def compute_price(row):
    power_cost = 0.00025 * row["tdp_watts"]
    compute_cost = 0.15 * row["fp32_tflops"]
    memory_cost = 0.000003 * row["vram_gb"]

    base = 2.0

    price = base + power_cost + compute_cost + memory_cost

    return round(price * 100, 2)

df["price_per_hour_usd_cents"] = df.apply(compute_price, axis=1)

df = df[[
    "brand",
    "gpu_model",
    "vram_gb",
    "tdp_watts",
    "fp32_tflops",
    "memory_bandwidth",
    "architecture",
    "price_per_hour_usd_cents"
]]

print(df[1000:])

def gpu_to_text(gpu):
    vram_gb = gpu["vram_gb"]

    return (
        f"{gpu['gpu_model']} GPU. "
        f"{memory_label(gpu['vram_gb'])}, {vram_gb} GB VRAM. "
        f"{gpu['fp32_tflops']} TFLOPS FP32 compute performance. "
        f"{gpu['memory_bandwidth']} GB/s memory bandwidth. "
        f"{price_label(gpu['price_per_hour_usd_cents'])}, {gpu['price_per_hour_usd_cents']} cents per hour. "
        f"{gpu['architecture']} architecture."
    )

def price_label(cents):
    if cents < df['price_per_hour_usd_cents'].quantile(0.25):
        return "cheap"
    if cents < df['price_per_hour_usd_cents'].quantile(0.5):
        return "moderately priced"
    return "expensive"

def memory_label(vram_gb):
    if vram_gb >= df['vram_gb'].quantile(0.75):
        return "very large VRAM"
    if vram_gb >= df['vram_gb'].quantile(0.5):
        return "large VRAM"
    if vram_gb >= df['vram_gb'].quantile(0.25):
        return "medium VRAM"
    return "small VRAM"


for gpu in df:
    print(gpu)

sentences = [gpu_to_text(gpu) for _, gpu in df.iterrows()]

print("\n".join(sentences))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(sentences)

query = "64 gb vram, cheap."

similarities = model.similarity(embeddings, model.encode(query))
i = int(np.argmax(similarities))

print(df['price_per_hour_usd_cents'].quantile(0.25))
print(df[(df['vram_gb'] == 64.0) & (df['price_per_hour_usd_cents'] < 450)])

print(sentences[i])
print(similarities[i])
