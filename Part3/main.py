import pandas as pd
import numpy as np

RELEVANT_COLUMNS = {
    "Brand": "brand",
    "Name": "gpuModel",
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

df["pricePerHourUsdCents"] = df.apply(compute_price, axis=1)

df = df[[
    "brand",
    "gpuModel",
    "vram_gb",
    "tdp_watts",
    "fp32_tflops",
    "memory_bandwidth",
    "architecture",
    "pricePerHourUsdCents"
]]

print(df.head())
