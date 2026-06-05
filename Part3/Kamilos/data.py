import pandas as pd
import numpy as np

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

gpu_data = pd.read_csv("gpu_1986-2026.csv")

gpu_data = gpu_data[list(RELEVANT_COLUMNS.keys())].rename(columns=RELEVANT_COLUMNS)

def extract_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return x
    digits = "".join(c for c in str(x) if c.isdigit() or c == ".")
    return float(digits) if digits else np.nan

def extract_memory(x):
    if pd.isna(x):
        return
    x = x.replace(",", "")
    n = x.split()[0]
    unit = x.split()[1][0]
    if unit == "K":
        return float(n) / (2**10 * 2**10)
    if unit == "M":
        return float(n) / 2**10
    if unit == "G":
        return float(n)
    if unit == "T":
        return float(n) * 2**10
    if unit in ["S", "D"]:
        return 0.0
    raise ValueError(f"AAAA {unit}, {x}")

for col in ["tdp_watts"]:
    gpu_data[col] = gpu_data[col].apply(extract_number)

for col in ["vram_gb", "memory_bandwidth", "fp32_tflops"]:
    gpu_data[col] = gpu_data[col].apply(extract_memory)

gpu_data = gpu_data.fillna({
    "tdp_watts": gpu_data["tdp_watts"].median(),
    "fp32_tflops": gpu_data["fp32_tflops"].median(),
    "vram_gb": gpu_data["vram_gb"].median()
})

def compute_price(row):
    power_cost = 0.00025 * row["tdp_watts"]
    compute_cost = 0.15 * row["fp32_tflops"] * 1e-3
    memory_cost = 0.000003 * row["vram_gb"]

    base = 2.0

    price = base + power_cost + compute_cost + memory_cost

    return round(price * 100, 2)

gpu_data["price_per_hour_usd_cents"] = gpu_data.apply(compute_price, axis=1)

gpu_data = gpu_data[[
    "brand",
    "gpu_model",
    "vram_gb",
    "tdp_watts",
    "fp32_tflops",
    "memory_bandwidth",
    "architecture",
    "price_per_hour_usd_cents"
]]

