from model import extract_gpu_criteria
from data import gpu_data

def rank( query:str ):
    crit = extract_gpu_criteria(query)

    scores = []
    for _, gpu in gpu_data.iterrows():
        score = calc_score(gpu, crit)
        scores.append(score)

    sorted_df = gpu_data . copy (  )
    sorted_df['_key'] = scores
    sorted_df = sorted_df.sort_values(by='_key', ascending=False).drop(columns='_key')
    return sorted_df


def calc_score( gpu, crit ) -> float:
    score = 0.0

    brand_want = crit["brand"].lower()
    brand_got = gpu["brand"].lower()
    if brand_want == brand_got:
        score += 1.0

    ram_want = float(crit["min_vram_gb"])
    ram_got = float(gpu["vram_gb"])
    if ram_got >= ram_want:
        score += ram_want / ram_got

    cheap_want = crit["is_cheap"]
    price = gpu["price_per_hour_usd_cents"]
    if cheap_want:
        score += min(0.5, 1.0 / price * 1e2)

    return score

