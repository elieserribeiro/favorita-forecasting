"""Data loading utilities for the Favorita demand forecasting project."""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"

# Tradução das 33 famílias de produtos (para gráficos e painel)
FAMILY_PT = {
    "AUTOMOTIVE": "Automotivo", "BABY CARE": "Cuidados com bebê",
    "BEAUTY": "Beleza", "BEVERAGES": "Bebidas", "BOOKS": "Livros",
    "BREAD/BAKERY": "Padaria", "CELEBRATION": "Festas", "CLEANING": "Limpeza",
    "DAIRY": "Laticínios", "DELI": "Frios e embutidos", "EGGS": "Ovos",
    "FROZEN FOODS": "Congelados", "GROCERY I": "Mercearia I",
    "GROCERY II": "Mercearia II", "HARDWARE": "Ferragens",
    "HOME AND KITCHEN I": "Casa e cozinha I", "HOME AND KITCHEN II": "Casa e cozinha II",
    "HOME APPLIANCES": "Eletrodomésticos", "HOME CARE": "Cuidados com a casa",
    "LADIESWEAR": "Moda feminina", "LAWN AND GARDEN": "Jardinagem",
    "LINGERIE": "Lingerie", "LIQUOR,WINE,BEER": "Bebidas alcoólicas",
    "MAGAZINES": "Revistas", "MEATS": "Carnes", "PERSONAL CARE": "Higiene pessoal",
    "PET SUPPLIES": "Pet shop", "PLAYERS AND ELECTRONICS": "Eletrônicos",
    "POULTRY": "Aves", "PREPARED FOODS": "Comida pronta", "PRODUCE": "Hortifrúti",
    "SCHOOL AND OFFICE SUPPLIES": "Papelaria", "SEAFOOD": "Pescados",
}


def load_processed(proc_dir: Path = PROC_DIR) -> dict[str, pd.DataFrame]:
    """Load all processed Parquet files into a dict of DataFrames."""
    names = ["train_2017", "items", "stores", "oil", "holidays", "transactions"]
    return {name: pd.read_parquet(proc_dir / f"{name}.parquet") for name in names}


def enrich_sales(train: pd.DataFrame, items: pd.DataFrame,
                 stores: pd.DataFrame) -> pd.DataFrame:
    """Add product (family, class, perishable) and store attributes to sales.

    Text columns are converted to 'category' to keep memory usage low.
    """
    items = items.astype({"item_nbr": "int32", "family": "category",
                          "class": "int16", "perishable": "int8"})
    stores = (stores.rename(columns={"type": "store_type"})
                    .astype({"store_nbr": "int8", "city": "category",
                             "state": "category", "store_type": "category",
                             "cluster": "int8"}))
    return (train
            .merge(items[["item_nbr", "family", "class", "perishable"]],
                   on="item_nbr", how="left")
            .merge(stores[["store_nbr", "city", "state", "store_type", "cluster"]],
                   on="store_nbr", how="left"))
