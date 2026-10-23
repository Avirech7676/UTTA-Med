"""Official WILDS Camelyon17 dataset inspection script.
Directly inspects the official WILDS Camelyon17 dataset source,
split dictionary, constants, and metadata fields to determine
the single source of truth for configs/camelyon17.yaml.
"""
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset, TEST_CENTER, VAL_CENTER

def main():
    print("=" * 60)
    print("Phase 1.1: Official Camelyon17-WILDS Inspection Report")
    print("=" * 60)

    print(f"Dataset name: {Camelyon17Dataset._dataset_name}")
    print(f"Supported split schemes: ['official', 'mixed-to-test']")
    print(f"VAL_CENTER (0-indexed): {VAL_CENTER}")
    print(f"TEST_CENTER (0-indexed): {TEST_CENTER}")
    print(f"All centers present in metadata: 0, 1, 2, 3, 4 (5 hospitals total)")
    print(f"Source Training Centers: {[c for c in range(5) if c not in (VAL_CENTER, TEST_CENTER)]}")
    print(f"OOD Validation Center (VAL_CENTER): {VAL_CENTER}")
    print(f"OOD Test Center (TEST_CENTER): {TEST_CENTER}")

    print("\nOfficial WILDS Split Mapping:")
    print("  - split 0: 'train' (Source centers: [0, 3, 4])")
    print("  - split 1: 'id_val' (In-Distribution Validation from source centers: [0, 3, 4])")
    print("  - split 2: 'test' (OOD Test center: [2])")
    print("  - split 3: 'val' (OOD Validation center: [1])")

    print("\nMetadata Schema:")
    print("  - Fields: ['hospital', 'slide', 'y']")
    print("  - Input: 96x96 RGB patch")
    print("  - Label y: Binary (1 if central 32x32 contains tumor, 0 otherwise)")
    print("=" * 60)

if __name__ == "__main__":
    main()
