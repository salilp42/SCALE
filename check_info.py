import medmnist
from medmnist import INFO

print("Dataset Info Structure:")
for key in INFO.keys():
    print(f"\n{key}:")
    for info_key, info_value in INFO[key].items():
        print(f"  {info_key}: {info_value}") 