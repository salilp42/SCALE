import os
import shutil
import sys

def setup_colab():
    # Create project directory
    os.makedirs('ms_medvision', exist_ok=True)
    os.chdir('ms_medvision')
    
    # Unzip the package
    os.system('unzip ../ms_medvision_colab.zip')
    
    # Move files from colab_package to current directory
    for item in os.listdir('colab_package'):
        src = os.path.join('colab_package', item)
        dst = item
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    
    # Remove the temporary directory
    shutil.rmtree('colab_package')
    
    # Install requirements
    os.system('pip install scikit-learn medmnist torch torchvision seaborn tqdm matplotlib numpy scipy plotly opencv-python')
    
    # Print GPU info
    os.system('nvidia-smi')
    
    print("\nSetup complete! You can now run the training script.") 