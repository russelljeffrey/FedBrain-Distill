import subprocess
import os
import glob

def unzip_figshare():
    """
    This is a function that unzips figshare dataset. The name of the dataset after downloading is 5.

    Args:
        None.

    Returns:
        None
    """
    subprocess.run(['unzip', '5'])
    subprocess.run(['unzip', 'brainTumorDataPublic_1-766.zip'])
    subprocess.run(['unzip', 'brainTumorDataPublic_1533-2298.zip'])
    subprocess.run(['unzip', 'brainTumorDataPublic_767-1532.zip'])
    subprocess.run(['unzip', 'brainTumorDataPublic_2299-3064.zip'])

def create_figshare_directory(directory_name: str ="figshare_data"):
    """
    This is a function that creates a directory for figshare dataset.

    Args:
        directory_name (str): name of the figshare directory

    Returns:
        None
    """
    subprocess.run(['mkdir', directory_name])
    
def move_files():
    """
    This is a function that moves the dataset files to figshare_data directory.

    Args:
        None

    Returns:
        None
    """
    source_dir = os.getcwd()
    destination_dir = "figshare_data/"

    files = glob.glob(os.path.join(source_dir, '*.mat'))

    for file in files:
        file_name = os.path.basename(file)
        destination_file = os.path.join(destination_dir, file_name)
        os.rename(file, destination_file)