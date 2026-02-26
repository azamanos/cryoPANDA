import os
import re
import requests
from multiprocessing import Pool

def download_emdb_map(emdb_id, output_dir='emdb_maps'):
    """
    Downloads the EMDB map file for a given EMDB ID.

    Parameters:
        emdb_id (str): The EMDB ID (e.g., 'EMD-22314').
        output_dir (str): Directory to save the downloaded map file.

    Returns:
        str: Path to the downloaded map file, or None if download failed.
    """
    # Ensure the EMDB ID is in the correct format
    emdb_id = emdb_id.upper()
    # Construct the download URL
    url = f'https://files.rcsb.org/pub/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz'
    # Define the path to save the downloaded file
    output_path = os.path.join(output_dir, f'{emdb_id}.map.gz')
    if os.path.exists(output_path):
        return
    try:
        # Send a GET request to download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Write the content to the output file
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Downloaded {emdb_id} to {output_path}")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"Failed to download {emdb_id}: {e}")
        return None

def download_emdb_maps_multiprocess(emd_list, output_dir='emds/', threads=8, only_the_list=False):
    """
    Downloads protein structures from PDB by utilizing different cpu threads.
    """

    pool = Pool(processes=threads)
    download = pool.starmap(download_emdb_map, [(emd_id, output_dir) for emd_id in emd_list])
    pool.close()
    pool.join()
    return