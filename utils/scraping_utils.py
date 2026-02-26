import os
import gzip
import shutil
import requests
from pathlib import Path
from multiprocessing import Pool
import xml.etree.ElementTree as ET
from utils.utils import PDB, compute_geometric_diameters

def get_empiar_metadata(empiar_id, category_filter=("micrographs - single frame",), empiar_dir='./metadata/empiar_xml', use_local=False):
    """
    Fetches EMPIAR XML and extracts all <imageSet> entries matching the specified category,
    only if headerFormat is not 'DM4'. Tries local XML if use_local is True.
    Returns a dictionary of matches keyed by index strings.
    """
    os.makedirs(empiar_dir, exist_ok=True)
    xml_filename = f"{empiar_id}.xml"
    xml_path = os.path.join(empiar_dir, xml_filename)

    # Load from local or download
    if use_local or os.path.exists(xml_path):
        if not os.path.exists(xml_path):
            print(f"XML file not found locally at {xml_path}")
            return None
        with open(xml_path, 'rb') as f:
            xml_data = f.read()
    else:
        url = f'https://ftp.ebi.ac.uk/pub/databases/emtest/empiar/headers/{empiar_id}.xml'
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch XML for EMPIAR ID {empiar_id}")
            return None
        xml_data = response.content
        with open(xml_path, 'wb') as f:
            f.write(xml_data)

    try:
        root = ET.fromstring(xml_data)

        # Extract namespace
        namespace = root.tag.split('}')[0].strip('{')
        ns = {'ns': namespace}

        image_sets = root.findall('ns:imageSet', ns)

        fields = [
            'name', 'directory', 'category',
            'micrographsFilePattern', 'pickedParticlesFilePattern',
            'pickedParticlesDirectory', 'numImagesOrTiltSeries',
            'imageWidth', 'imageHeight', 'pixelWidth', 'pixelHeight',
            'headerFormat'
        ]

        results = {}
        index = 0

        for image_set in image_sets:
            name = image_set.find('ns:name', ns)
            if name is not None and name.text.strip().lower()[:4] == 'gain':
                continue

            category = image_set.find('ns:category', ns)
            if category is not None and category.text in category_filter:
                header_format = image_set.find('ns:headerFormat', ns)
                if header_format is not None and header_format.text.strip().upper() == 'DM4':
                    continue  # Skip DM4 formats

                record = {}
                for field in fields:
                    if field in ['imageWidth', 'imageHeight', 'pixelWidth', 'pixelHeight']:
                        val = image_set.find(f'ns:dimensions/ns:{field}', ns)
                    else:
                        val = image_set.find(f'ns:{field}', ns)
                    record[field] = val.text if val is not None else ''
                
                results[str(index)] = record
                index += 1

        return results if results else None

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return None
        
def get_emdb_metadata(emd_id, emdb_dir='./metadata/emdb_xml', use_local=False):
    """
    Fetches and parses EMDB XML metadata for a given EMD ID.
    Tries local file in emdb_dir first if use_local is True; otherwise fetches from the internet.
    Returns a dictionary of structured values.
    """
    # Ensure directory exists
    os.makedirs(emdb_dir, exist_ok=True)
    xml_filename = f"emd-{emd_id}-v30.xml"
    xml_path = os.path.join(emdb_dir, xml_filename)

    # Load from local or download
    if use_local or os.path.exists(xml_path):
        if not os.path.exists(xml_path):
            print(f"XML file not found locally at {xml_path}")
            return None
        with open(xml_path, 'rb') as f:
            xml_data = f.read()
    else:
        url = f'https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emd_id}/header/{xml_filename}'
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch XML for EMD-{emd_id}")
            return None
        xml_data = response.content
        with open(xml_path, 'wb') as f:
            f.write(xml_data)

    root = ET.fromstring(xml_data)

    # Parse macromolecules
    macromolecules = []
    for pep in root.findall('.//protein_or_peptide'):
        macromolecule_id = pep.attrib.get('macromolecule_id')
        seq = pep.findtext('sequence/string', default='')
        uniprot = pep.findtext('sequence/external_references[@type="UNIPROTKB"]', default='')
        molecular_weight = pep.findtext('molecular_weight/theoretical', default='')

        macromolecules.append({
            'macromolecule_id': macromolecule_id,
            'name': pep.findtext('name', default=''),
            'sequence': seq.replace('\n', '').strip(),
            'uniprot_id': uniprot,
            'molecular_weight_MDa': float(molecular_weight) if molecular_weight else None
        })

    symmetry = root.findtext('.//applied_symmetry/point_group', default='')
    number_images_used = root.findtext('.//number_images_used', default='')
    accel_voltage = root.findtext('.//acceleration_voltage', default='')
    nominal_cs = root.findtext('.//nominal_cs', default='')
    dose = root.findtext('.//average_electron_dose_per_image', default='')
    resolution = root.findtext('.//resolution', default='')

    map_node = root.find('./map')
    if map_node is not None:
        dimensions = {
            'col': map_node.findtext('dimensions/col', ''),
            'row': map_node.findtext('dimensions/row', ''),
            'sec': map_node.findtext('dimensions/sec', ''),
        }
        cell = {
            'a': map_node.findtext('cell/a', ''),
            'b': map_node.findtext('cell/b', ''),
            'c': map_node.findtext('cell/c', ''),
        }
        pixel_spacing = {
            'x': map_node.findtext('pixel_spacing/x', ''),
            'y': map_node.findtext('pixel_spacing/y', ''),
            'z': map_node.findtext('pixel_spacing/z', ''),
        }
    else:
        dimensions = cell = pixel_spacing = {}

    pdb_id = root.findtext('.//pdb_reference/pdb_id', default='')

    return {
        'emdb_id': f'{emd_id}',
        'number_images_used': number_images_used,
        'resolution': resolution,
        'macromolecules': macromolecules,
        'symmetry': symmetry,
        'acceleration_voltage_kV': accel_voltage,
        'nominal_cs_mm': nominal_cs,
        'electron_dose_per_image': dose,
        'dimensions': dimensions,
        'cell': cell,
        'pixel_spacing': pixel_spacing,
        'pdb_id': pdb_id
    }

def download_pdb_assembly_cif(pdb_id, output_dir='.', decompress=False):
    """
    Downloads and optionally decompresses the biological assembly mmCIF file from RCSB PDB.
    
    Args:
        pdb_id (str): 4-character PDB ID.
        output_dir (str): Directory to save the file.
        decompress (bool): If True, decompress the .gz file.

    Returns:
        Path to the downloaded (and optionally decompressed) file.
    """
    pdb_id = pdb_id.lower()
    url = f'https://files.rcsb.org/pub/pdb/data/assemblies/mmCIF/all/{pdb_id}-assembly1.cif.gz'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gz_path = output_dir / f"{pdb_id}-assembly1.cif.gz"
    cif_path = gz_path.with_suffix('')  # remove .gz

    response = requests.get(url, stream=True)
    if response.status_code != 200:
        raise ValueError(f"Failed to download {url}: Status {response.status_code}")

    with open(gz_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)

def fetch_empiar_emd_metadata_and_pdb_data(empiar_id, emd_ids, category_filter, use_local, download_pdb):
    obsolete = {'7tpq':'9cth','6fbs':'6fuw'}
    empiar_dict = {}
    empiar_dict[empiar_id] = {}
    empiar_info = get_empiar_metadata(empiar_id, category_filter, use_local=use_local)
    if empiar_info is None:
        return
    empiar_dict[empiar_id]['empiar_info'] = empiar_info
    empiar_dict[empiar_id]['emd_info'] = {}
    empiar_dict[empiar_id]['pdb_info'] = {}
    for emd_id_ in emd_ids:
        scraped_info = get_emdb_metadata(emd_id_, use_local=use_local)
        if scraped_info is not None:
            empiar_dict[empiar_id]['emd_info'][emd_id_] = scraped_info
        try:
            pdb_id_ = empiar_dict[empiar_id]['emd_info'][emd_id_]['pdb_id']
        except (KeyError, TypeError):
            continue
        if len(pdb_id_):
            if pdb_id_ in obsolete.keys():
                pdb_id_ = obsolete[pdb_id_]
            pdb_path = f'./metadata/pdbs/{pdb_id_}-assembly1.cif.gz'
            if not os.path.exists(pdb_path) and download_pdb:
                download_pdb_assembly_cif(pdb_id_, './pdbs/')
            try:
                pdb = PDB(pdb_path)
                empiar_dict[empiar_id]['pdb_info'][pdb_id_] = compute_geometric_diameters(pdb.coords)
            except Exception:
                continue
    return empiar_dict

def fetch_empiar_emd_metadata_and_pdb_data_multiprocess(empiar_emd_dict, threads=10, category_filter=("micrographs - single frame",), use_local=False, download_pdb=False):
    with Pool(processes=threads) as pool:
        info = pool.starmap(fetch_empiar_emd_metadata_and_pdb_data, [(empiar_id, emd_id, category_filter, use_local, download_pdb) for empiar_id, emd_id in empiar_emd_dict.items()])
    return info
