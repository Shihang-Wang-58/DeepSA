from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

def smiles2mw(smiles):
    """
    Calculate molecular weight
    
    Parameters:
        smiles: SMILES string
        
    Returns:
        Molecular weight or 'smiles_unvaild' (if SMILES is invalid)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        MW = Descriptors.MolWt(mol)
    except:
        MW = 'smiles_unvaild'
    return MW

def smiles2HA(smiles):
    """
    Calculate heavy atom count
    
    Parameters:
        smiles: SMILES string
        
    Returns:
        Heavy atom count or 'smiles_unvaild' (if SMILES is invalid)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        HA_num = mol.GetNumHeavyAtoms()
    except:
        HA_num = 'smiles_unvaild'
    return HA_num

def smiles2RingNum(smiles):
    """
    Calculate ring count
    
    Parameters:
        smiles: SMILES string
        
    Returns:
        Ring count or 'smiles_unvaild' (if SMILES is invalid)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        Ring_num = mol.GetRingInfo().NumRings()
    except:
        Ring_num = 'smiles_unvaild'
    return Ring_num

def GetRingSystems(mol, includeSpiro=False):
    """
    Get ring systems
    
    Parameters:
        mol: RDKit molecule object
        includeSpiro: Whether to include spiro rings
        
    Returns:
        List of ring systems
    """
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon>1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems
    return systems

def smiles2RS(smiles):
    """
    Calculate ring system count
    
    Parameters:
        smiles: SMILES string
        
    Returns:
        Ring system count or 'smiles_unvaild' (if SMILES is invalid)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        RS_num = len(GetRingSystems(mol))
    except:
        RS_num = 'smiles_unvaild'
    return RS_num

def rule_of_five(smiles):
    """
    Check if the molecule complies with Lipinski's rule of five
    
    Parameters:
        smiles: SMILES string
        
    Returns:
        1 (complies) or 0 (does not comply)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumLipinskiHBD(mol)
        hba = rdMolDescriptors.CalcNumLipinskiHBA(mol)
        nrb = Descriptors.NumRotatableBonds(mol)
        # psa = Descriptors.TPSA(mol)
        if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10 and logp >= -2 and nrb <= 10):
            return 1
        else:
            return 0
    except:
        return 'smiles_unvaild'

def gen_smiles(smiles, kekule=False, random=False):
    """
    Generate standardized SMILES
    
    Parameters:
        smiles: SMILES string
        kekule: Whether to use Kekule form
        random: Whether to generate random SMILES
        
    Returns:
        Standardized SMILES
    """
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        random_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random)
    except:
        random_smiles = smiles
    return random_smiles

def gen_canonical_smiles(smiles, kekule=False):
    """
    Generate canonical SMILES
    
    Parameters:
        smiles: SMILES string
        kekule: Whether to use Kekule form
        
    Returns:
        Canonical SMILES or 'gen_smiles_faild' (if failed)
    """
    try:   
        mol = Chem.MolFromSmiles(smiles) 
        Chem.SanitizeMol(mol)
        canonical_smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule)
    except:
        canonical_smiles = 'gen_smiles_faild'
    return canonical_smiles