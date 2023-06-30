
import numpy as np
import pickle as pkl
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# reference from: https://github.com/NTU-MedAI/S2DV
def get_ECFP(mol, radio):
    ECFPs = mol2alt_sentence(mol, radio)
    if len(ECFPs) % (radio + 1) != 0:
        ECFPs = ECFPs[:-(len(ECFPs) % (radio + 1))]
    ECFP_by_radio = list((np.array(ECFPs).reshape((int(len(ECFPs) / (radio + 1)), (radio + 1))))[:, radio])
    return ECFP_by_radio


def mol2alt_sentence(mol, radius):
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def get_sentence_vec(tokens_list, embedding, token_dict):
    sent_vec_list = []
    for tokens in tokens_list:
        feature_vec = np.zeros(512)
        n = 0
        for token in tokens:
            if token in token_dict:
                feature_vec = np.add(feature_vec, embedding[token_dict[token]])
            else:
                n += 1
        sent_vec = np.divide(feature_vec, len(tokens)-n)
        sent_vec_list.append(sent_vec)
    # print(' {} token not found in tokens'.format(n))
    return sent_vec_list


def main(smiles_path):
    text = pd.read_csv(smiles_path)
    text.reset_index(inplace=True, drop=True)
    smiles_list = np.asarray(text['SMILES'])

    ECFP_list = []
    mol_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        mol_list.append(mol)
    for mol in mol_list:
        ECFP = get_ECFP(mol, 1)
        ECFP_list.append(ECFP)
    model_root = 'DataLoader/embedding/smiles_embedding/'

    HBV_token = pkl.load(open(os.path.join(model_root, 'HBV_token.pkl'), 'rb+'))
    HBV_emb = pkl.load(open(os.path.join(model_root, 'HBV_emb.pkl'), 'rb+'))

    HBV_vec = get_sentence_vec(ECFP_list, HBV_emb, HBV_token)
    text['smiles_embedding'] = HBV_vec

    # print('size of embedding vector: ',HBV_vec[1].size)
    # print('example: ', HBV_vec[1])
    return text
