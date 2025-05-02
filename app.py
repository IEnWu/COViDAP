#   /Website
#       app.py
#       /templates
#               index.html
#       /static
#           /css
#               style.css
#           /image
#               logo.png
#       /uploads
#           csv generate output: DeepSP_descriptors.csv
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors

from flask import Flask, request, render_template, redirect, url_for,jsonify
import os
from urllib.parse import quote as url_quote

import base64
from io import BytesIO
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from lightgbm import LGBMClassifier

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score


model_lgbm = joblib.load('lgbm_descriptor.joblib')


app = Flask(__name__)

app.secret_key = 'pkl'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt','csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    descriptor_values = {}
    for descriptor, descriptor_func in Descriptors.descList:
        try:
            value = descriptor_func(mol)
            descriptor_values[descriptor] = value
        except Exception as e:
            print(f"Error computing descriptor {descriptor}: {e}")
    return descriptor_values

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET','POST'])

def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])

def upload_file():
    result = {}
    if request.method == 'POST':
        smile = request.form['smile_name']
        mole = Chem.MolFromSmiles(smile)
        if mole is not None:
            img = Draw.MolToImage(mole)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        try:
            expected_features = ['MaxAbsEStateIndex', 'MaxEStateIndex', 'MinAbsEStateIndex', 'MinEStateIndex',
 'qed', 'SPS', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
 'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge',
 'MaxAbsPartialCharge', 'MinAbsPartialCharge', 'FpDensityMorgan1',
 'FpDensityMorgan2', 'FpDensityMorgan3', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW',
 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n',
 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n',
 'Chi4v', 'HallKierAlpha', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA',
 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3' ,'PEOE_VSA4','PEOE_VSA5', 'PEOE_VSA6',
 'PEOE_VSA7', 'PEOE_VSA8','PEOE_VSA9' ,'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2',
 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5' ,'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8',
 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10' ,'SlogP_VSA11' ,'SlogP_VSA12',
 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4' ,'SlogP_VSA5', 'SlogP_VSA6',
 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA' ,'EState_VSA1',
 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3' ,'EState_VSA4',
 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8' ,'EState_VSA9',
 'VSA_EState1', 'VSA_EState10', 'VSA_EState2' ,'VSA_EState3' ,'VSA_EState4',
 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9',
 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
 'NumAromaticCarbocycles', 'NumAromaticHeterocycles' ,'NumAromaticRings',
 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles' ,'NumSaturatedRings',
 'RingCount', 'MolLogP' ,'MolMR', 'fr_Al_COO' ,'fr_Al_OH', 'fr_Al_OH_noTert',
 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N' ,'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole',
 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate' ,'fr_alkyl_halide',
 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
 'fr_azide', 'fr_azo', 'fr_barbitur' ,'fr_benzene', 'fr_benzodiazepine',
 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine' ,'fr_epoxide' ,'fr_ester',
 'fr_ether', 'fr_furan' ,'fr_guanido', 'fr_halogen', 'fr_hdrzine' ,'fr_hdrzone',
 'fr_imidazole', 'fr_imide', 'fr_isocyan' ,'fr_isothiocyan' ,'fr_ketone',
 'fr_ketone_Topliss', 'fr_lactam' ,'fr_lactone', 'fr_methoxy' ,'fr_morpholine',
 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
 'fr_nitroso', 'fr_oxazole' ,'fr_oxime' ,'fr_para_hydroxylation', 'fr_phenol',
 'fr_phenol_noOrthoHbond','fr_phos_acid' ,'fr_phos_ester', 'fr_piperdine',
 'fr_piperzine', 'fr_priamide' ,'fr_prisulfonamd' ,'fr_pyridine', 'fr_quatN',
 'fr_sulfide', 'fr_sulfonamd' ,'fr_sulfone' ,'fr_term_acetylene',
 'fr_tetrazole', 'fr_thiazole' ,'fr_thiocyan' ,'fr_thiophene',
 'fr_unbrch_alkane', 'fr_urea']
            X = compute_descriptors(smile)
            X_df = pd.DataFrame([X], columns=expected_features)
            
            X_df.fillna(0, inplace=True)

            X_values = np.array(list(X.values())).reshape(1, -1)

            if np.isnan(X_values).any():
                predictions = [None, None, None, None]
            else:
                predictions = [
                    model_lgbm.predict(X_df),  # Load these models appropriately
                ]
            result = {
                "SMILE": smile,
                "Structure": f"data:image/png;base64,{img_base64}",
                "Predicted Activity Against SARS-CoV-2": predictions[0],
                "Drug Likeness": predictions[0],
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            result = {"error": "An error occurred while processing your SMILE string."}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

