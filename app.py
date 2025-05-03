#   /Website
#       app.py
#       /templates
#               index.html
#       /static
#           /css
#               style.css
#           /image
#               logo.png

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


model_lgbm = joblib.load('lgbm_descriptor.joblib')


app = Flask(__name__)

app.secret_key = 'pkl'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'txt','csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_mol_image_with_custom_bg(mol, bg_rgb=(173, 216, 230)):  # #add8e6
    bg_rgb_normalized = tuple([c / 255.0 for c in bg_rgb])

    d2d = Draw.MolDraw2DCairo(300, 300)
    d2d.drawOptions().setBackgroundColour(bg_rgb_normalized)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()


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
            # img = Draw.MolToImage(mole)
            # buffered = BytesIO()
            # img.save(buffered, format="PNG")
            # img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            img_bytes = get_mol_image_with_custom_bg(mole)
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        try:
            expected_features = [
            "MaxAbsEStateIndex", "MinAbsEStateIndex", "MinEStateIndex", "SPS", "MolWt",
            "BCUT2D_MWHI", "BCUT2D_MRHI", "AvgIpc", "BalabanJ", "HallKierAlpha", "Ipc", "Kappa1",
            "PEOE_VSA1", "PEOE_VSA10", "PEOE_VSA11", "PEOE_VSA12", "PEOE_VSA13", "PEOE_VSA14",
            "PEOE_VSA2", "PEOE_VSA3", "PEOE_VSA4", "PEOE_VSA5", "PEOE_VSA6", "PEOE_VSA7", "PEOE_VSA8", "PEOE_VSA9",
            "SMR_VSA10", "SMR_VSA2", "SMR_VSA3", "SMR_VSA4", "SMR_VSA5", "SMR_VSA6", "SMR_VSA7", "SMR_VSA9",
            "SlogP_VSA1", "SlogP_VSA10", "SlogP_VSA11", "SlogP_VSA12", "SlogP_VSA2", "SlogP_VSA3",
            "SlogP_VSA4", "SlogP_VSA7", "SlogP_VSA8", "TPSA",
            "EState_VSA10", "EState_VSA11", "EState_VSA2", "EState_VSA3", "EState_VSA4",
            "EState_VSA5", "EState_VSA6", "EState_VSA7", "EState_VSA8", "EState_VSA9",
            "VSA_EState10", "VSA_EState2", "VSA_EState3", "VSA_EState4", "VSA_EState5",
            "VSA_EState6", "VSA_EState7", "VSA_EState8", "VSA_EState9",
            "NHOHCount", "NumAliphaticCarbocycles", "NumAliphaticHeterocycles", "NumAliphaticRings",
            "NumAmideBonds", "NumAromaticCarbocycles", "NumAromaticHeterocycles", "NumAromaticRings",
            "NumAtomStereoCenters", "NumBridgeheadAtoms", "NumHeterocycles", "NumSaturatedCarbocycles",
            "NumSaturatedHeterocycles", "NumSaturatedRings", "NumUnspecifiedAtomStereoCenters",
            "RingCount", "MolLogP", "fr_Al_COO", "fr_Al_OH", "fr_Al_OH_noTert", "fr_Ar_N", "fr_Ar_NH", "fr_Ar_OH",
            "fr_COO", "fr_NH0", "fr_NH1", "fr_NH2", "fr_Ndealkylation1", "fr_Ndealkylation2",
            "fr_alkyl_halide", "fr_allylic_oxid", "fr_aniline", "fr_aryl_methyl", "fr_bicyclic",
            "fr_ester", "fr_ether", "fr_halogen", "fr_ketone", "fr_methoxy", "fr_para_hydroxylation",
            "fr_piperdine", "fr_pyridine", "fr_sulfonamd"]
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
                "Predicted Activity Against SARS-CoV-2": int(predictions[0][0]),
                "Drug Likeness": int(predictions[0][0])
            }
        except Exception as e:
            print(f"An error occurred: {e}")
            result = {"error": "An error occurred while processing your SMILE string."}

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

