# CAMP-RT
Correlations Across Multiple Patients in Radiation Therapy

## Motivation
Radiation therapy is one of the most common treatments for cancer, using high-energy radiation that can kill cancer cells and shrink tumors. However, the radiation can also damage normal cells, leading to side effects:

  - Dry mouth
  - Mouth and gum sores
  - Difficulty Swallowing
  - Stiffness in the jaw
  - Nausea
  - Tooth decay
  - Increased risk of stroke
  - Peripheral neuropathy
  - etc.

OARs (Organs at Risk): Healthy organs located in the radiation field during radiation therapy.

## GOAL
To provide an automated method to compute the similarities between multiple cancer patients and rank them accordingly. This would allow radiologists to optimize the radiation therapy plan in a short timeframe, while minimizing pain and side effects to healthy organs at risk.

## DATA
Currently, data for each patient is manually extracted from the respective DICOM files (using Slicer-RT) and encoded into JSON format. Next, the JSON file is preprocessed to calculate similarity scores and ranks between patients, which is then added to the JSON file. The JSON file is then ready to be read by the application. JSON for a single patient looks like this:

```json
{
    "id": 1,
    "name": "Patient 1",
    "organData": {
        "Brainstem": {
            "volume": 26.568,
            "meanDose": 14.8374,
            "minDose": 1.82336,
            "maxDose": 39.2479,
            "dosePerVolume": 0.558468835
        },
        "Cricopharyngeal_Muscle": {
            "volume": 2,
            "meanDose": 11.4323,
            "minDose": 5.55954,
            "maxDose": 46.3538,
            "dosePerVolume": 5.71615
        },
        "Esophagus": {
            "volume": 7.064,
            "meanDose": 33.8255,
            "minDose": 5.63062,
            "maxDose": 43.0086,
            "dosePerVolume": 4.788434315
        },
        "Extended_Oral_Cavity": {
            "volume": 155.656,
            "meanDose": 66.9469,
            "minDose": 37.7613,
            "maxDose": 78.2921,
            "dosePerVolume": 0.43009521
        },
        "Glottic_Area": {
            "volume": 0.392,
            "meanDose": 40.4576,
            "minDose": 14.8824,
            "maxDose": 62.0515,
            "dosePerVolume": 103.2081633
        },
        "IPC": {
            "volume": 2.184,
            "meanDose": 50.289,
            "minDose": 32.9462,
            "maxDose": 70.755,
            "dosePerVolume": 23.0260989
        },
        "Lower_Lip": {
            "volume": 4.16,
            "meanDose": 33.062,
            "minDose": 22.1697,
            "maxDose": 44.0036,
            "dosePerVolume": 7.947596154
        },
        "Lt_Anterior_Seg_Eyeball": {
            "volume": 0.224,
            "meanDose": 0.958014,
            "minDose": 0.874109,
            "maxDose": 1.06843,
            "dosePerVolume": 4.276848214
        },
        "Lt_Parotid_Gland": {
            "volume": 19.816,
            "meanDose": 27.6332,
            "minDose": 6.75872,
            "maxDose": 65.4652,
            "dosePerVolume": 1.394489302
        },
        "Lt_Posterior_Seg_Eyeball": {
            "volume": 7.096,
            "meanDose": 1.131,
            "minDose": 0.752475,
            "maxDose": 2.05858,
            "dosePerVolume": 0.159385569
        },
        "Lt_Submandibular_Gland": {
            "volume": 6.168,
            "meanDose": 64.3312,
            "minDose": 56.5119,
            "maxDose": 70.6549,
            "dosePerVolume": 10.42983139
        },
        "Lt_thyroid_lobe": {
            "volume": 5,
            "meanDose": 41.4774,
            "minDose": 8.74768,
            "maxDose": 53.3287,
            "dosePerVolume": 8.29548
        },
        "MPC": {
            "volume": 1.288,
            "meanDose": 69.9821,
            "minDose": 65.4769,
            "maxDose": 72.0159,
            "dosePerVolume": 54.33392857
        },
        "Pituitary_Gland": {
            "volume": 0.264,
            "meanDose": 3.66464,
            "minDose": 3.27668,
            "maxDose": 4.1969,
            "dosePerVolume": 13.88121212
        },
        "Rt_Anterior_Seg_Eyeball": {
            "volume": 0.24,
            "meanDose": 1.08952,
            "minDose": 0.953692,
            "maxDose": 1.27762,
            "dosePerVolume": 4.539666667
        },
        "Rt_Parotid_Gland": {
            "volume": 24.064,
            "meanDose": 35.8278,
            "minDose": 7.43498,
            "maxDose": 73.1618,
            "dosePerVolume": 1.488854721
        },
        "Rt_Posterior_Seg_Eyeball": {
            "volume": 7.344,
            "meanDose": 1.50401,
            "minDose": 0.865224,
            "maxDose": 2.83794,
            "dosePerVolume": 0.20479439
        },
        "Rt_Submandibular_Gland": {
            "volume": 9.968,
            "meanDose": 71.4021,
            "minDose": 68.2396,
            "maxDose": 73.7273,
            "dosePerVolume": 7.163132022
        },
        "Rt_thyroid_lobe": {
            "volume": 6.104,
            "meanDose": 54.9649,
            "minDose": 11.8696,
            "maxDose": 63.7452,
            "dosePerVolume": 9.0047346
        },
        "SPC": {
            "volume": 12.472,
            "meanDose": 70.9766,
            "minDose": 65.6312,
            "maxDose": 73.51,
            "dosePerVolume": 5.690875561
        },
        "Spinal_Cord": {
            "volume": 21.12,
            "meanDose": 24.4353,
            "minDose": 5.84861,
            "maxDose": 40.4765,
            "dosePerVolume": 1.156974432
        },
        "Supraglottic_Larynx": {
            "volume": 13.448,
            "meanDose": 66.4997,
            "minDose": 32.0927,
            "maxDose": 74.7228,
            "dosePerVolume": 4.944950922
        },
        "Thyroid_cartilage": {
            "volume": 8.256,
            "meanDose": 49.8683,
            "minDose": 8.96123,
            "maxDose": 73.3606,
            "dosePerVolume": 6.040249516
        },
        "Upper_Lip": {
            "volume": 5.32,
            "meanDose": 30.1019,
            "minDose": 14.8215,
            "maxDose": 40.8792,
            "dosePerVolume": 5.65825188
        }
    },
    "similarity": [1, 6, 9, 10, 15, 11, 13, 5, 7, 8, 3, 14, 12, 2, 4],
    "scores": [1, 0.98852, 0.97781, 0.95906, 0.85954, 0.82487, 0.81777, 0.79799, 0.72638, 0.72515, 0.67101, 0.66705, 0.65221, 0.64747, 0.63087]
}
```