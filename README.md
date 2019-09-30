# Intracranial-Hemorrhage-Detection

Intracranial hemorrhage, bleeding that occurs inside the cranium, is a serious health problem requiring rapid and often intensive medical treatment. For example, intracranial hemorrhages account for approximately 10% of strokes in the U.S., where stroke is the fifth-leading cause of death. Identifying the location and type of any hemorrhage present is a critical step in treating the patient.

Diagnosis requires an urgent procedure. When a patient shows acute neurological symptoms such as severe headache or loss of consciousness, highly trained specialists review medical images of the patient’s cranium to look for the presence, location and type of hemorrhage. The process is complicated and often time consuming.

In this competition, your challenge is to build an algorithm to detect acute intracranial hemorrhage and its subtypes.

## Setup
- Convert Ddicom formant to .png. (Since we are dealing with CT scans its important to select window so far I have been using 40/40 for more details check `src/dicom_to_png.py`) -> After conversion png files are `512x512`
- One dicom file was corupted please run `src/00_DICOM_PNG.ipynb` to adjust train dataframe
- Resize images to `224x224` using `src/resize_pngs_fast.ipynb` for faster testing
