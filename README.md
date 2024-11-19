## Steps to set up:
1. Clone the repo and `cd` into it
2. Downloading required models
   * Download [https://drive.google.com/file/d/1Q5qXlek3HF_-f-YYE2wP01S0HBl9k9o9/view?usp=drive_link](kerasVggSigFeatures.h5) and place it in `SOURCE\vgg_finetuned_model`
   * Download [https://drive.google.com/file/d/1KJIs-XULJNG96dsgRGAnAnmhpcxqEzNr/view?usp=drive_link](bbox_regression_cnn.h5) and place it in `SOURCE\vgg_finetuned_model`
   * Download [https://drive.google.com/file/d/1JXXIeefqNzqqiD0MagerCP2nqbZ0NpwF/view?usp=drive_link](best.pt) and place it in `SOURCE\yolo_files`
   * Download [https://drive.google.com/file/d/1zP2EW8jwK7hkC3np_UOaJ-1iR_xPetRe/view?usp=sharing](latest_net_G.pth) and place it in `SOURCE/gan_files/checkpoints/gan_signdata_kaggle`
3. In `SOURCE/vgg_finetuned_model/vgg_verify.py`, go to line 50 and put your own model path as raw string /Users/akhilbabu/Documents/work/Signature-Verification/SOURCE/vgg_finetuned_model/bbox_regression_cnn.h5#L50
4. Create Conda Environment
   ```
   conda env create -f ./environment.yml
   ```
5. Activate Conda Environment
   ```
   conda activate signature-verification
   ```
6. Run Streamlit App
   ```
   streamlit run ./ui.py
   ```
