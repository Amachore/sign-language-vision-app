REALTIME SIGN AND EMOTION CAMERA APP
====================================

Data source rule
----------------
Existing online files under models/ASL/ and models/FSL-105/ are reference-only.
Your own dataset clips and manifest are kept separately under
models/personal_dataset/. The trainer reads only that personal manifest.
recordings/ is reserved for final outputs from using the application.

Requirements
------------
Use Python 3.10 or 3.11 on Windows. The app uses the local webcam, MediaPipe,
scikit-learn, DeepFace, and pyttsx3. The first DeepFace emotion inference may
download its pretrained model weights. Camera frames and emotion inference are
processed locally.

Install from the project directory:

	python -m venv .venv
	.venv\Scripts\activate
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

Make and label your dataset
---------------------------
Run the recorder:

	python -m src.record_dataset

Enter exactly three sign labels for ASL and three for FSL, then enter a signer
ID. These are your annotations: choose the label that matches the sign you
intend to demonstrate, and select that label before recording each clip.
Clips and models/personal_dataset/dataset_manifest.csv are saved under the
personal dataset folder. The existing online datasets and recordings/ folder
are not used for training or changed by the dataset recorder.

In the recorder window, A selects ASL, F selects FSL, 1-3 selects one of that
language's labels, and R toggles recording: press once to start and again to
save that clip and return to the preview. Only Q exits the recorder. Keep each
clip at least 12 frames long. The selected label and signer ID are put in the
filename and manifest. Each fifth clip for each label is assigned to the
validation split; the others go to training. This is a clip-level split, so
for a stronger evaluation record multiple sessions and, when possible, more
than one signer. Aim for at least 15 clips per label.

The manifest's sign_class column is the authoritative label. To correct a
label after recording, edit that field in
models/personal_dataset/dataset_manifest.csv to one of the same three labels
for that language. Do not edit the online files under models/ for this workflow.

Train your models after recording:

	python -m src.train_sign_models --language all

Training requires exactly three recorded labels per language, with at least
three usable training clips and one usable validation clip per label. It prints
validation accuracy and writes outputs/asl_sign_model.joblib and
outputs/fsl_sign_model.joblib. If accuracy is poor, record more varied clips
with consistent labels and train again. These models recognize only your
three recorded classes; they are not general sign-language translators.

Run the application:

	python -m src.main

Controls
--------
M cycles through POSES, EMOTION, ASL, and FSL. Keys 1-4 select a mode directly.
R resets session metrics and histories. E writes timestamped CSV/JSON reports
under reports/ and the recognized sign transcript under recordings/. Q or Escape
exits.

POSES recognizes OPEN_PALM, FIST, PEACE, POINT, and THUMBS_UP using explicit
landmark rules. Pose and sign events require five stable frames. In ASL/FSL,
the app accepts only the three labels in that language's trained model and
reports low-confidence or unsupported input as Unknown. A class learned to
require both hands is rejected when only one hand is detected. Accepted signs
are timestamped, added to the visible text history, and spoken locally when
pyttsx3 is available.

Emotion inference is asynchronous so it does not block camera display. The app
marks predictions below its confidence threshold as uncertain and reports
session duration, valid observations, emotion counts/percentages, dominant
emotion, and mean confidence in EMOTION mode. Face detection alone is not
counted as an emotion observation.

Validation
----------
Run the unit tests with:

	python -m unittest discover -s tests