import os
import argparse

import joblib
import utils

import face_recognition

import cv2
import numpy as np


dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
models_dir = os.path.join(project_dir, "models")

DEFAULT_MODEL = "balanced_bagging_mlp_32x32_a.jl"
DEFAULT_PCA_MODEL = "pca_32x32.npy"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--pca-model", type=str, default=DEFAULT_PCA_MODEL)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--video-file", type=str, default=None, required=True)
    parser.add_argument("--blur", type=str, default="yes")

    return parser.parse_args()


def main():
    args = parse_args()

    capture = cv2.VideoCapture(args.video_file)

    # Read model and PCA model.
    clf = joblib.load(os.path.join(models_dir, args.model))
    pca = joblib.load(os.path.join(models_dir, args.pca_model))

    size = (800, 600)
    image_size = (args.image_size,) * 2

    result = cv2.VideoWriter(
        "filename.avi",
        cv2.VideoWriter_fourcc(*"MJPG"),
        10,
        size,
    )

    while True:
        # Take an image.
        success, frame = capture.read()

        if not success:
            print("Failed to load a frame.")
            break

        frame = cv2.resize(frame, size)

        # Detect faces.
        boxes = face_recognition.face_locations(frame)

        if boxes:
            # Extract faces from an image.
            faces = utils.extract_faces_from_image(frame, boxes)

            # Preprocess faces.
            faces = np.stack(
                [utils.preprocess(face, image_size, False) for face in faces]
            )

            # Reduce dimensionality.
            faces = pca.transform(faces)

            # Classify.
            probs = clf.predict_proba(faces)

            predicted = np.argmax(probs, axis=1)

            # Compute labels.

            if args.blur == "no":
                labels = ["adult" if y == 1 else "minor" for y in predicted]
                utils.label_image(frame, boxes, labels)
            else:
                frame = utils.blur_image(frame, boxes, predicted)

        cv2.imshow("Video", frame)
        result.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    result.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
