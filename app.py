import time
import edgeiq
"""
Simultaneously use object detection to detect human faces and classification to classify
the detected faces in terms of age groups, and output results to
shared output stream.

To change the computer vision models, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_model.html

To change the engine and accelerator, follow this guide:
https://dashboard.alwaysai.co/docs/application_development/changing_the_engine_and_accelerator.html
"""


def main():

    # Step 1b: first make a detector to detect facial objects
    facial_detector = edgeiq.ObjectDetection(
            "alwaysai/res10_300x300_ssd_iter_140000")
    facial_detector.load(engine=edgeiq.Engine.DNN)

    # Step 2a: then make a classifier to classify the age of the image
    classifier = edgeiq.Classification("alwaysai/agenet")
    classifier.load(engine=edgeiq.Engine.DNN)

    # Step 2b: descriptions printed to console
    print("Engine: {}".format(facial_detector.engine))
    print("Accelerator: {}\n".format(facial_detector.accelerator))
    print("Model:\n{}\n".format(facial_detector.model_id))

    print("Engine: {}".format(classifier.engine))
    print("Accelerator: {}\n".format(classifier.accelerator))
    print("Model:\n{}\n".format(classifier.model_id))

    fps = edgeiq.FPS()

    try:
        with edgeiq.WebcamVideoStream(cam=0) as video_stream, \
                edgeiq.Streamer() as streamer:

            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()

            # loop detection
            while True:

                # Step 3a: track how many faces are detected in a frame
                count = 1

                # read in the video stream
                frame = video_stream.read()


                # detect human faces
                results = facial_detector.detect_objects(
                        frame, confidence_level=.5)

                # Step 3b: altering the labels to show which face was detected
                for p in results.predictions:
                    p.label = "Face " + str(count)
                    count = count + 1



                # Step 3c: alter the original frame mark up to just show labels
                frame = edgeiq.markup_image(
                        frame, results.predictions, show_labels=True, show_confidences=False)

                # generate labels to display the face detections on the streamer
                text = ["Model: {}".format(facial_detector.model_id)]
                text.append(
                        "Inference time: {:1.3f} s".format(results.duration))

                # Step 3d:
                text.append("Faces:")

                # Step 4a: add a counter for the face detection label
                age_label = 1

                # append each predication to the text output
                for prediction in results.predictions:

                    # Step 4b: append labels for face detection & classification
                    text.append("Face {} ".format(
                        age_label))

                    age_label = age_label + 1

                    ## to show confidence, use the following instead of above:
                    # text.append("Face {}: detected with {:2.2f}% confidence,".format(
                        #count, prediction.confidence * 100))

                    # Step 4c: cut out the face and use for the classification
                    face_image = edgeiq.cutout_image(frame, prediction.box)

                    # Step 4d: attempt to classify the image in terms of age
                    age_results = classifier.classify_image(face_image)

                    # Step 4e: if there are predictions for age classification,
                    # generate these labels for the output stream
                    if age_results.predictions:
                        text.append("is {}".format(
                            age_results.predictions[0].label,
                        ))
                    else:
                        text.append("No age prediction")

                    ## to append classification confidence, use the following
                    ## instead of the above if/else:

                    # if age_results.predictions:
                    #     text.append("age: {}, confidence: {:.2f}\n".format(
                    #         age_results.predictions[0].label,
                    #         age_results.predictions[0].confidence))
                    # else:
                    #     text.append("No age prediction")
                
                # send the image frame and the predictions to the output stream
                streamer.send_data(frame, text)

                fps.update()

                if streamer.check_exit():
                    break

    finally:
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))

        print("Program Ending")


if __name__ == "__main__":
    main()
