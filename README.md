# Hand Gestures Classification
Full details at www.joaobrito.net/projects/hand_gestures_classification

# Dataset Stats
* Subjects: **4** (**3** for training + **1** for testing)
* Total number of instances: ***~2040*** (*90%* for training + *10%* for testing)

* Image Names:
    
    *(example)* "class_time-of-day_subject_illumination_background_distance_fframe-number.jpg"
        
    * **class** - the number represented by the gesture (**0**, **1**, **2**, **3**, **4**, **5**, **6**, **7**, **8** or **9**);

    * **time-of-day** - the time of day when the photo was taken (**m** = morning, **a** = afternoon, **n** = night);

    * **subject** - the "id" of the person to whom the hand belongs to (**0** = "person 1", **1** = "person 2", **2** = "person 3", **3** = "person 4");

    * **illumination** - the lighting conditions of the scene (**0** = "dark", **1** = "bright");

    * **background** - the backdrop of the image (**0** = "wall", **1** = "outside", **2** = "subject's body");

    * **distance** - the distance between the hand and the camera (**0** = "close", **1** = "far");

    * **fframe-number** - a number (unique identifier) assigned to the image.
