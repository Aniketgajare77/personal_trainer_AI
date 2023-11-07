import cv2
import numpy as np
import time
import PoseModule as pm

WEIGHT_KG = 70
CALORIES_PER_PUSHUP = 0.5

cap = cv2.VideoCapture(r"C:\AL_DL\personal_trainer\push up.mp4")

detector = pm.poseDetector()

count = 0
dir = 0

# Initialize fonts and colors for text and graphics
font = cv2.FONT_HERSHEY_TRIPLEX
font_scale = 0.5
font_color = (255, 255, 255)
font_thickness = 2

# Initialize the timer
start_time = time.time()
exercise_duration = 60  # Set the exercise duration in seconds (e.g., 60 seconds)

while True:
    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(exercise_duration - elapsed_time, 0)

    success, img = cap.read()

    img = detector.findPose(img, )
    lmList = detector.findPosition(img, False)

    if len(lmList) != 0:
        # Left arm angle
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (197, 291), (0, 100))
        # right arm
        R_angle = detector.findAngle(img, 10, 12, 14)

        # Update count based on arm angle
        if per == 100:
            if dir == 0:
                count += 1  # Count as one push-up when arm angle reaches 100 and the direction is up
                dir = 1
        if per == 0:
            if dir == 1:
                dir = 0

        # Calculate the center of the rectangle
        rect_x1, rect_y1 = 0, 3
        rect_x2, rect_y2 = 100, 99
        center_x = (rect_x1 + rect_x2) // 2
        center_y = (rect_y1 + rect_y2) // 2

        # Draw the rectangle and put text in the middle of the rectangle
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 155), cv2.FILLED)
        cv2.putText(img, str(int(count)), (center_x - 20, center_y + 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)

        # Display the timer
        timer_text = f"Time Left: {int(remaining_time)} seconds"
        cv2.putText(img, timer_text, (728, 18), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 1)

        calorie_burn = count * CALORIES_PER_PUSHUP
        calorie_text = f"Calories Burned: {calorie_burn:.2f} kcal"
        cv2.putText(img, calorie_text, (680, 38), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 1)


        # Add additional elements to the user interface
        # Display exercise instructions
        cv2.putText(img, "Push-Up Trainer", (0, 380), font, 1.2, font_color, 1)
        cv2.putText(img, "Instructions:", (0, 410), font, 1.1, font_color, 1)
        cv2.putText(img, "- Keep your body straight", (0, 440), font, 0.5, font_color, 1)
        cv2.putText(img, "- Touch your chest to the ground", (0, 470), font, 0.5, font_color, 1)
        cv2.putText(img, "- Push up until your arms are straight", (0, 500), font, 0.5, font_color, 1)
        cv2.putText(img, "- Perform the exercise at a steady pace", (0, 530), font, 0.5, font_color, 1)

    cv2.imshow("Image", img)

    # Check if the exercise duration is over
    if elapsed_time >= exercise_duration:
        break

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
