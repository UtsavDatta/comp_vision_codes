import cv2

cap = cv2.VideoCapture("peo.mp4")  # input video file

ret, primary = cap.read()
ret, secondary = cap.read()

while cap.isOpened():

    diff = cv2.absdiff(primary, secondary)
    gray_img = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, None, iterations=7)
    cont, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cont:
        (x, y, w, h) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 2000:
            continue
        cv2.rectangle(primary, (x, y), (x + w, y + h), (255, 0, 0), 2)

    frame_show = cv2.resize(primary, (854, 480))
    cv2.imshow("Output Video", frame_show)
    primary = secondary
    ret, secondary = cap.read()

    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
